"""
concat_vqvae_caches.py

Concatenates the stage-2 VQ-VAE crop caches of many experiments into one
`.pt` + one `.parquet`, keeping every row traceable to the experiment it came
from.

A crop cache is a pair written by SMRFXMAnalysis alongside the CELLGROUPED hdf5:

    <base_id>_CELLGROUPED.pt               {'bf_u8': uint8 (N, 1, S, S), ...}
    <base_id>_CELLGROUPED_metadata.parquet one row per crop

The two are joined by position and nothing else — parquet row i describes tensor
row i. This script therefore advances both in lockstep from a single ordered list
of sources, so the outputs cannot drift, and it adds three columns to every row:

    experiment    the superdir the row came from
    sample_name   the sample subdir the row came from
    cache_row     the row's index into the concatenated tensor

Caches are found under the layout the other aggregators in this repo walk:

    <superdir>/<sample>/<YYYYMMDD.HHMMSS>_imaging_fxm_results/

Runs from before SMRFXMAnalysis dropped its stage1/stage2 split instead nest
this under a stage2_analysis/ subdirectory; both layouts are checked. The
newest run directory wins when a sample has been reprocessed.

No PyTorch is required, and no file is ever read whole. torch.save writes an
uncompressed zip whose storage blobs are 64-byte aligned, so for a contiguous
tensor the blob is exactly the C-order buffer and concatenating along axis 0 is
byte concatenation. Blobs are streamed from input to output a chunk at a time,
which keeps memory flat regardless of how large the merge is. Tensors that are
not byte-concatenable (non-contiguous, offset into a larger storage) are detected
and skipped rather than silently corrupted.

Usage:
    python concat_vqvae_caches.py <superdir> [<superdir> ...] [--from-file FILE]
                                  [--output DIR] [--keys KEYS] [--require-fl]
                                  [--dry-run]

    <superdir>      One or more experiment superdirs to scan.
    --from-file     Read superdirs from a file, one path per line.
    --output DIR    Parent directory for the output folder
                    (default: the first superdir).
    --keys KEYS     Comma-separated tensor keys to merge
                    (default: those present in every source).
    --require-fl    Fail if any source lacks fl_u8, instead of dropping it.
    --dry-run       Discover and validate, print the plan, write nothing.

compile_experiment.py imports this module to write the same pair into its own
`*_compiled/` output dir for the samples it discovered, via
`_sample_cache` -> `plan_concat` -> `run_concat`. Keep those three importable
without side effects.
"""
import argparse
import io
import pickle
import re
import sys
import types
import zipfile
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from fsutil import is_appledouble

# The torch-free .pt reader, already validated by browse_pt.py. Imported rather
# than duplicated so the two stay in agreement about the archive format.
from browse_pt import _STORAGE_ITEMSIZE, _TensorInfo, _read_pt, sibling_path


_CHUNK_BYTES  = 1 << 20   # streaming copy size
_ZIP_ALIGN    = 64        # torch aligns storage blobs to 64-byte boundaries
_BATCH_ROWS   = 8_192     # parquet rows per streamed batch

_RUN_SUFFIX   = '_imaging_fxm_results'
_STAGE2_DIR   = 'stage2_analysis'
_CACHE_SUFFIX = '_CELLGROUPED.pt'

# Output file names. Public because compile_experiment.py writes the same pair
# into its own output dir through concat_caches(), and the two must agree.
OUT_PT_NAME      = 'concat_vqvae_cache.pt'
OUT_PARQUET_NAME = 'concat_vqvae_cache_metadata.parquet'

# Columns this script adds. A source column of the same name is renamed rather
# than overwritten, so nothing upstream is ever lost.
_LABEL_EXPERIMENT = 'experiment'
_LABEL_SAMPLE     = 'sample_name'
_LABEL_ROW        = 'cache_row'
_ADDED_COLUMNS    = (_LABEL_EXPERIMENT, _LABEL_SAMPLE, _LABEL_ROW)

# Matches the timestamp prefix on a run directory. SMRFXMAnalysis's
# imaging_fxm_results dirs are dot-separated (YYYYMMDD.HHMMSS) only -- there is
# no underscore-separated form in the wild for this suffix.
_TS_PATTERN = re.compile(r'^(\d{8}\.\d{6})_')

_RUN_DIR_PATTERN = re.compile(re.escape(_RUN_SUFFIX) + r'$')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Concatenate stage-2 VQ-VAE crop caches across experiments.')
    parser.add_argument('superdirs', type=str, nargs='*',
                        help='Experiment superdirs to scan')
    parser.add_argument('--from-file', type=str, default=None,
                        help='Read superdirs from a file, one path per line')
    parser.add_argument('--output', type=str, default=None,
                        help='Parent dir for the output folder (default: first superdir)')
    parser.add_argument('--keys', type=str, default=None,
                        help='Comma-separated tensor keys to merge '
                             '(default: keys present in every source)')
    parser.add_argument('--require-fl', action='store_true',
                        help='Fail if any source lacks fl_u8 rather than dropping it')
    parser.add_argument('--dry-run', action='store_true',
                        help='Discover and validate only; write nothing')
    return parser.parse_args()


def _resolve_dirs(args: argparse.Namespace) -> list[Path]:
    """
    Turn the positional superdirs or --from-file into validated Paths.

    Follows aggregate_bm_vol_files.py: exactly one of the two forms must be
    given, and every entry must exist.
    """
    if args.from_file and args.superdirs:
        raise ValueError('Give either superdir arguments or --from-file, not both')
    if not args.from_file and not args.superdirs:
        raise ValueError('No superdirs given (pass paths or --from-file)')

    if args.from_file:
        listing = Path(args.from_file)
        if not listing.is_file():
            raise FileNotFoundError(f'File not found: {listing}')
        raw = []
        for line in listing.read_text().splitlines():
            line = line.strip()
            # Blank lines and comments let the file double as a lab notebook.
            if line and not line.startswith('#'):
                raw.append(line)
    else:
        raw = list(args.superdirs)

    dirs = []
    for entry in raw:
        p = Path(entry)
        if not p.is_dir():
            raise FileNotFoundError(f'Directory not found: {p}')
        dirs.append(p)
    return dirs


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

class CacheSource:
    """One cache pair, with the labels its rows will carry."""

    def __init__(self, pt_path: Path, parquet_path: Path, experiment: str,
                 sample_name: str, run_timestamp: str):
        self.pt_path       = pt_path
        self.parquet_path  = parquet_path
        self.experiment    = experiment
        self.sample_name   = sample_name
        self.run_timestamp = run_timestamp
        # Filled in by inspect_source.
        self.tensors: dict[str, _TensorInfo] = {}
        self.nrows: int = 0
        self.columns: list[str] = []

    @property
    def label(self) -> str:
        return f'{self.experiment}/{self.sample_name}'


def _last_matching_dir(parent: Path, pattern: re.Pattern) -> Path | None:
    """
    The lexicographically last subdir whose name matches pattern.

    Same logic as compile_experiment.py:153 — reimplemented rather than imported
    because that module pulls in matplotlib, h5py and tkinter at import time.
    Lexicographic order is chronological for YYYYMMDD_HHMMSS names, so the last
    match is the newest run.
    """
    try:
        matches = sorted(
            d for d in parent.iterdir()
            if d.is_dir() and not is_appledouble(d) and pattern.search(d.name)
        )
    except OSError:
        return None
    return matches[-1] if matches else None


def _sample_cache(superdir_name: str, sample_dir: Path, warn) -> CacheSource | None:
    """
    The newest crop cache under one sample dir, or None if it has none.

    Split out of discover_caches so compile_experiment.py can ask the same
    question about a single sample without rescanning a whole superdir, and so
    both agree on which run directory wins.
    """
    run_dir = _last_matching_dir(sample_dir, _RUN_DIR_PATTERN)
    if run_dir is None:
        return None
    # Current SMRFXMAnalysis writes the cache directly in run_dir. Runs from
    # before the stage1/stage2 split was dropped nest it under stage2_analysis/.
    stage2 = run_dir / _STAGE2_DIR
    if not stage2.is_dir():
        stage2 = run_dir

    caches = sorted(f for f in stage2.iterdir()
                    if f.is_file() and not is_appledouble(f)
                    and f.name.endswith(_CACHE_SUFFIX))
    if not caches:
        return None
    if len(caches) > 1:
        warn(f'{superdir_name}/{sample_dir.name}: {len(caches)} caches in '
             f'{run_dir.name}, using {caches[0].name}')

    pt_path = caches[0]
    parquet_path = sibling_path(pt_path)
    if parquet_path is None:
        warn(f'{superdir_name}/{sample_dir.name}: {pt_path.name} has no '
             f'sibling metadata parquet')
        return None

    m = _TS_PATTERN.match(run_dir.name)
    return CacheSource(
        pt_path=pt_path, parquet_path=parquet_path,
        experiment=superdir_name, sample_name=sample_dir.name,
        run_timestamp=m.group(1) if m else '')


def discover_caches(superdir: Path, warn) -> list[CacheSource]:
    """
    Find one cache pair per sample subdir of superdir.

    Depth is fixed at two levels (superdir -> sample -> run dir), matching
    calculate_baseline_density.py: anything deeper is another tool's output.
    Samples missing any part of the chain are skipped, not fatal.
    """
    found = []
    try:
        entries = sorted(superdir.iterdir())
    except OSError as exc:
        warn(f'cannot list {superdir}: {exc}')
        return found

    for sample_dir in entries:
        if not sample_dir.is_dir() or is_appledouble(sample_dir):
            continue
        src = _sample_cache(superdir.name, sample_dir, warn)
        if src is not None:
            found.append(src)
    return found


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _concatenable(info: _TensorInfo) -> str | None:
    """
    Why this tensor cannot be byte-concatenated, or None if it can be.

    Appending raw buffers only reproduces torch.cat along axis 0 when the buffer
    is the tensor's exact C-order contents. A permuted tensor has the wrong
    element order; a sliced one starts partway into a larger storage. Both would
    corrupt silently, so both are refused.
    """
    if not info.shape:
        return 'is a scalar (no axis 0 to concatenate)'
    if not info.contiguous:
        return f'is non-contiguous (stride {tuple(info.stride)})'
    if info.offset:
        return f'starts at storage offset {info.offset}, not 0'
    if info.storage is not None and info.storage.numel != info.numel:
        return (f'is a view of a larger storage '
                f'({info.numel:,} of {info.storage.numel:,} elements)')
    if info.storage is not None and info.storage.storage_class not in _STORAGE_ITEMSIZE:
        return f'has unknown storage class {info.storage.storage_class}'
    return None


def inspect_source(src: CacheSource) -> str | None:
    """
    Read both files' metadata into src. Returns a reason to skip, or None.

    Metadata only: no tensor data and no parquet rows are read, so validating
    every source up front costs almost nothing and lets a bad input abort before
    any output is written.
    """
    try:
        obj, _ = _read_pt(src.pt_path)
    except Exception as exc:
        return f'cannot read {src.pt_path.name}: {exc}'
    if not isinstance(obj, dict):
        return f'{src.pt_path.name} holds a {type(obj).__name__}, not a dict'

    tensors = {k: v for k, v in obj.items() if isinstance(v, _TensorInfo)}
    if not tensors:
        return f'{src.pt_path.name} holds no tensors'
    for key, info in sorted(tensors.items()):
        problem = _concatenable(info)
        if problem is not None:
            return f'{key} {problem}'
    src.tensors = tensors

    try:
        pf = pq.ParquetFile(str(src.parquet_path))
    except Exception as exc:
        return f'cannot read {src.parquet_path.name}: {exc}'
    try:
        src.nrows = pf.metadata.num_rows
        src.columns = [f.name for f in pf.schema_arrow]
    except Exception as exc:
        return f'cannot read {src.parquet_path.name} metadata: {exc}'
    finally:
        try:
            pf.close()
        except Exception:
            pass

    # The whole point of the pair is that row i describes tensor row i. If that
    # is already broken upstream, merging would spread mislabeled rows through
    # the output, so this source is unusable rather than merely suspect.
    lengths = {k: int(v.shape[0]) for k, v in tensors.items()}
    distinct = set(lengths.values())
    if len(distinct) > 1:
        detail = ', '.join(f'{k}={n:,}' for k, n in sorted(lengths.items()))
        return f'tensor lengths disagree ({detail})'
    n = distinct.pop()
    if src.nrows != n:
        return (f'parquet has {src.nrows:,} rows but tensors have N={n:,}')
    return None


def select_keys(sources: list[CacheSource], requested: str | None,
                require_fl: bool, note) -> list[str]:
    """
    Decide which tensor keys to merge.

    Defaults to the keys every source has. A key present in only some sources
    cannot be merged: its length would be shorter than the others', which is
    exactly the row misalignment this script exists to prevent.
    """
    common = set(sources[0].tensors)
    union = set()
    for src in sources:
        common &= set(src.tensors)
        union |= set(src.tensors)

    if requested:
        keys = [k.strip() for k in requested.split(',') if k.strip()]
        missing = [k for k in keys if k not in common]
        if missing:
            for key in missing:
                lacking = [s.label for s in sources if key not in s.tensors]
                raise ValueError(
                    f'--keys asked for {key!r} but {len(lacking)} source(s) lack '
                    f'it: {", ".join(lacking[:4])}'
                    f'{" ..." if len(lacking) > 4 else ""}')
        return keys

    for key in sorted(union - common):
        lacking = [s.label for s in sources if key not in s.tensors]
        if key == 'fl_u8' and require_fl:
            raise ValueError(
                f'--require-fl given but {len(lacking)} source(s) lack fl_u8: '
                f'{", ".join(lacking[:4])}{" ..." if len(lacking) > 4 else ""}')
        note(f'dropping {key}: absent from {len(lacking)} of {len(sources)} '
             f'source(s) ({", ".join(lacking[:3])}'
             f'{" ..." if len(lacking) > 3 else ""})')
    return sorted(common)


def check_shapes(sources: list[CacheSource], keys: list[str]) -> None:
    """
    Require identical trailing dimensions and dtype per key.

    Crops of different sizes cannot share a tensor, and unlike a missing key this
    is not something to work around silently — a 32x32 and a 64x64 cache are not
    the same dataset.
    """
    for key in keys:
        shapes = {}
        dtypes = {}
        for src in sources:
            info = src.tensors[key]
            shapes.setdefault(tuple(int(d) for d in info.shape[1:]), []).append(src.label)
            dtypes.setdefault(info.dtype, []).append(src.label)
        if len(shapes) > 1:
            detail = '; '.join(
                f'{shape} in {len(labels)} source(s) e.g. {labels[0]}'
                for shape, labels in sorted(shapes.items(), key=lambda kv: str(kv[0])))
            raise ValueError(f'{key} trailing dimensions differ: {detail}')
        if len(dtypes) > 1:
            detail = '; '.join(
                f'{dtype} in {len(labels)} source(s) e.g. {labels[0]}'
                for dtype, labels in sorted(dtypes.items()))
            raise ValueError(f'{key} dtypes differ: {detail}')


# ---------------------------------------------------------------------------
# Tensor writing
# ---------------------------------------------------------------------------

# A stub torch module, registered only so pickle can name the globals a real
# torch.save writes (torch._utils._rebuild_tensor_v2, torch.ByteStorage). pickle
# refuses to emit a global it cannot resolve by import, so these have to exist
# under those names. Nothing here is ever called, and no real torch code runs.
_torch_stub = types.ModuleType('torch')
_torch_stub._utils = types.ModuleType('torch._utils')


def _rebuild_tensor_v2(*args, **kwargs):
    """Never called: it exists so pickle can reference it by name."""
    raise NotImplementedError('placeholder for torch._utils._rebuild_tensor_v2')


_rebuild_tensor_v2.__module__ = 'torch._utils'
_rebuild_tensor_v2.__qualname__ = _rebuild_tensor_v2.__name__ = '_rebuild_tensor_v2'
_torch_stub._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

for _name in _STORAGE_ITEMSIZE:
    _cls = type(_name, (object,), {})
    _cls.__module__ = 'torch'
    _cls.__qualname__ = _name
    setattr(_torch_stub, _name, _cls)

sys.modules.setdefault('torch', _torch_stub)
sys.modules.setdefault('torch._utils', _torch_stub._utils)


class _StorageRef:
    """Marker the pickler turns into torch's ('storage', ...) persistent id."""

    def __init__(self, key: str, storage_class: str, numel: int):
        self.key           = key
        self.storage_class = storage_class
        self.numel         = numel


class _TensorPlaceholder:
    """
    Pickles exactly as a real torch tensor of the given shape.

    The mirror image of browse_pt's stubbed unpickler: it emits the same
    _rebuild_tensor_v2 call with a C-contiguous stride, so torch.load
    reconstructs a tensor pointing at the storage blob written alongside.
    """

    def __init__(self, storage_key: str, storage_class: str, shape: tuple):
        self.storage_key   = storage_key
        self.storage_class = storage_class
        self.shape         = tuple(int(d) for d in shape)

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    def __reduce_ex__(self, protocol):
        stride, acc = [], 1
        for d in reversed(self.shape):
            stride.append(acc)
            acc *= d
        return (_rebuild_tensor_v2,
                (_StorageRef(self.storage_key, self.storage_class, self.numel),
                 0, self.shape, tuple(reversed(stride)), False, None))


class _Pickler(pickle.Pickler):
    def persistent_id(self, obj):
        if isinstance(obj, _StorageRef):
            return ('storage', getattr(_torch_stub, obj.storage_class),
                    obj.key, 'cpu', obj.numel)
        return None


def _open_aligned(zf: zipfile.ZipFile, name: str, size: int):
    """
    Open a STORED zip entry whose payload starts on a _ZIP_ALIGN boundary.

    torch pads each local file header's extra field so storage blobs land on
    64-byte boundaries, and its mmap-based loading path relies on that. zipfile
    exposes no alignment control, so the padding is computed from the current
    file position and placed in ZipInfo.extra by hand.
    """
    info = zipfile.ZipInfo(name)
    info.compress_type = zipfile.ZIP_STORED
    info.file_size = size
    # Local header is 30 bytes + the name, then the extra field, then the data.
    overhead = zf.fp.tell() + 30 + len(name.encode('utf-8'))
    pad = (-overhead) % _ZIP_ALIGN
    if pad:
        info.extra = b'\x00' * pad
    return zf.open(info, mode='w')


def _blob_name(zf: zipfile.ZipFile, storage_key: str) -> str:
    """The archive entry holding one storage blob."""
    suffix = f'data/{storage_key}'
    for name in zf.namelist():
        if name == suffix or name.endswith(f'/{suffix}'):
            return name
    raise KeyError(f'no storage blob {storage_key!r} in {zf.filename}')


def _copy_blob(src_path: Path, storage_key: str, dst, expect: int) -> int:
    """
    Stream one storage blob from a source archive into dst.

    Chunked so peak memory is a chunk, not a tensor: the whole reason this
    script can merge multi-gigabyte caches.
    """
    written = 0
    with zipfile.ZipFile(str(src_path)) as zf:
        name = _blob_name(zf, storage_key)
        with zf.open(name) as fh:
            while True:
                chunk = fh.read(_CHUNK_BYTES)
                if not chunk:
                    break
                dst.write(chunk)
                written += len(chunk)
    if written != expect:
        raise ValueError(
            f'{src_path.name}: storage {storage_key} held {written:,} bytes, '
            f'expected {expect:,}')
    return written


def write_concat_pt(sources: list[CacheSource], keys: list[str],
                    out_path: Path, prefix: str = 'archive') -> dict:
    """
    Write the concatenated .pt, streaming every source blob through.

    Returns {key: shape}. Sources are consumed in list order, which is the same
    order the parquet pass uses, so tensor rows and parquet rows stay aligned.
    """
    shapes = {}
    plan = []           # (storage_key, tensor key, total bytes)
    payload = {}
    for i, key in enumerate(keys):
        first = sources[0].tensors[key]
        tail = tuple(int(d) for d in first.shape[1:])
        total_n = sum(int(s.tensors[key].shape[0]) for s in sources)
        storage_class = first.storage.storage_class
        itemsize = _STORAGE_ITEMSIZE[storage_class]

        shape = (total_n,) + tail
        placeholder = _TensorPlaceholder(str(i), storage_class, shape)
        payload[key] = placeholder
        shapes[key] = shape
        plan.append((str(i), key, placeholder.numel * itemsize))

    buf = io.BytesIO()
    _Pickler(buf, protocol=2).dump(payload)
    data_pkl = buf.getvalue()

    with zipfile.ZipFile(str(out_path), 'w', zipfile.ZIP_STORED) as zf:
        with _open_aligned(zf, f'{prefix}/data.pkl', len(data_pkl)) as dst:
            dst.write(data_pkl)
        # Written by torch.save and read back by torch.load; keep both so the
        # output is indistinguishable from a real save.
        with _open_aligned(zf, f'{prefix}/byteorder', len(sys.byteorder)) as dst:
            dst.write(sys.byteorder.encode('ascii'))
        with _open_aligned(zf, f'{prefix}/version', 2) as dst:
            dst.write(b'3\n')

        for storage_key, key, total_bytes in plan:
            with _open_aligned(zf, f'{prefix}/data/{storage_key}',
                               total_bytes) as dst:
                for src in sources:
                    info = src.tensors[key]
                    itemsize = _STORAGE_ITEMSIZE[info.storage.storage_class]
                    _copy_blob(src.pt_path, info.storage.key, dst,
                               info.numel * itemsize)
    return shapes


# ---------------------------------------------------------------------------
# Parquet writing
# ---------------------------------------------------------------------------

def _union_schema(sources: list[CacheSource]) -> tuple[pa.Schema, dict]:
    """
    A schema covering every source's columns, plus the added label columns.

    Column order follows first appearance so the common case (identical schemas)
    comes out in the original order. Returns (schema, {column: [labels lacking
    it]}) so the caller can report what was null-filled.
    """
    fields: list[pa.Field] = []
    seen: dict[str, pa.DataType] = {}
    for src in sources:
        try:
            schema = pq.ParquetFile(str(src.parquet_path)).schema_arrow
        except Exception:
            continue
        for field in schema:
            name = _renamed(field.name)
            if name in seen:
                continue
            seen[name] = field.type
            fields.append(pa.field(name, field.type))

    missing = {}
    for name in seen:
        lacking = [s.label for s in sources
                   if name not in [_renamed(c) for c in s.columns]]
        if lacking:
            missing[name] = lacking

    fields.append(pa.field(_LABEL_EXPERIMENT, pa.string()))
    fields.append(pa.field(_LABEL_SAMPLE, pa.string()))
    fields.append(pa.field(_LABEL_ROW, pa.int64()))
    return pa.schema(fields), missing


def _renamed(column: str) -> str:
    """
    The output name for a source column.

    A source that already has an `experiment` column would otherwise be
    overwritten by the label; suffix it instead so upstream data survives.
    """
    return f'{column}_source' if column in _ADDED_COLUMNS else column


def write_concat_parquet(sources: list[CacheSource], out_path: Path,
                         schema: pa.Schema) -> int:
    """
    Write the concatenated parquet, streaming batches.

    cache_row counts rows across the whole output, so it indexes axis 0 of the
    concatenated tensor. Sources are consumed in the same order as the tensor
    pass, which is what keeps the two in step.
    """
    written = 0
    writer = pq.ParquetWriter(str(out_path), schema)
    try:
        for src in sources:
            pf = pq.ParquetFile(str(src.parquet_path))
            try:
                for batch in pf.iter_batches(batch_size=_BATCH_ROWS):
                    n = batch.num_rows
                    columns = []
                    present = {_renamed(name): batch.column(i)
                               for i, name in enumerate(batch.schema.names)}
                    for field in schema:
                        if field.name == _LABEL_EXPERIMENT:
                            columns.append(pa.array([src.experiment] * n,
                                                    type=pa.string()))
                        elif field.name == _LABEL_SAMPLE:
                            columns.append(pa.array([src.sample_name] * n,
                                                    type=pa.string()))
                        elif field.name == _LABEL_ROW:
                            columns.append(pa.array(
                                range(written, written + n), type=pa.int64()))
                        elif field.name in present:
                            col = present[field.name]
                            columns.append(col if col.type == field.type
                                           else col.cast(field.type))
                        else:
                            # Absent from this source: null-filled so the row
                            # still lines up with its tensor row.
                            columns.append(pa.nulls(n, type=field.type))
                    writer.write_table(
                        pa.Table.from_arrays(columns, schema=schema))
                    written += n
            finally:
                try:
                    pf.close()
                except Exception:
                    pass
    finally:
        writer.close()
    return written


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

def self_check(pt_path: Path, parquet_path: Path,
               keys: list[str]) -> list[str]:
    """
    Re-read both outputs and confirm the invariant this script exists to keep.

    Metadata only, so it is nearly free. Turning a silent misalignment into a
    loud failure is worth the second open.
    """
    problems = []
    try:
        obj, _ = _read_pt(pt_path)
    except Exception as exc:
        return [f'cannot re-read {pt_path.name}: {exc}']

    lengths = {}
    for key in keys:
        info = obj.get(key)
        if not isinstance(info, _TensorInfo):
            problems.append(f'{key} missing from the written .pt')
            continue
        lengths[key] = int(info.shape[0])
    if len(set(lengths.values())) > 1:
        problems.append(f'tensor lengths disagree in output: {lengths}')

    try:
        pf = pq.ParquetFile(str(parquet_path))
        nrows = pf.metadata.num_rows
        last = None
        if nrows:
            # cache_row is monotonic, so the final row group holds its maximum.
            tail = pf.read_row_groups([pf.metadata.num_row_groups - 1],
                                      columns=[_LABEL_ROW])
            values = tail.column(_LABEL_ROW).to_pylist()
            last = values[-1] if values else None
        pf.close()
    except Exception as exc:
        return problems + [f'cannot re-read {parquet_path.name}: {exc}']

    for key, n in lengths.items():
        if n != nrows:
            problems.append(f'{key} has N={n:,} but parquet has {nrows:,} rows')
    if nrows and last != nrows - 1:
        problems.append(f'{_LABEL_ROW} ends at {last}, expected {nrows - 1}')
    return problems


# ---------------------------------------------------------------------------
# Plan / run
# ---------------------------------------------------------------------------

class ConcatPlan:
    """
    A validated merge, ready to write: which sources are usable, which tensor
    keys they share, and the parquet schema their union needs.

    Separated from the writing step so a caller can inspect or print the plan
    (and, for --dry-run, stop there). compile_experiment.py builds a plan from
    the samples it already discovered rather than rescanning the superdir.
    """

    def __init__(self, usable: list[CacheSource], skipped: list[tuple[str, str]],
                 keys: list[str], schema: pa.Schema):
        self.usable  = usable
        self.skipped = skipped
        self.keys    = keys
        self.schema  = schema

    @property
    def total_rows(self) -> int:
        return sum(s.nrows for s in self.usable)


class ConcatResult:
    """What run_concat wrote: output paths, per-key shapes, and any problems."""

    def __init__(self, pt_path: Path, parquet_path: Path, shapes: dict,
                 rows: int, problems: list[str]):
        self.pt_path      = pt_path
        self.parquet_path = parquet_path
        self.shapes       = shapes
        self.rows         = rows
        self.problems     = problems


def plan_concat(sources: list[CacheSource], requested_keys: str | None = None,
                require_fl: bool = False, note=print, log=print) -> ConcatPlan:
    """
    Validate every source and decide what the merge will contain.

    Metadata only — no tensor data and no parquet rows are read, so a bad input
    aborts before anything is written. Raises ValueError when no usable source
    survives, when the sources share no tensor key, or when their shapes/dtypes
    are incompatible.
    """
    usable, skipped = [], []
    for src in sources:
        reason = inspect_source(src)
        if reason is None:
            usable.append(src)
        else:
            skipped.append((src.label, reason))
            log(f'  [skip] {src.label}: {reason}')
    if not usable:
        raise ValueError('no usable sources after validation')

    keys = select_keys(usable, requested_keys, require_fl, note)
    if not keys:
        raise ValueError('no tensor key is present in every source')
    check_shapes(usable, keys)

    schema, missing_cols = _union_schema(usable)
    for column, lacking in sorted(missing_cols.items()):
        note(f'column {column!r} absent from {len(lacking)} source(s), '
             f'null-filled ({", ".join(lacking[:3])}'
             f'{" ..." if len(lacking) > 3 else ""})')
    for column in sorted({c for s in usable for c in s.columns
                          if c in _ADDED_COLUMNS}):
        note(f'source column {column!r} kept as {column}_source '
             f'(the label column takes that name)')

    return ConcatPlan(usable, skipped, keys, schema)


def run_concat(plan: ConcatPlan, out_dir: Path, log=print) -> ConcatResult:
    """
    Write the plan's .pt and .parquet into out_dir, then self-check them.

    The tensor pass and the parquet pass walk plan.usable in the same order,
    which is what keeps tensor row i and parquet row i describing the same crop.
    """
    pt_out      = out_dir / OUT_PT_NAME
    parquet_out = out_dir / OUT_PARQUET_NAME

    log(f'Writing {pt_out.name}...')
    shapes = write_concat_pt(plan.usable, plan.keys, pt_out)
    log(f'Writing {parquet_out.name}...')
    rows = write_concat_parquet(plan.usable, parquet_out, plan.schema)

    problems = self_check(pt_out, parquet_out, plan.keys)
    return ConcatResult(pt_out, parquet_out, shapes, rows, problems)


def summarize(plan: ConcatPlan, result: ConcatResult, n_sources: int,
              log=print) -> None:
    """Print the per-key shapes, row/column counts and any self-check failure."""
    log(f'  {len(plan.usable)}/{n_sources} sources merged, '
        f'{len(plan.skipped)} skipped\n')
    for key in plan.keys:
        info = plan.usable[0].tensors[key]
        nbytes = _STORAGE_ITEMSIZE[info.storage.storage_class]
        for d in result.shapes[key]:
            nbytes *= d
        log(f'  {key:<8}{str(result.shapes[key]):<24}{info.dtype:<8}'
            f'{nbytes / (1024 * 1024):>9,.1f} MB')
    log(f'  {"rows":<8}{result.rows:,}')
    log(f'  {"columns":<8}{len(plan.schema.names)} '
        f'({len(plan.schema.names) - len(_ADDED_COLUMNS)} source + '
        f'{len(_ADDED_COLUMNS)} added)')

    if result.problems:
        log('\n  [FAIL] output failed its own consistency check:')
        for problem in result.problems:
            log(f'         {problem}')
    if plan.skipped:
        log('')
        for label, reason in plan.skipped:
            log(f'  [skipped] {label}: {reason}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()

    try:
        superdirs = _resolve_dirs(args)
    except Exception as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)

    warnings: list[str] = []

    def warn(message: str):
        warnings.append(message)
        print(f'  [warn] {message}')

    def note(message: str):
        print(f'  [note] {message}')

    # ---- Discover ----
    sources: list[CacheSource] = []
    for superdir in superdirs:
        print(f'\nScanning: {superdir.name}')
        found = discover_caches(superdir, warn)
        if not found:
            print('  No crop caches found.')
        for src in found:
            print(f'  [found] {src.label}')
        sources += found

    if not sources:
        print('\nNo crop caches found under any superdir.', file=sys.stderr)
        sys.exit(1)

    # ---- Validate ----
    print(f'\nValidating {len(sources)} source(s)...')
    try:
        plan = plan_concat(sources, args.keys, args.require_fl, note)
    except ValueError as exc:
        print(f'\nError: {exc}', file=sys.stderr)
        sys.exit(1)

    print(f'\nPlan: {len(plan.usable)} source(s), {plan.total_rows:,} rows, '
          f'keys {", ".join(plan.keys)}')
    for src in plan.usable:
        shape_bits = ', '.join(
            f'{k}{tuple(int(d) for d in src.tensors[k].shape)}' for k in plan.keys)
        print(f'  {src.label:<44}{src.nrows:>10,} rows   {shape_bits}')

    if args.dry_run:
        print('\n[dry-run] Nothing written.')
        return

    # ---- Write ----
    out_parent = Path(args.output) if args.output else superdirs[0]
    if not out_parent.is_dir():
        print(f'Error: Directory not found: {out_parent}', file=sys.stderr)
        sys.exit(1)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = out_parent / f'{timestamp}_vqvae_concat'
    # No exist_ok: the timestamp makes collisions impossible, so one would mean
    # something is wrong and should be loud (as compile_experiment.py does).
    out_dir.mkdir()

    print()
    result = run_concat(plan, out_dir)

    # ---- Summary ----
    print(f'\n{"=" * 60}')
    print('VQVAE CONCAT SUMMARY')
    print(f'{"=" * 60}')
    summarize(plan, result, len(sources))

    print(f'\nDone. Output written to: {out_dir}')
    if result.problems:
        sys.exit(1)


if __name__ == '__main__':
    main()
