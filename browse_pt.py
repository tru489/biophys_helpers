"""
browse_pt.py

Structure browser for PyTorch (.pt / .pth) files. Reports the dimensions, dtype
and device of every tensor inside, interprets 3-D and 4-D tensors as image stacks,
and surfaces the sibling metadata parquet that accompanies a VQ-VAE crop cache.

Written for the crop caches emitted by stage 2 of the ImageFXMAnalysis pipeline.
Those files hold only image tensors — `bf_u8`, always, and `fl_u8` when
fluorescence export is enabled, both uint8 of shape (N, 1, S, S). All per-crop
metadata deliberately lives in a sibling `<stem>_metadata.parquet` with one row
per crop, row-aligned with axis 0 of the tensors. This tool therefore looks for
that sibling and reports its row count, its constant-down-file columns, and
whether its row count still matches N — a mismatch means a broken export.

The file is read WITHOUT importing torch. A .pt is a zip archive holding one
pickle plus raw storage blobs; every shape and dtype lives in the pickle, so
unpickling with stubbed classes recovers the full structure without loading a
single byte of tensor data and without a ~2 GB torch install. No class named in
the file is ever resolved to real code, so nothing in the file can execute.

Usage:
    python browse_pt.py [<pt_path>] [--no-sibling] [--dump]

    <pt_path>     Path to a .pt / .pth file. If omitted, a file picker opens.
    --no-sibling  Skip the sibling metadata-parquet lookup.
    --dump        Print the structure to stdout and exit; no window is opened.
                  Requires <pt_path>.
"""
import argparse
import io
import pickle
import subprocess
import sys
import zipfile
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from fsutil import is_appledouble

# pyarrow is only needed for the sibling metadata parquet, which is optional —
# a .pt with no sibling must still open.
try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None


_MAX_KIDS   = 200      # children shown per container
_MAX_DEPTH  = 40       # recursion cap on the object graph
_VAL_MAXLEN = 200      # truncate scalar/opaque value reprs

# Bytes per element for each torch storage class. Used to report a tensor's
# nominal size without reading its storage. Anything absent reports '?' rather
# than guessing.
_STORAGE_ITEMSIZE = {
    'ByteStorage': 1, 'CharStorage': 1, 'BoolStorage': 1,
    'ShortStorage': 2, 'HalfStorage': 2, 'BFloat16Storage': 2,
    'IntStorage': 4, 'FloatStorage': 4,
    'LongStorage': 8, 'DoubleStorage': 8, 'ComplexFloatStorage': 8,
    'ComplexDoubleStorage': 16,
}

# Storage class name -> the torch dtype name a user would recognise. torch names
# its storages after C types but its dtypes after widths, so the two disagree
# (ByteStorage backs uint8, CharStorage backs int8).
_STORAGE_DTYPE = {
    'ByteStorage': 'uint8', 'CharStorage': 'int8', 'BoolStorage': 'bool',
    'ShortStorage': 'int16', 'HalfStorage': 'float16',
    'BFloat16Storage': 'bfloat16', 'IntStorage': 'int32',
    'FloatStorage': 'float32', 'LongStorage': 'int64',
    'DoubleStorage': 'float64', 'ComplexFloatStorage': 'complex64',
    'ComplexDoubleStorage': 'complex128',
}

# Columns the ImageFXMAnalysis exporter writes with the same value on every row —
# effectively the file header, and the reason a viewer bothers with the sibling.
_CONSTANT_COLUMNS = ('crop_size', 'center_pixel', 'source_file',
                     'cellgrouped_file', 'frame_selection')

# The tensor whose length defines N. Checked against the sibling's row count.
_PRIMARY_KEYS = ('bf_u8', 'fl_u8')

_SCALARS = (int, float, bool, str, bytes, complex, type(None))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Browse the tensors and metadata inside a PyTorch .pt file.')
    parser.add_argument('pt_path', type=str, nargs='?',
                        help='Path to a .pt / .pth file (a picker opens if omitted)')
    parser.add_argument('--no-sibling', action='store_true',
                        help='Skip the sibling metadata-parquet lookup')
    parser.add_argument('--dump', action='store_true',
                        help='Print the structure to stdout and exit')
    return parser.parse_args()


def _resolve_path(arg: str | None, allow_picker: bool) -> Path:
    """
    Turn the CLI argument into a validated Path, prompting with a file dialog
    when no argument was given.
    """
    if arg is None:
        if not allow_picker:
            raise ValueError('No file given (a path is required with --dump)')
        # A bare Tk root is needed for the dialog; discard it afterwards so the
        # real application window owns the mainloop.
        tmp = tk.Tk()
        tmp.withdraw()
        chosen = filedialog.askopenfilename(
            title='Select a PyTorch file',
            filetypes=[('PyTorch files', '*.pt *.pth'), ('All files', '*.*')])
        tmp.destroy()
        if not chosen:
            raise ValueError('No file selected')
        arg = chosen

    p = Path(arg)
    if not p.exists():
        raise FileNotFoundError(f'File not found: {p}')
    if not p.is_file():
        raise ValueError(f'Not a file: {p}')
    if is_appledouble(p):
        raise ValueError(
            f'{p.name} is a macOS AppleDouble sidecar, not a PyTorch file. '
            f'Try {p.name[2:]} instead.')
    return p


# ---------------------------------------------------------------------------
# Torch-free .pt reading
# ---------------------------------------------------------------------------

class _Opaque:
    """
    Stand-in for any class or function the pickle references.

    Instantiated in place of real classes (nn.Module subclasses, optimizers,
    argparse Namespaces) so the graph can be walked without importing torch and
    without executing anything the file asks for. Needs to tolerate every way
    pickle might construct it: as a constructor, as a factory called on
    arguments, and via __setstate__.

    Constructor and call arguments are kept, because an unrecognised
    torch._utils._rebuild_* function would otherwise swallow the tensors passed
    to it and they would vanish from the tree.
    """

    def __init__(self, *args, **kwargs):
        self._args = args

    def __call__(self, *args, **kwargs):
        out = _Opaque()
        out._args = args
        return out

    def __setstate__(self, state):
        self._state = state

    def __repr__(self):
        return f'<{type(self).__name__}>'


_opaque_cache: dict[tuple[str, str], type] = {}


def _opaque_class(module: str, name: str) -> type:
    """
    A distinctly-named _Opaque subclass per (module, name).

    Must be a class, not an instance: pickle's NEWOBJ opcode requires a type and
    raises 'class argument must be a type' otherwise. Cached so repeated
    references to the same class compare equal.
    """
    key = (module, name)
    if key not in _opaque_cache:
        _opaque_cache[key] = type(name, (_Opaque,),
                                  {'_module': module, '_qualname': f'{module}.{name}'})
    return _opaque_cache[key]


class _StorageInfo:
    """A tensor's backing storage, described but never read."""

    def __init__(self, key: str, storage_class: str, device: str, numel: int):
        self.key           = key
        self.storage_class = storage_class
        self.device        = device
        self.numel         = numel

    @property
    def dtype(self) -> str:
        return _STORAGE_DTYPE.get(self.storage_class, self.storage_class)

    @property
    def itemsize(self) -> int | None:
        return _STORAGE_ITEMSIZE.get(self.storage_class)


class _TensorInfo:
    """A tensor's shape/dtype/device, recovered from the pickle alone."""

    def __init__(self, storage: _StorageInfo, shape: tuple,
                 stride: tuple, offset: int):
        self.storage = storage
        self.shape   = shape
        self.stride  = stride
        self.offset  = offset

    @property
    def dtype(self) -> str:
        return self.storage.dtype if self.storage is not None else '?'

    @property
    def device(self) -> str:
        return self.storage.device if self.storage is not None else '?'

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= int(d)
        return n

    @property
    def nbytes(self) -> int | None:
        item = self.storage.itemsize if self.storage is not None else None
        return None if item is None else self.numel * item

    @property
    def contiguous(self) -> bool:
        """True when the strides are the C-contiguous ones for this shape."""
        expected = []
        acc = 1
        for d in reversed(self.shape):
            expected.append(acc)
            acc *= int(d)
        return tuple(reversed(expected)) == tuple(self.stride)

    def __repr__(self):
        return f'Tensor({tuple(self.shape)}, {self.dtype})'


def _rebuild_tensor(storage, storage_offset, size, stride, *rest):
    """
    Stand-in for torch._utils._rebuild_tensor_v2 (and _rebuild_tensor).

    Records the shape the pickle asks for and returns immediately; the storage
    blob it refers to is never opened.
    """
    return _TensorInfo(storage if isinstance(storage, _StorageInfo) else None,
                       tuple(size), tuple(stride), storage_offset)


def _rebuild_parameter(data, requires_grad=True, backward_hooks=None, *rest):
    """
    Stand-in for torch._utils._rebuild_parameter.

    Every weight in a saved nn.Module goes through this rather than through
    _rebuild_tensor_v2 directly, so without it a model checkpoint's tensors all
    collapse into opaque objects. The tensor it wraps is already a _TensorInfo.
    """
    return data


class _StubUnpickler(pickle.Unpickler):
    """Unpickler that resolves nothing to real code."""

    def find_class(self, module, name):
        if module == 'torch._utils':
            if name.startswith('_rebuild_tensor'):
                return _rebuild_tensor
            if name.startswith('_rebuild_parameter'):
                return _rebuild_parameter
        if module == 'collections' and name == 'OrderedDict':
            # Ordering is preserved by dict, and this keeps the tree simple.
            return dict
        return _opaque_class(module, name)

    def persistent_load(self, pid):
        # torch writes ('storage', storage_type, key, location, numel). The
        # storage type has already been through find_class, so it is one of our
        # opaque classes and carries the original name; location is the device
        # string, which is how the device is reported without torch.
        try:
            storage_class = getattr(pid[1], '__name__', str(pid[1]))
            return _StorageInfo(str(pid[2]), storage_class, str(pid[3]),
                                int(pid[4]))
        except Exception:
            return _StorageInfo('?', '?', '?', 0)


def _read_pt(path: Path) -> tuple[object, dict]:
    """
    Read a .pt and return (object graph, file info).

    Only the pickle entry is decompressed; the storage blobs are left alone,
    which is why this is instant even on multi-gigabyte files.
    """
    if not zipfile.is_zipfile(str(path)):
        # Distinguish a genuine pre-1.6 torch file from a file that simply isn't
        # torch at all: the legacy format is a bare pickle, so it opens with a
        # protocol-2 PROTO opcode. Reading those needs torch's legacy loader,
        # which this tool avoids on purpose.
        try:
            head = path.read_bytes()[:2]
        except OSError:
            head = b''
        if head[:1] == b'\x80':
            raise ValueError(
                f'{path.name} looks like a pre-1.6 PyTorch file (a bare pickle, '
                f'saved with _use_new_zipfile_serialization=False). Only the '
                f'zip format is supported; re-save it with a current torch.')
        raise ValueError(
            f'{path.name} is not a PyTorch file: it is neither a zip archive '
            f'nor a pickle.')

    with zipfile.ZipFile(str(path)) as zf:
        names = zf.namelist()
        pkl_names = [n for n in names if n.endswith('data.pkl')]
        if not pkl_names:
            raise ValueError(
                f'{path.name} is a zip archive but holds no data.pkl, so it is '
                f'not a torch.save file.')
        # Shortest path wins: a nested archive would sort later but the top-level
        # pickle is the one torch.load reads.
        pkl_name = min(pkl_names, key=len)
        prefix = pkl_name[:-len('data.pkl')].rstrip('/')

        info = {
            'entries':    len(names),
            'prefix':     prefix or '(none)',
            'pickle':     pkl_name,
            'pickle_size': zf.getinfo(pkl_name).file_size,
            'blobs':      sum(1 for n in names if f'{prefix}/data/' in n
                              or n.startswith('data/')),
            'size':       path.stat().st_size,
        }
        for extra in ('byteorder', 'version'):
            candidate = f'{prefix}/{extra}' if prefix else extra
            if candidate in names:
                try:
                    info[extra] = zf.read(candidate).decode('ascii', 'replace').strip()
                except Exception:
                    pass

        raw = zf.read(pkl_name)

    obj = _StubUnpickler(io.BytesIO(raw)).load()
    return obj, info


# ---------------------------------------------------------------------------
# Node classification
# ---------------------------------------------------------------------------

def classify(node) -> dict:
    """
    Describe one node of the object graph.

    Returns a dict with:
        kind       short type name for the tree's Kind column
        shape      shape / length as display text
        dtype      dtype as display text
        expandable whether the node has children worth showing
    """
    info = {'kind': type(node).__name__, 'shape': '', 'dtype': '',
            'expandable': False}

    if isinstance(node, _TensorInfo):
        info['kind'] = 'Tensor'
        info['shape'] = 'scalar' if not node.shape else str(tuple(node.shape))
        info['dtype'] = node.dtype
        return info

    if isinstance(node, dict):
        info['kind'] = 'dict'
        info['shape'] = f'{len(node)} item(s)'
        info['expandable'] = len(node) > 0
        return info

    if isinstance(node, (list, tuple, set)):
        info['kind'] = type(node).__name__
        info['shape'] = f'{len(node)} item(s)'
        info['expandable'] = len(node) > 0
        return info

    if isinstance(node, _Opaque):
        # An opaque object's contents are whatever __setstate__ received, plus
        # any constructor arguments — an unrecognised _rebuild_* function shows
        # up here, and its tensors must not disappear.
        info['kind'] = getattr(node, '_qualname', type(node).__name__)
        n = len(_opaque_children(node))
        if n:
            state = getattr(node, '_state', None)
            info['shape'] = (f'{n} attr(s)' if isinstance(state, dict)
                             else f'{n} item(s)')
            info['expandable'] = True
        return info

    if isinstance(node, _SCALARS):
        info['kind'] = type(node).__name__
        info['shape'] = _fmt_value(node)
        return info

    return info


def _opaque_children(node) -> list[tuple[str, object]]:
    """
    (name, child) pairs for an opaque object: its __setstate__ payload, then any
    constructor arguments.

    The arguments matter because a torch._utils._rebuild_* variant this tool
    doesn't recognise lands here holding real tensors; showing them keeps the
    tree honest instead of silently dropping them.
    """
    items: list[tuple[str, object]] = []
    state = getattr(node, '_state', None)
    if isinstance(state, dict):
        items += [(str(k), v) for k, v in state.items()]
    elif isinstance(state, (list, tuple)):
        items += [(f'state[{i}]', v) for i, v in enumerate(state)]
    elif state is not None:
        items.append(('state', state))
    for i, arg in enumerate(getattr(node, '_args', ()) or ()):
        # Bare scalars in a constructor call are rarely informative; tensors and
        # containers are.
        if not isinstance(arg, _SCALARS):
            items.append((f'arg[{i}]', arg))
    return items


def children(node) -> list[tuple[str, object]]:
    """(name, child) pairs for a container node, capped at _MAX_KIDS."""
    items: list[tuple[str, object]] = []
    if isinstance(node, dict):
        items = [(str(k), v) for k, v in node.items()]
    elif isinstance(node, (list, tuple, set)):
        items = [(f'[{i}]', v) for i, v in enumerate(node)]
    elif isinstance(node, _Opaque):
        items = _opaque_children(node)
    return items[:_MAX_KIDS]


def _fmt_value(value) -> str:
    """Render a scalar as bounded, single-line text."""
    if isinstance(value, bytes):
        return f'<{len(value)} bytes>'
    text = ' '.join(str(value).split())
    if len(text) > _VAL_MAXLEN:
        text = text[:_VAL_MAXLEN] + f'... ({len(text)} chars)'
    return text


# ---------------------------------------------------------------------------
# Image-stack interpretation
# ---------------------------------------------------------------------------

def _stack_lines(node: _TensorInfo) -> list[str]:
    """
    Read a tensor's shape as an image stack.

    The crop caches this tool targets are (N, C, H, W); 3-D tensors written by
    other tools are usually (N, H, W). Anything else is reported as-is rather
    than forced into a layout it may not have.
    """
    lines = ['', '  --- interpreted as image stack ---']
    shape = tuple(int(d) for d in node.shape)

    if len(shape) == 4:
        n, c, h, w = shape
        lines.append(f'  {"frames (N)":<16}{n:,}')
        lines.append(f'  {"channels (C)":<16}{c}')
        lines.append(f'  {"height (H)":<16}{h}')
        lines.append(f'  {"width (W)":<16}{w}')
    elif len(shape) == 3:
        n, h, w = shape
        lines.append(f'  {"frames (N)":<16}{n:,}')
        lines.append(f'  {"height (H)":<16}{h}')
        lines.append(f'  {"width (W)":<16}{w}')
        lines.append('  (no channel axis; read as (N, H, W))')
    elif len(shape) == 2:
        h, w = shape
        lines.append(f'  {"height (H)":<16}{h}')
        lines.append(f'  {"width (W)":<16}{w}')
        lines.append('  (single image, read as (H, W))')
    else:
        lines.append(f'  {"shape":<16}{shape}')
        lines.append(f'  (rank {len(shape)} is not a recognised image-stack '
                     f'layout)')

    nbytes = node.nbytes
    if nbytes is not None:
        lines.append(f'  {"bytes":<16}{nbytes / (1024 * 1024):,.1f} MB '
                     f'({node.dtype})')
    return lines


def _primary_length(obj) -> tuple[str, int] | None:
    """
    (key, N) for the tensor whose axis 0 the sibling parquet should match.

    Prefers bf_u8 (always written by the exporter), then fl_u8, then any single
    top-level tensor.
    """
    if not isinstance(obj, dict):
        return None
    for key in _PRIMARY_KEYS:
        node = obj.get(key)
        if isinstance(node, _TensorInfo) and node.shape:
            return key, int(node.shape[0])
    tensors = [(k, v) for k, v in obj.items()
               if isinstance(v, _TensorInfo) and v.shape]
    if len(tensors) == 1:
        return tensors[0][0], int(tensors[0][1].shape[0])
    return None


# ---------------------------------------------------------------------------
# Sibling metadata parquet
# ---------------------------------------------------------------------------

def sibling_path(path: Path) -> Path | None:
    """
    The `<stem>_metadata.parquet` the exporter writes next to a crop cache.

    See ImageFXMAnalysis/pipeline/stage2/export.py, which derives it as
    os.path.splitext(output_path)[0] + '_metadata.parquet'.
    """
    candidate = path.with_name(f'{path.stem}_metadata.parquet')
    if candidate.is_file() and not is_appledouble(candidate):
        return candidate
    return None


def _sibling_lines(path: Path, expected: tuple[str, int] | None) -> list[str]:
    """
    Describe the sibling metadata parquet, if there is one.

    The row-count check against the tensor's N is the point of this block: the
    two files are joined by position alone, so a disagreement means the export
    is broken and nothing downstream can be trusted.
    """
    lines = ['', '  --- sibling metadata ---']
    sib = sibling_path(path)
    if sib is None:
        lines.append(f'  (no {path.stem}_metadata.parquet beside this file)')
        return lines
    if pq is None:
        lines.append(f'  {"file":<18}{sib.name}')
        lines.append('  (pyarrow not installed; cannot read it)')
        return lines

    lines.append(f'  {"file":<18}{sib.name}')
    try:
        pf = pq.ParquetFile(str(sib))
    except Exception as exc:
        lines.append(f'  (unreadable: {exc.__class__.__name__}: {exc})')
        return lines

    try:
        nrows = pf.metadata.num_rows
        names = [f.name for f in pf.schema_arrow]
        note = ''
        if expected is not None:
            key, n = expected
            note = (f'   (matches {key} N)' if nrows == n
                    else f'   (MISMATCH: {key} has N={n:,})')
        lines.append(f'  {"rows":<18}{nrows:,}{note}')
        lines.append(f'  {"columns":<18}{len(names)}')

        # The constant-down-file columns are the exporter's header. Read row 0
        # only — one row group, whatever the file's size.
        present = [c for c in _CONSTANT_COLUMNS if c in names]
        if present and nrows:
            head = pf.read_row_groups([0], columns=present).slice(0, 1)
            for col in present:
                value = head.column(col).to_pylist()[0]
                lines.append(f'  {col:<18}{_fmt_value(value)}')
        elif not present:
            # SMRFXMAnalysis writes a wider, differently-named schema, so a
            # missing header block is a different-producer signal, not an error.
            lines.append('  (no ImageFXMAnalysis header columns; '
                         'a different producer wrote this)')
    except Exception as exc:
        lines.append(f'  (partly unreadable: {exc.__class__.__name__}: {exc})')
    finally:
        try:
            pf.close()
        except Exception:
            pass
    return lines


def _file_lines(obj, info: dict, path: Path,
                want_sibling: bool) -> list[str]:
    """The file-level block shown for the root node."""
    lines = [str(path), '']
    lines.append(f'  {"file size":<18}{info["size"] / (1024 * 1024):,.1f} MB')
    lines.append(f'  {"kind":<18}{type(obj).__name__}')
    if isinstance(obj, dict):
        lines.append(f'  {"top-level keys":<18}{len(obj)}')
    lines.append(f'  {"zip entries":<18}{info["entries"]:,}')
    lines.append(f'  {"storage blobs":<18}{info["blobs"]:,}')
    lines.append(f'  {"pickle":<18}{info["pickle"]} '
                 f'({info["pickle_size"]:,} bytes)')
    for key in ('byteorder', 'version'):
        if key in info:
            lines.append(f'  {key:<18}{info[key]}')

    tensors = _all_tensors(obj)
    if tensors:
        total = sum(t.nbytes or 0 for t in tensors)
        lines.append('')
        lines.append(f'  {"tensors":<18}{len(tensors):,}')
        lines.append(f'  {"tensor bytes":<18}{total / (1024 * 1024):,.1f} MB')

    if want_sibling:
        lines += _sibling_lines(path, _primary_length(obj))
    return lines


def _all_tensors(obj, depth: int = 0, seen: set | None = None) -> list[_TensorInfo]:
    """
    Every tensor in the graph, for the file-level totals.

    A saved nn.Module can hold back-references (a submodule pointing at its
    parent), so visited ids are tracked: without that this recurses forever, and
    shared tensors would be counted twice.
    """
    if seen is None:
        seen = set()
    if depth > _MAX_DEPTH or id(obj) in seen:
        return []
    seen.add(id(obj))
    if isinstance(obj, _TensorInfo):
        return [obj]
    out = []
    for _, child in children(obj):
        out += _all_tensors(child, depth + 1, seen)
    return out


# ---------------------------------------------------------------------------
# Text dump
# ---------------------------------------------------------------------------

def dump(obj, info: dict, path: Path, want_sibling: bool,
         out=sys.stdout) -> None:
    """Print the whole structure as an indented text tree."""
    for line in _file_lines(obj, info, path, want_sibling):
        print(line, file=out)

    print('', file=out)
    print('  --- structure ---', file=out)

    def walk(node, depth: int, seen: set) -> None:
        pad = '  ' * depth
        kids = children(node)
        for name, child in kids:
            desc = classify(child)
            bits = [desc['kind']]
            if desc['shape']:
                bits.append(desc['shape'])
            if desc['dtype']:
                bits.append(desc['dtype'])
            suffix = ''
            if isinstance(child, _TensorInfo):
                suffix = f'  {child.device}'
                if not child.contiguous:
                    suffix += '  non-contiguous'
            print(f"{pad}{name}  {'  '.join(bits)}{suffix}", file=out)
            if not desc['expandable']:
                continue
            if id(child) in seen:
                # A saved module can point back at its parent; printing it again
                # would loop.
                print(f'{pad}  ... already shown above', file=out)
            elif depth >= _MAX_DEPTH:
                print(f'{pad}  ... deeper nodes not shown '
                      f'(depth capped at {_MAX_DEPTH})', file=out)
            else:
                walk(child, depth + 1, seen | {id(child)})
        # children() already truncated; say so rather than silently dropping.
        if len(kids) == _MAX_KIDS:
            print(f'{pad}... further children may exist '
                  f'(capped at {_MAX_KIDS} per node)', file=out)

    walk(obj, 1, {id(obj)})

    # The stacks are the reason for the tool, so repeat them at the end where a
    # long tree won't bury them.
    if isinstance(obj, dict):
        for key, child in obj.items():
            if isinstance(child, _TensorInfo) and len(child.shape) >= 2:
                print('', file=out)
                print(f'  {key}', file=out)
                for line in _stack_lines(child):
                    print(line, file=out)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class BrowsePtApp:

    _MONO      = ('TkFixedFont', 11)
    _HDR_FONT  = ('TkDefaultFont', 10, 'bold')
    _ROOT_ID   = ''

    def __init__(self, root: tk.Tk, path: Path, obj, info: dict,
                 want_sibling: bool):
        self._root = root
        self._path = path
        self._obj  = obj
        self._info = info
        self._want_sibling = want_sibling

        # Treeview item id -> graph node. Display text can't rebuild a path when
        # names repeat at different depths, and the node itself is what the
        # details pane needs anyway.
        self._nodes: dict[str, object] = {}
        # Item id -> dotted key path, for Copy key.
        self._keys: dict[str, str] = {}

        root.title(f'browse_pt - {path.name}')
        root.geometry('1200x800')

        self._build_ui()
        self._populate()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # 'clam' renders Treeview reliably across platforms (the macOS aqua
        # theme ignores row tag styling).
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except tk.TclError:
            pass

        # ---- Top bar ----
        top = tk.Frame(self._root)
        top.pack(fill=tk.X, padx=10, pady=(8, 4))

        size_mb = self._path.stat().st_size / (1024 * 1024)
        tk.Label(top, text=f'{self._path.name}   ({size_mb:,.1f} MB)',
                 font=self._HDR_FONT, anchor='w').pack(side=tk.LEFT)

        tk.Button(top, text='Open...', command=self._open_other).pack(
            side=tk.RIGHT, padx=(6, 0))
        tk.Button(top, text='Copy key', command=self._copy_key).pack(
            side=tk.RIGHT, padx=(6, 0))
        self._sib_btn = tk.Button(top, text='Open metadata parquet',
                                  command=self._open_sibling)
        self._sib_btn.pack(side=tk.RIGHT, padx=(6, 0))
        if sibling_path(self._path) is None:
            self._sib_btn.config(state=tk.DISABLED)

        self._key_label = tk.Label(top, text='', anchor='w', fg='#444')
        self._key_label.pack(side=tk.LEFT, padx=(16, 0))

        # ---- Split: tree | details ----
        split = ttk.PanedWindow(self._root, orient=tk.HORIZONTAL)
        split.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))

        split.add(self._build_tree_pane(split), weight=3)
        split.add(self._build_details_pane(split), weight=2)

    def _build_tree_pane(self, parent) -> tk.Frame:
        frame = tk.Frame(parent)

        vsb = ttk.Scrollbar(frame, orient=tk.VERTICAL)
        hsb = ttk.Scrollbar(frame, orient=tk.HORIZONTAL)

        self._tree = ttk.Treeview(
            frame, columns=('kind', 'shape', 'dtype'),
            show='tree headings', selectmode='browse',
            yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.config(command=self._tree.yview)
        hsb.config(command=self._tree.xview)

        self._tree.heading('#0',    text='Name',  anchor='w')
        self._tree.heading('kind',  text='Kind',  anchor='w')
        self._tree.heading('shape', text='Shape / value', anchor='w')
        self._tree.heading('dtype', text='dtype', anchor='w')
        self._tree.column('#0',    width=280, minwidth=140, stretch=True)
        self._tree.column('kind',  width=140, minwidth=90,  stretch=False)
        self._tree.column('shape', width=200, minwidth=90,  stretch=False)
        self._tree.column('dtype', width=90,  minwidth=70,  stretch=False)

        self._tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self._tree.bind('<<TreeviewSelect>>', self._on_select)
        return frame

    def _build_details_pane(self, parent) -> tk.Frame:
        frame = tk.Frame(parent)

        tk.Label(frame, text='Details', font=self._HDR_FONT, anchor='w').pack(
            fill=tk.X)

        text_frame = tk.Frame(frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        tvsb = ttk.Scrollbar(text_frame, orient=tk.VERTICAL)
        thsb = ttk.Scrollbar(text_frame, orient=tk.HORIZONTAL)
        self._text = tk.Text(text_frame, wrap=tk.NONE, font=self._MONO,
                             state=tk.DISABLED,
                             yscrollcommand=tvsb.set, xscrollcommand=thsb.set)
        tvsb.config(command=self._text.yview)
        thsb.config(command=self._text.xview)
        self._text.grid(row=0, column=0, sticky='nsew')
        tvsb.grid(row=0, column=1, sticky='ns')
        thsb.grid(row=1, column=0, sticky='ew')
        text_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)

        return frame

    # ------------------------------------------------------------------
    # Tree population
    # ------------------------------------------------------------------

    def _populate(self):
        """
        Insert the whole graph eagerly.

        Unlike browse_h5's lazy expansion, the entire .pt graph is already in
        memory as metadata — the storages were never read — so there is nothing
        to defer and no placeholder machinery is needed.
        """
        root_item = self._tree.insert(
            '', 'end', text=self._path.name, open=True,
            values=(type(self._obj).__name__,
                    f'{len(self._obj)} item(s)' if isinstance(self._obj, dict)
                    else '', ''))
        self._nodes[root_item] = self._obj
        self._keys[root_item] = ''
        self._insert_children(root_item, self._obj, '', 0, {id(self._obj)})

        # Land on the root so the file block and sibling check are visible
        # without a click.
        self._tree.selection_set(root_item)
        self._tree.focus(root_item)

    def _insert_children(self, parent_id: str, node, prefix: str, depth: int,
                         seen: set):
        for name, child in children(node):
            desc = classify(child)
            key = f'{prefix}.{name}' if prefix else name
            item = self._tree.insert(
                parent_id, 'end', text=name,
                values=(desc['kind'], desc['shape'], desc['dtype']))
            self._nodes[item] = child
            self._keys[item] = key
            if not desc['expandable']:
                continue
            if id(child) in seen:
                # A saved module can point back at its parent; recursing would
                # not terminate.
                self._tree.insert(item, 'end', text='(already shown above)',
                                  values=('cycle', '', ''))
            elif depth >= _MAX_DEPTH:
                self._tree.insert(item, 'end', text='...',
                                  values=('depth capped', '', ''))
            else:
                # Tensors are the interesting leaves and the top of the graph is
                # usually shallow, so open the first level or two by default.
                self._tree.item(item, open=depth < 1)
                self._insert_children(item, child, key, depth + 1,
                                      seen | {id(child)})

    # ------------------------------------------------------------------
    # Details pane
    # ------------------------------------------------------------------

    def _write(self, lines: list[str]):
        self._text.config(state=tk.NORMAL)
        self._text.delete('1.0', tk.END)
        self._text.insert('1.0', '\n'.join(lines))
        self._text.config(state=tk.DISABLED)

    def _on_select(self, _event=None):
        sel = self._tree.selection()
        if not sel:
            return
        item = sel[0]
        if item not in self._nodes:
            return
        node = self._nodes[item]
        key = self._keys.get(item, '')
        self._key_label.config(text=key)

        if node is self._obj:
            self._write(_file_lines(self._obj, self._info, self._path,
                                    self._want_sibling))
            return

        desc = classify(node)
        lines = [key or self._tree.item(item, 'text'), '']
        lines.append(f'  {"kind":<18}{desc["kind"]}')

        if isinstance(node, _TensorInfo):
            lines.append(f'  {"shape":<18}'
                         f'{tuple(node.shape) if node.shape else "scalar"}')
            lines.append(f'  {"dtype":<18}{node.dtype}')
            lines.append(f'  {"device":<18}{node.device}')
            lines.append(f'  {"elements":<18}{node.numel:,}')
            nbytes = node.nbytes
            lines.append(f'  {"bytes":<18}'
                         f'{"?" if nbytes is None else f"{nbytes:,}"}')
            lines.append(f'  {"strides":<18}{tuple(node.stride)}')
            lines.append(f'  {"contiguous":<18}'
                         f'{"yes" if node.contiguous else "no"}')
            if node.storage is not None:
                lines.append(f'  {"storage":<18}{node.storage.storage_class}')
            if len(node.shape) >= 2:
                lines += _stack_lines(node)
        else:
            if desc['shape']:
                lines.append(f'  {"shape":<18}{desc["shape"]}')
            if isinstance(node, _SCALARS) and not isinstance(node, bytes):
                lines.append('')
                lines.append('  --- value ---')
                lines.append(f'  {_fmt_value(node)}')
            elif isinstance(node, (dict, list, tuple)):
                lines.append('')
                lines.append('  --- children ---')
                for name, child in children(node):
                    kid = classify(child)
                    bits = [b for b in (kid['kind'], kid['shape'], kid['dtype'])
                            if b]
                    lines.append(f'  {name:<24}{"  ".join(bits)}')

        self._write(lines)

    # ------------------------------------------------------------------
    # Buttons
    # ------------------------------------------------------------------

    def _copy_key(self):
        sel = self._tree.selection()
        if not sel:
            return
        key = self._keys.get(sel[0], '')
        self._root.clipboard_clear()
        self._root.clipboard_append(key)

    def _open_sibling(self):
        """Launch browse_parquet.py on the sibling metadata file."""
        sib = sibling_path(self._path)
        if sib is None:
            return
        script = Path(__file__).with_name('browse_parquet.py')
        if not script.is_file():
            messagebox.showerror('Cannot open',
                                 f'browse_parquet.py not found beside {Path(__file__).name}')
            return
        try:
            subprocess.Popen([sys.executable, str(script), str(sib)])
        except Exception as exc:
            messagebox.showerror('Cannot open', str(exc))

    def _open_other(self):
        chosen = filedialog.askopenfilename(
            title='Select a PyTorch file',
            filetypes=[('PyTorch files', '*.pt *.pth'), ('All files', '*.*')])
        if not chosen:
            return
        try:
            path = _resolve_path(chosen, allow_picker=False)
            obj, info = _read_pt(path)
        except Exception as exc:
            messagebox.showerror('Cannot open file', str(exc))
            return

        self._path, self._obj, self._info = path, obj, info
        self._root.title(f'browse_pt - {path.name}')
        # Rebuild the whole UI so the header, sibling button and panes all match
        # the new file.
        for child in self._root.winfo_children():
            child.destroy()
        self._nodes.clear()
        self._keys.clear()
        self._build_ui()
        self._populate()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()

    try:
        path = _resolve_path(args.pt_path, allow_picker=not args.dump)
    except Exception as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)

    try:
        obj, info = _read_pt(path)
    except Exception as exc:
        print(f'Error: cannot read {path} as a PyTorch file.\n  {exc}',
              file=sys.stderr)
        sys.exit(1)

    want_sibling = not args.no_sibling

    if args.dump:
        dump(obj, info, path, want_sibling)
        return

    root = tk.Tk()
    BrowsePtApp(root, path, obj, info, want_sibling)
    root.mainloop()


if __name__ == '__main__':
    main()
