"""
browse_pt.py

Structure browser for PyTorch (.pt / .pth) files. Reports the dimensions, dtype
and device of every tensor inside, interprets 3-D and 4-D tensors as image stacks,
and surfaces the sibling metadata parquet and CELLGROUPED HDF5 that accompany a
VQ-VAE crop cache.

Written for the crop caches emitted by stage 2 of the ImageFXMAnalysis pipeline.
Those files hold only image tensors — `bf_u8`, always, and `fl_u8` when
fluorescence export is enabled, both uint8 of shape (N, 1, S, S). All per-crop
metadata deliberately lives in a sibling `<stem>_metadata.parquet` with one row
per crop, row-aligned with axis 0 of the tensors. This tool therefore looks for
that sibling and reports its row count, its constant-down-file columns, and
whether its row count still matches N — a mismatch means a broken export.
SMRFXMAnalysis also writes a `<stem>.hdf5` CELLGROUPED file beside the crop
cache; `Open HDF5` launches browse_h5.py on it directly.

The file is read WITHOUT importing torch. A .pt is a zip archive holding one
pickle plus raw storage blobs; every shape and dtype lives in the pickle, so
unpickling with stubbed classes recovers the full structure without loading a
single byte of tensor data and without a ~2 GB torch install. No class named in
the file is ever resolved to real code, so nothing in the file can execute.

An Images tab beside the details pane renders the crops themselves, one row per
transit in the style of browse_images.py: a horizontally scrollable strip of that
transit's frames, five transits per page. Where browse_images reads a transit's
frames from its own HDF5 group, a crop cache is a flat stack with no such
structure — the grouping lives in the sibling parquet's `cell_group` (or
`transit_index`) column, and a transit's rows are NOT contiguous, because transits
overlap in time and the exporter writes in frame order. Rows are therefore
gathered by key and sorted by `frame_idx`, not sliced from a range.

Pixels are read straight out of the zip's storage blob by seeking to the rows a
page needs. torch.save stores each tensor's buffer as a single uncompressed entry,
so row i sits at a known offset; a page costs one seek per crop shown rather than
a load of the whole tensor, which keeps paging through a multi-gigabyte cache
instant and flat in memory — still without torch.

Usage:
    python browse_pt.py [<pt_path>] [--no-sibling] [--dump]

    <pt_path>     Path to a .pt / .pth file. If omitted, a file picker opens.
    --no-sibling  Skip the sibling metadata-parquet lookup. Also disables the
                  transit grouping in the Images tab, which comes from it.
    --dump        Print the structure to stdout and exit; no window is opened.
                  Requires <pt_path>.
"""
import argparse
import io
import pickle
import random
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

# numpy and Pillow back the Images tab only. A .pt with no viewable tensor, or a
# machine without them, must still open the structure browser.
try:
    import numpy as np
except ImportError:
    np = None

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = ImageTk = None

# matplotlib backs the Images tab's "Save sample" button only, which lays out a
# grid of transits (rows) x frames (columns) and rasterises it to a file. Guarded
# like numpy/Pillow above so a machine missing it can still browse structure.
try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
except ImportError:
    Figure = FigureCanvasAgg = None


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

# ---- Images tab ----

_ROWS_PER_PAGE = 5     # transits shown per page, as in browse_images.py
_FRAME_HEIGHT  = 80    # display height (px) per crop; width scales to aspect
_MAX_STRIP     = 400   # crops rendered in one row before the rest are elided
_RAW_PER_ROW   = 24    # crops per row when there is no transit grouping
_SAMPLE_SIZE   = 20    # transits drawn into a "Save sample" figure

# The transit key in the sibling metadata parquet, best first. cell_id is the
# grouping column written by the current (SMRFXMAnalysis-style) producer;
# cell_group/transit_index are the older ImageFXMAnalysis names, kept as a
# fallback for caches exported before that producer became the default.
_TRANSIT_KEYS = ('cell_id', 'cell_group', 'transit_index')

# Orders crops within a transit. frame_in_cell is the current producer's
# within-transit frame counter; the rest are fallbacks for older exports.
_FRAME_ORDER_KEYS = ('frame_in_cell', 'frame_idx', 'frame_number_bf', 'loop_index')

# Per-row columns worth showing beside a transit. Only those present are used.
# cls/conf are the current producer's names for classification/confidence.
_ROW_DETAIL_KEYS = ('cls', 'conf', 'classification', 'confidence', 'volume',
                    'matched_mass', 'buoyant_density', 'frame_time')

# classification-like column, best first, for the per-crop caption.
_CLASS_DETAIL_KEYS = ('cls', 'classification')

# Grouping columns added by concat_vqvae_caches.py. When present the tensor holds
# several experiments, so a transit key alone is no longer unique across the file
# and these have to join it.
_GROUP_KEYS = ('experiment', 'sample_name')

# dtypes that can be turned into a displayable image. uint8 is what the crop
# caches hold; the float and wider int cases are normalised per crop instead.
_VIEWABLE_DTYPES = ('uint8', 'int8', 'uint16', 'int16', 'int32', 'int64',
                    'float16', 'float32', 'float64', 'bool', 'bfloat16')

# numpy dtype per torch storage class, for reading raw blob bytes. bfloat16 has
# no numpy equivalent, so it is absent and reported as unviewable rather than
# silently misread as float16.
_STORAGE_NUMPY = {
    'ByteStorage': 'u1', 'CharStorage': 'i1', 'BoolStorage': 'u1',
    'ShortStorage': '<i2', 'HalfStorage': '<f2', 'IntStorage': '<i4',
    'FloatStorage': '<f4', 'LongStorage': '<i8', 'DoubleStorage': '<f8',
}



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
        """
        True when the buffer is the tensor's C-order contents.

        Mirrors torch's own compute_contiguous: axes of size 1 are skipped,
        because a single element can be reached by any stride, so its value says
        nothing about the layout. Real caches hit this — a crop stack built by
        unsqueezing a channel axis is (N, 1, S, S) with stride (S*S, 0, S, 1),
        which is byte-identical to C-order despite the 0.
        """
        if self.numel == 0:
            return True
        expected = 1
        for size, stride in zip(reversed(self.shape), reversed(self.stride)):
            size = int(size)
            if size == 1:
                continue
            if int(stride) != expected:
                return False
            expected *= size
        return True

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
# Crop reading
# ---------------------------------------------------------------------------

def viewable_tensors(obj) -> list[tuple[str, _TensorInfo]]:
    """
    (key, tensor) for every top-level tensor the Images tab can render.

    Requires rank 3 or 4 — a stack of 2-D images — plus a layout whose rows can
    be located by arithmetic: contiguous, offset 0, and exactly its own storage.
    Anything else would need real strided indexing, which is torch's job.
    """
    if not isinstance(obj, dict):
        return []
    out = []
    for key, node in obj.items():
        if not isinstance(node, _TensorInfo) or node.storage is None:
            continue
        if len(node.shape) not in (3, 4):
            continue
        if node.dtype not in _VIEWABLE_DTYPES:
            continue
        if node.storage.storage_class not in _STORAGE_NUMPY:
            continue
        if not node.contiguous or node.offset:
            continue
        if node.storage.numel != node.numel:
            continue
        out.append((key, node))
    return out


def _frame_shape(info: _TensorInfo) -> tuple[int, int]:
    """(H, W) of one image in the stack, collapsing a 4-D channel axis."""
    shape = tuple(int(d) for d in info.shape)
    return (shape[-2], shape[-1])


class CropReader:
    """
    Reads individual image rows out of a .pt storage blob.

    A torch.save archive stores each tensor's buffer as a single STORED (never
    deflated) zip entry, so the bytes of row i sit at a known offset and can be
    seeked to directly. That is what makes paging through a 300k-crop cache
    instant: a page costs one seek and one read per crop shown, not a load of the
    whole tensor.

    The archive and the entry stay open for the reader's lifetime, since reopening
    per crop would dominate the cost of drawing a page.
    """

    def __init__(self, pt_path: Path, info: _TensorInfo):
        self._path = pt_path
        self._info = info
        self._h, self._w = _frame_shape(info)
        shape = tuple(int(d) for d in info.shape)
        # A 4-D (N, C, H, W) crop cache is displayed one channel at a time; C is
        # 1 in every cache this tool targets, and channel 0 is the meaningful one.
        self._channels = shape[1] if len(shape) == 4 else 1
        self._n = shape[0]
        self._np_dtype = _STORAGE_NUMPY[info.storage.storage_class]
        self._itemsize = _STORAGE_ITEMSIZE[info.storage.storage_class]
        self._frame_bytes = self._h * self._w * self._itemsize
        # Stride between rows: a 4-D row holds every channel.
        self._row_bytes = self._frame_bytes * self._channels

        self._zf = zipfile.ZipFile(str(pt_path))
        self._entry = None
        self._fh = None
        try:
            self._entry = self._blob_name()
            self._fh = self._zf.open(self._entry)
            if not self._fh.seekable():
                raise ValueError('storage entry is not seekable '
                                 '(the archive is compressed)')
        except Exception:
            self.close()
            raise

    def _blob_name(self) -> str:
        """The archive entry holding this tensor's storage."""
        suffix = f'data/{self._info.storage.key}'
        for name in self._zf.namelist():
            if name == suffix or name.endswith(f'/{suffix}'):
                return name
        raise KeyError(f'no storage blob {self._info.storage.key!r} in '
                       f'{self._path.name}')

    @property
    def n(self) -> int:
        return self._n

    @property
    def frame_shape(self) -> tuple[int, int]:
        return (self._h, self._w)

    def read(self, row: int, channel: int = 0):
        """
        One image as a 2-D uint8 array ready for display.

        Anything that is not already uint8 is min-max normalised per crop: the
        alternative is a black or saturated image, since a float crop's range is
        arbitrary and a 16-bit one uses only part of its span.
        """
        if not 0 <= row < self._n:
            raise IndexError(f'row {row} out of range (N={self._n})')
        channel = min(max(channel, 0), self._channels - 1)
        offset = row * self._row_bytes + channel * self._frame_bytes
        self._fh.seek(offset)
        raw = self._fh.read(self._frame_bytes)
        if len(raw) < self._frame_bytes:
            raise ValueError(f'storage ended early at row {row} '
                             f'({len(raw)} of {self._frame_bytes} bytes)')
        flat = np.frombuffer(raw, dtype=np.dtype(self._np_dtype))
        frame = flat.reshape(self._h, self._w)

        if frame.dtype == np.uint8:
            return frame
        if frame.dtype == np.bool_:
            return frame.astype(np.uint8) * 255
        data = frame.astype(np.float64)
        lo = float(np.nanmin(data))
        hi = float(np.nanmax(data))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            # A constant (or all-NaN) crop has no range to stretch; mid-grey
            # shows it exists rather than implying detail that isn't there.
            return np.full((self._h, self._w), 128, dtype=np.uint8)
        return np.clip((data - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)

    def close(self):
        for handle in (self._fh, self._zf):
            try:
                if handle is not None:
                    handle.close()
            except Exception:
                pass
        self._fh = None
        self._zf = None


# ---------------------------------------------------------------------------
# Transit index
# ---------------------------------------------------------------------------

class TransitIndex:
    """
    Groups a cache's rows into transits, read from the sibling metadata parquet.

    The .pt itself is a flat (N, C, S, S) stack with no notion of a transit; the
    grouping lives entirely in the sibling, whose row i describes tensor row i.
    So this reads the transit key column, gathers the rows sharing each value, and
    hands back per-transit row lists that index straight into the tensor.

    Rows of one transit are NOT contiguous in the file. Transits overlap in time,
    and the exporter writes in frame order, so transit 533's rows interleave with
    534's. Gathering by value rather than slicing a range is what makes the strips
    correct.
    """

    def __init__(self, labels: list[str], rows: list[list[int]],
                 key_column: str, detail: dict[str, list]):
        self.labels     = labels
        self.rows       = rows
        self.key_column = key_column
        self.detail     = detail

    def __len__(self) -> int:
        return len(self.labels)

    @classmethod
    def build(cls, parquet_path: Path, nrows: int) -> tuple['TransitIndex | None', str]:
        """
        Read the sibling and group its rows. Returns (index, note).

        The note explains a None, or qualifies a successful build, so the UI can
        say why it fell back to ungrouped rows instead of showing nothing.
        """
        if pq is None:
            return None, 'pyarrow is not installed, so the sibling cannot be read'
        try:
            pf = pq.ParquetFile(str(parquet_path))
            names = [f.name for f in pf.schema_arrow]
            file_rows = pf.metadata.num_rows
        except Exception as exc:
            return None, f'cannot read {parquet_path.name}: {exc}'

        if file_rows != nrows:
            # Position is the only join between the two files, so a length
            # mismatch means any grouping built here would mislabel crops.
            pf.close()
            return None, (f'{parquet_path.name} has {file_rows:,} rows but the '
                          f'tensor has N={nrows:,}, so rows cannot be matched')

        key = next((k for k in _TRANSIT_KEYS if k in names), None)
        if key is None:
            pf.close()
            return None, (f'{parquet_path.name} has no '
                          f'{" or ".join(_TRANSIT_KEYS)} column to group by')

        order_key = next((k for k in _FRAME_ORDER_KEYS if k in names), None)
        group_keys = [k for k in _GROUP_KEYS if k in names]
        details = [k for k in _ROW_DETAIL_KEYS if k in names]
        wanted = [key] + group_keys + ([order_key] if order_key else []) + details

        try:
            table = pf.read(columns=wanted)
        except Exception as exc:
            return None, f'cannot read columns from {parquet_path.name}: {exc}'
        finally:
            try:
                pf.close()
            except Exception:
                pass

        keys = [_fmt_label(v) for v in table.column(key).to_pylist()]
        # A concatenated cache repeats transit labels across samples, so the label
        # alone would merge unrelated crops into one strip.
        if group_keys:
            parts = [[_fmt_label(v) for v in table.column(g).to_pylist()]
                     for g in group_keys]
            keys = ['/'.join(list(bits) + [k])
                    for *bits, k in zip(*parts, keys)]

        order = (table.column(order_key).to_pylist() if order_key
                 else list(range(nrows)))

        groups: dict[str, list[int]] = {}
        for row, label in enumerate(keys):
            groups.setdefault(label, []).append(row)

        labels = list(groups.keys())
        rows = []
        for label in labels:
            member = groups[label]
            # Sort by the frame counter so a strip reads left-to-right in time
            # even though the rows are interleaved on disk. Row index breaks ties.
            member.sort(key=lambda r: (_sort_key(order[r]), r))
            rows.append(member)

        detail = {name: table.column(name).to_pylist() for name in details}
        note = f'grouped by {key}'
        if group_keys:
            note += f' within {" / ".join(group_keys)}'
        if order_key:
            note += f', ordered by {order_key}'
        else:
            note += ', in file order (no frame counter column)'
        return cls(labels, rows, key, detail), note


def _fmt_label(value) -> str:
    """A transit key value as display text, with None made visible."""
    if value is None:
        return '(none)'
    return str(value)


def _sort_key(value):
    """
    Sortable form of a frame-order value.

    Mixed or missing values must not raise mid-sort, so anything non-numeric is
    ordered after the numbers rather than compared against them. Ties are broken
    by row index at the call site, which keeps the sort stable and predictable.
    """
    if isinstance(value, (int, float, bool)):
        return (0, float(value))
    return (1, 0.0)


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


def sibling_hdf5_path(path: Path) -> Path | None:
    """
    The CELLGROUPED HDF5 SMRFXMAnalysis writes beside its crop cache.

    Same stem, different extension: <base_id>_CELLGROUPED.pt sits next to
    <base_id>_CELLGROUPED.hdf5.
    """
    candidate = path.with_suffix('.hdf5')
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
# Images tab
# ---------------------------------------------------------------------------

class ImagePane:
    """
    Paginated per-transit crop viewer, in the style of browse_images.py.

    One row per transit, _ROWS_PER_PAGE rows per page, each row a horizontally
    scrollable strip of that transit's crops in frame order. Where browse_images
    reads a transit's frames from its own HDF5 group, here the grouping comes from
    the sibling parquet and the pixels are seeked out of the .pt storage blob, so
    only the crops actually on screen are ever read.

    Falls back to paginating raw tensor rows when there is no usable sibling: the
    tensor is still viewable, just without transit labels.
    """

    _ROWS       = _ROWS_PER_PAGE
    _FH         = _FRAME_HEIGHT
    _LABEL_FONT = ('TkDefaultFont', 10)
    _NAV_FONT   = ('TkDefaultFont', 11)
    _NAV_FONT_B = ('TkDefaultFont', 11, 'bold')
    _BG         = '#f0f0f0'

    def __init__(self, parent: tk.Frame):
        self._frame = tk.Frame(parent)

        self._reader: CropReader | None = None
        self._index: TransitIndex | None = None
        self._tensor_key = ''
        self._channel = 0
        self._channels = 1
        self._note = ''
        # Per-tensor page memory, so switching tensors and back keeps position.
        self._pages: dict[str, int] = {}
        # PhotoImage objects must be referenced or Tk garbage-collects them and
        # draws blank labels.
        self._photo_refs: list = []
        self._tensor_keys: list[str] = []
        self._on_pick_tensor = None

        self._build_ui()

    @property
    def widget(self) -> tk.Frame:
        return self._frame

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        top = tk.Frame(self._frame)
        top.pack(fill=tk.X, pady=(0, 4))

        tk.Label(top, text='Tensor:', font=self._NAV_FONT).pack(side=tk.LEFT)
        self._tensor_var = tk.StringVar()
        self._tensor_cb = ttk.Combobox(top, textvariable=self._tensor_var,
                                       state='readonly', width=14,
                                       font=self._NAV_FONT)
        self._tensor_cb.pack(side=tk.LEFT, padx=(6, 12))
        self._tensor_cb.bind('<<ComboboxSelected>>', self._on_tensor_change)

        # Only shown for a multi-channel stack; a 1-channel crop cache has no
        # choice to offer and the control would be noise.
        self._channel_label = tk.Label(top, text='Channel:', font=self._NAV_FONT)
        self._channel_var = tk.StringVar(value='0')
        self._channel_cb = ttk.Combobox(top, textvariable=self._channel_var,
                                        state='readonly', width=4,
                                        font=self._NAV_FONT)
        self._channel_cb.bind('<<ComboboxSelected>>', self._on_channel_change)

        self._info_label = tk.Label(top, text='', font=self._NAV_FONT, anchor='w')
        self._info_label.pack(side=tk.LEFT)

        # Needs a transit index to have rows to put in the grid, so it lives
        # disabled until one is built.
        self._sample_btn = tk.Button(top, text='Save sample...',
                                     command=self._save_sample, state=tk.DISABLED)
        self._sample_btn.pack(side=tk.RIGHT)

        # Rows live in a scrollable column: five strips of tall crops overflow a
        # short pane, and the tab shares its height with the tree.
        body = tk.Frame(self._frame, bg=self._BG)
        body.pack(fill=tk.BOTH, expand=True)

        self._vcanvas = tk.Canvas(body, bg=self._BG, highlightthickness=0)
        vsb = ttk.Scrollbar(body, orient=tk.VERTICAL,
                            command=self._vcanvas.yview)
        self._vcanvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._vcanvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._rows_host = tk.Frame(self._vcanvas, bg=self._BG)
        self._host_id = self._vcanvas.create_window((0, 0), window=self._rows_host,
                                                    anchor='nw')
        self._rows_host.bind(
            '<Configure>',
            lambda e: self._vcanvas.configure(
                scrollregion=self._vcanvas.bbox('all')))
        self._vcanvas.bind(
            '<Configure>',
            lambda e: self._vcanvas.itemconfig(self._host_id, width=e.width))

        self._row_widgets = [self._make_row(self._rows_host)
                             for _ in range(self._ROWS)]

        # Shown instead of the rows when there is nothing to display.
        self._empty_label = tk.Label(self._frame, text='', fg='#555',
                                     font=self._NAV_FONT, justify=tk.LEFT,
                                     wraplength=520)

        bot = tk.Frame(self._frame)
        bot.pack(fill=tk.X, pady=(4, 0))
        self._nav_frame = bot
        self._prev_btn = tk.Button(bot, text='← Prev', font=self._NAV_FONT_B,
                                   width=9, command=self._prev)
        self._prev_btn.pack(side=tk.LEFT)
        self._next_btn = tk.Button(bot, text='Next →', font=self._NAV_FONT_B,
                                   width=9, command=self._next)
        self._next_btn.pack(side=tk.LEFT, padx=(6, 0))
        self._nav_label = tk.Label(bot, text='', font=self._NAV_FONT)
        self._nav_label.pack(side=tk.LEFT, padx=16)
        # Its own full-width line: sharing the nav row clipped it, and how the
        # rows were grouped is the caption a strip needs to be read correctly.
        self._note_label = tk.Label(self._frame, text='', font=self._LABEL_FONT,
                                    fg='#666', anchor='w')
        self._note_label.pack(fill=tk.X)

    def _make_row(self, parent: tk.Frame) -> dict:
        """Build one transit row: a fixed label column plus a scrolling strip."""
        bg = self._BG
        row = tk.Frame(parent, bg=bg, relief=tk.GROOVE, bd=1)
        row.pack(fill=tk.X, pady=3, padx=2)

        side = tk.Frame(row, bg=bg)
        side.pack(side=tk.LEFT, fill=tk.Y, padx=(6, 4))
        lbl = tk.Label(side, text='', width=18, anchor='w',
                       font=('TkDefaultFont', 10, 'bold'), bg=bg)
        lbl.pack(anchor='w')
        # wraplength keeps a long experiment/sample qualifier inside the column
        # instead of pushing the strip off the pane.
        sub = tk.Label(side, text='', width=18, anchor='w',
                       font=self._LABEL_FONT, fg='#555', bg=bg,
                       justify=tk.LEFT, wraplength=130)
        sub.pack(anchor='w')

        canvas = tk.Canvas(row, height=self._FH + 20, bg=bg,
                           highlightthickness=0)
        scroll = tk.Scrollbar(row, orient=tk.HORIZONTAL, command=canvas.xview)
        canvas.configure(xscrollcommand=scroll.set)
        scroll.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.pack(side=tk.LEFT, fill=tk.X, expand=True)

        inner = tk.Frame(canvas, bg=bg)
        win = canvas.create_window((0, 0), window=inner, anchor='nw')
        inner.bind('<Configure>',
                   lambda e, c=canvas: c.configure(scrollregion=c.bbox('all')))
        canvas.bind('<Configure>',
                    lambda e, c=canvas, w=win: c.itemconfig(w, height=e.height))
        # The strip scrolls horizontally under the wheel, which is what a row of
        # frames wants; the page itself scrolls with the scrollbar.
        for widget in (canvas, inner):
            widget.bind('<MouseWheel>',
                        lambda e, c=canvas: c.xview_scroll(
                            -1 * (e.delta // 120), 'units'))

        return {'row': row, 'label': lbl, 'sub': sub,
                'canvas': canvas, 'inner': inner}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def set_source(self, pt_path: Path, obj, want_sibling: bool):
        """Point the pane at a file, choosing the first viewable tensor."""
        self.close()
        self._pages.clear()
        self._pt_path = pt_path
        self._obj = obj
        self._want_sibling = want_sibling
        self._viewable = dict(viewable_tensors(obj))
        self._tensor_keys = list(self._viewable)

        self._tensor_cb.config(values=self._tensor_keys)
        if not self._tensor_keys:
            self._tensor_var.set('')
            self._tensor_cb.config(state=tk.DISABLED)
            self._show_empty(self._no_tensor_reason())
            return
        self._tensor_cb.config(state='readonly')
        # bf_u8 is the crop cache's primary stack, so prefer it when present.
        first = next((k for k in _PRIMARY_KEYS if k in self._viewable),
                     self._tensor_keys[0])
        self._tensor_var.set(first)
        self._load_tensor(first)

    def _no_tensor_reason(self) -> str:
        """Why nothing can be shown, specific enough to act on."""
        if np is None or Image is None:
            missing = [n for n, m in (('numpy', np), ('Pillow', Image))
                       if m is None]
            return (f'The image viewer needs {" and ".join(missing)}, which '
                    f'{"is" if len(missing) == 1 else "are"} not installed in '
                    f'this environment.')
        if not isinstance(self._obj, dict):
            return (f'This file holds a {type(self._obj).__name__} at the top '
                    f'level, not a dict of tensors, so there is no crop stack '
                    f'to display.')

        tensors = [(k, v) for k, v in self._obj.items()
                   if isinstance(v, _TensorInfo)]
        if not tensors:
            return 'This file holds no top-level tensors.'

        # Say which requirement each tensor failed rather than a bare "nothing to
        # show": on a model checkpoint the answer is simply that weights are not
        # images, and that is worth stating.
        bits = []
        for key, info in tensors[:8]:
            shape = tuple(int(d) for d in info.shape)
            if len(shape) not in (3, 4):
                why = f'rank {len(shape)} is not an image stack'
            elif info.dtype not in _VIEWABLE_DTYPES:
                why = f'dtype {info.dtype} is not displayable'
            elif info.storage is None:
                why = 'has no storage'
            elif info.storage.storage_class not in _STORAGE_NUMPY:
                why = f'{info.storage.storage_class} has no numpy equivalent'
            elif not info.contiguous or info.offset:
                why = 'is not a contiguous, zero-offset buffer'
            elif info.storage.numel != info.numel:
                why = 'is a view into a larger storage'
            else:
                why = 'cannot be read'
            bits.append(f'  {key}  {shape}  —  {why}')
        more = ('\n  ...' if len(tensors) > 8 else '')
        return ('No tensor in this file can be displayed as an image stack:\n'
                + '\n'.join(bits) + more)

    def _load_tensor(self, key: str):
        """Open a reader for one tensor and build its transit index."""
        self._close_reader()
        info = self._viewable.get(key)
        if info is None:
            self._show_empty(self._no_tensor_reason())
            return

        self._tensor_key = key
        try:
            self._reader = CropReader(self._pt_path, info)
        except Exception as exc:
            self._show_empty(f'Cannot read {key} from {self._pt_path.name}:\n'
                             f'  {exc}')
            return

        shape = tuple(int(d) for d in info.shape)
        self._channels = shape[1] if len(shape) == 4 else 1
        self._channel = 0
        if self._channels > 1:
            self._channel_cb.config(
                values=[str(i) for i in range(self._channels)])
            self._channel_var.set('0')
            self._channel_label.pack(side=tk.LEFT, before=self._info_label)
            self._channel_cb.pack(side=tk.LEFT, padx=(6, 12),
                                  before=self._info_label)
        else:
            self._channel_label.pack_forget()
            self._channel_cb.pack_forget()

        self._index, self._note = None, ''
        sib = sibling_path(self._pt_path) if self._want_sibling else None
        if sib is not None:
            self._index, self._note = TransitIndex.build(sib, self._reader.n)
            if self._index is None:
                self._note = f'ungrouped: {self._note}'
        elif self._want_sibling:
            self._note = ('ungrouped: no '
                          f'{self._pt_path.stem}_metadata.parquet beside this file')
        else:
            self._note = 'ungrouped: --no-sibling was given'

        can_sample = (self._index is not None and len(self._index) > 0
                     and Figure is not None)
        self._sample_btn.config(state=tk.NORMAL if can_sample else tk.DISABLED)

        self._show_page()

    # ------------------------------------------------------------------
    # Paging
    # ------------------------------------------------------------------

    @property
    def _page(self) -> int:
        return self._pages.get(self._tensor_key, 0)

    @_page.setter
    def _page(self, value: int):
        self._pages[self._tensor_key] = value

    def _n_units(self) -> int:
        """Rows on the page's axis: transits when grouped, crop-chunks when not."""
        if self._reader is None:
            return 0
        if self._index is not None:
            return len(self._index)
        return (self._reader.n + _RAW_PER_ROW - 1) // _RAW_PER_ROW

    def _max_page(self) -> int:
        return max(0, (self._n_units() - 1) // self._ROWS)

    def _unit(self, i: int) -> tuple[str, str, list[int]]:
        """
        (label, sublabel, tensor rows) for one display row.

        A concatenated cache's labels are qualified with experiment and sample, so
        the transit goes on the bold line and its qualifier underneath — the label
        column is too narrow to show the whole path on one line. Only the last
        qualifier segment (the sample) is shown, because the experiment is
        identical down the whole page and the full path is one hover away in the
        Details tab.
        """
        if self._index is not None:
            full = self._index.labels[i]
            rows = self._index.rows[i]
            parts = full.split('/')
            sub = f'{len(rows)} frame(s)'
            if len(parts) > 1:
                sub = f'{parts[-2]}\n{sub}'
            return parts[-1], sub, rows
        start = i * _RAW_PER_ROW
        end = min(start + _RAW_PER_ROW, self._reader.n)
        return (f'rows {start:,}–', f'{end - 1:,}\n{end - start} crop(s)',
                list(range(start, end)))

    def _show_empty(self, message: str):
        """Hide the rows and explain why, without tearing down the pane."""
        self._photo_refs.clear()
        self._reader = None
        self._index = None
        self._vcanvas.master.pack_forget()
        # before= keeps the message above the nav row rather than below it, since
        # the nav row was packed first.
        self._empty_label.pack(fill=tk.BOTH, expand=True, padx=12, pady=12,
                               before=self._nav_frame)
        self._empty_label.config(text=message)
        self._info_label.config(text='')
        self._nav_label.config(text='')
        self._note_label.config(text='')
        self._prev_btn.config(state=tk.DISABLED)
        self._next_btn.config(state=tk.DISABLED)
        self._sample_btn.config(state=tk.DISABLED)

    def _show_page(self):
        if self._reader is None:
            return
        # Restore the row area if an earlier file had nothing to show.
        self._empty_label.pack_forget()
        if not self._vcanvas.master.winfo_ismapped():
            self._vcanvas.master.pack(fill=tk.BOTH, expand=True,
                                      before=self._nav_frame)

        self._photo_refs.clear()
        n_units = self._n_units()
        self._page = min(self._page, self._max_page())
        start = self._page * self._ROWS
        end = min(start + self._ROWS, n_units)

        unit_name = 'transit' if self._index is not None else 'block'
        self._info_label.config(
            text=f'{self._tensor_key}   {self._reader.n:,} crops   '
                 f'{n_units:,} {unit_name}(s)   '
                 f'{"×".join(str(d) for d in self._reader.frame_shape)}')
        self._nav_label.config(
            text=f'Page {self._page + 1} / {self._max_page() + 1}'
                 f'   ({unit_name}s {start + 1:,}–{end:,})')
        self._note_label.config(text=self._note)
        self._prev_btn.config(
            state=tk.NORMAL if self._page > 0 else tk.DISABLED)
        self._next_btn.config(
            state=tk.NORMAL if self._page < self._max_page() else tk.DISABLED)

        for i, rw in enumerate(self._row_widgets):
            for child in rw['inner'].winfo_children():
                child.destroy()
            if start + i >= end:
                rw['label'].config(text='')
                rw['sub'].config(text='')
                rw['row'].pack_forget()
                continue
            rw['row'].pack(fill=tk.X, pady=3, padx=2)
            label, sub, rows = self._unit(start + i)
            rw['label'].config(text=label)
            rw['sub'].config(text=sub)
            self._fill_strip(rw['inner'], rows)
            rw['canvas'].xview_moveto(0)

    def _fill_strip(self, inner: tk.Frame, rows: list[int]):
        """Render one transit's crops left to right."""
        h, w = self._reader.frame_shape
        dh = self._FH
        dw = max(1, round(w / h * dh)) if h else dh

        shown = rows[:_MAX_STRIP]
        for row in shown:
            cell = tk.Frame(inner, bg=self._BG)
            cell.pack(side=tk.LEFT, padx=2, pady=(2, 0))
            try:
                frame = self._reader.read(row, self._channel)
                # NEAREST keeps 32x32 crops readable as pixels; a smooth filter
                # would invent detail that isn't in the data.
                photo = ImageTk.PhotoImage(
                    Image.fromarray(frame, mode='L').resize((dw, dh),
                                                            Image.NEAREST))
                self._photo_refs.append(photo)
                tk.Label(cell, image=photo, bg=self._BG, relief=tk.SOLID,
                         bd=1).pack()
            except Exception as exc:
                tk.Label(cell, text='(err)', fg='#b00', bg=self._BG,
                         font=self._LABEL_FONT, width=5,
                         relief=tk.SOLID, bd=1).pack()
                tk.Label(cell, text=str(exc)[:12], fg='#b00', bg=self._BG,
                         font=('TkDefaultFont', 7)).pack()
                continue
            tk.Label(cell, text=self._caption(row), bg=self._BG, fg='#444',
                     font=('TkDefaultFont', 7)).pack()

        if len(rows) > len(shown):
            tk.Label(inner, text=f'+{len(rows) - len(shown):,} more',
                     bg=self._BG, fg='#666', font=self._LABEL_FONT).pack(
                side=tk.LEFT, padx=8)

    def _caption(self, row: int) -> str:
        """
        The line under one crop: its tensor row, plus a per-row metadata value.

        The classification-like column is what marks a crop as a usable cell,
        so it is appended when the sibling has one, under whichever name the
        producer used (cls is current, classification is the older name).
        """
        text = str(row)
        if self._index is not None:
            key = next((k for k in _CLASS_DETAIL_KEYS if k in self._index.detail),
                      None)
            values = self._index.detail.get(key) if key else None
            if values is not None and row < len(values) and values[row] is not None:
                text += f' c{values[row]}'
        return text

    def _save_sample(self):
        """
        Save a grid of _SAMPLE_SIZE transits (rows) x their frames (columns) to
        an image file, in the same tensor/channel currently on screen.

        Randomly sampled rather than the first page: the first _SAMPLE_SIZE
        transits in file order are whatever the exporter happened to write
        first, not representative of the cache as a whole.
        """
        if self._index is None or self._reader is None:
            return
        n_transits = len(self._index)
        k = min(_SAMPLE_SIZE, n_transits)
        chosen = sorted(random.sample(range(n_transits), k))

        out_path = filedialog.asksaveasfilename(
            title='Save transit sample',
            initialdir=str(self._pt_path.parent),
            initialfile=f'{self._pt_path.stem}_{self._tensor_key}_sample.png',
            defaultextension='.png',
            filetypes=[('PNG image', '*.png'), ('PDF', '*.pdf'),
                      ('SVG', '*.svg'), ('All files', '*.*')])
        if not out_path:
            return

        try:
            self._render_sample(chosen, Path(out_path))
        except Exception as exc:
            messagebox.showerror('Save sample', f'Could not save sample:\n{exc}')
            return
        messagebox.showinfo('Save sample',
                            f'Saved {k} transit(s) to {Path(out_path).name}')

    def _render_sample(self, transit_indices: list[int], out_path: Path):
        """Render the chosen transits into a rows-of-frames grid and write it out."""
        rows = [self._index.rows[i] for i in transit_indices]
        labels = [self._index.labels[i] for i in transit_indices]
        n_rows = len(rows)
        n_cols = max(1, max(len(r) for r in rows))

        fig = Figure(figsize=(max(n_cols * 1.1, 4), max(n_rows * 1.1, 2)))
        FigureCanvasAgg(fig)
        axes = fig.subplots(n_rows, n_cols, squeeze=False)

        for r, (label, tensor_rows) in enumerate(zip(labels, rows)):
            for c in range(n_cols):
                ax = axes[r][c]
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if c < len(tensor_rows):
                    frame = self._reader.read(tensor_rows[c], self._channel)
                    ax.imshow(frame, cmap='gray', vmin=0, vmax=255)
                else:
                    ax.set_facecolor('#eeeeee')
                if r == 0:
                    ax.set_title(str(c), fontsize=6)
            axes[r][0].set_ylabel(label, rotation=0, ha='right', va='center',
                                  fontsize=6)

        chan = f'  ch{self._channel}' if self._channels > 1 else ''
        fig.suptitle(f'{self._pt_path.name}  —  {self._tensor_key}{chan}  '
                    f'({n_rows} transits x {n_cols} frames)')
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(str(out_path), dpi=150)

    def _prev(self):
        if self._page > 0:
            self._page -= 1
            self._show_page()

    def _next(self):
        if self._page < self._max_page():
            self._page += 1
            self._show_page()

    def _on_tensor_change(self, _event=None):
        key = self._tensor_var.get()
        if key and key != self._tensor_key:
            self._load_tensor(key)

    def _on_channel_change(self, _event=None):
        try:
            channel = int(self._channel_var.get())
        except ValueError:
            return
        if channel != self._channel:
            self._channel = channel
            self._show_page()

    def select_tensor(self, key: str) -> bool:
        """Switch to a tensor by name; False if it isn't viewable."""
        if key not in getattr(self, '_viewable', {}):
            return False
        if key != self._tensor_key:
            self._tensor_var.set(key)
            self._load_tensor(key)
        return True

    def _close_reader(self):
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def close(self):
        """Release the archive handle. Safe to call more than once."""
        self._close_reader()
        self._photo_refs.clear()


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
        # The Images tab opens the archive and reads the sibling, so that work is
        # deferred until the tab is actually looked at.
        self._images: ImagePane | None = None
        self._images_loaded = False

        root.title(f'browse_pt - {path.name}')
        root.geometry('1500x850')

        self._build_ui()
        self._populate()
        # Releasing the archive handle on close matters on Windows, where an open
        # handle keeps the file locked against other tools.
        root.protocol('WM_DELETE_WINDOW', self._on_close)

    def _on_close(self):
        if self._images is not None:
            self._images.close()
        self._root.destroy()

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
        self._img_btn = tk.Button(top, text='View images',
                                  command=self._view_images)
        self._img_btn.pack(side=tk.RIGHT, padx=(6, 0))
        if not viewable_tensors(self._obj) or np is None or Image is None:
            self._img_btn.config(state=tk.DISABLED)
        self._sib_btn = tk.Button(top, text='Open metadata parquet',
                                  command=self._open_sibling)
        self._sib_btn.pack(side=tk.RIGHT, padx=(6, 0))
        if sibling_path(self._path) is None:
            self._sib_btn.config(state=tk.DISABLED)
        self._hdf5_btn = tk.Button(top, text='Open HDF5',
                                   command=self._open_sibling_hdf5)
        self._hdf5_btn.pack(side=tk.RIGHT, padx=(6, 0))
        if sibling_hdf5_path(self._path) is None:
            self._hdf5_btn.config(state=tk.DISABLED)

        self._key_label = tk.Label(top, text='', anchor='w', fg='#444')
        self._key_label.pack(side=tk.LEFT, padx=(16, 0))

        # ---- Split: tree | details ----
        split = ttk.PanedWindow(self._root, orient=tk.HORIZONTAL)
        split.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))

        split.add(self._build_tree_pane(split), weight=3)
        # Weighted toward the right so a strip of crops has room; the tree's
        # columns are fixed-width and the divider is draggable either way.
        split.add(self._build_right_pane(split), weight=5)

    def _build_right_pane(self, parent) -> ttk.Notebook:
        """
        Details and Images as tabs.

        A notebook rather than a second window: the tree stays visible, so
        selecting a tensor and flipping to its crops is one click, and there is
        only one window to manage.
        """
        book = ttk.Notebook(parent)
        book.add(self._build_details_pane(book), text='Details')

        self._images = ImagePane(book)
        book.add(self._images.widget, text='Images')
        self._book = book
        book.bind('<<NotebookTabChanged>>', self._on_tab_change)
        return book

    def _on_tab_change(self, _event=None):
        """
        Load the Images tab the first time it is opened.

        Deferred because it opens the archive and reads the sibling parquet, which
        should not be paid by someone who only wants the structure tree.
        """
        if self._book.index('current') != 1 or self._images_loaded:
            return
        self._images_loaded = True
        self._images.set_source(self._path, self._obj, self._want_sibling)
        # If a tensor was already selected in the tree, show that one.
        sel = self._tree.selection()
        if sel:
            key = self._keys.get(sel[0], '')
            if key and '.' not in key:
                self._images.select_tensor(key)

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

    def _view_images(self):
        """Switch to the Images tab, on the selected tensor when there is one."""
        sel = self._tree.selection()
        key = self._keys.get(sel[0], '') if sel else ''
        # Tab change triggers the deferred load, so select the tab first.
        self._book.select(1)
        if key and '.' not in key and self._images is not None:
            self._images.select_tensor(key)

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

    def _open_sibling_hdf5(self):
        """Launch browse_h5.py on the sibling CELLGROUPED HDF5 file."""
        sib = sibling_hdf5_path(self._path)
        if sib is None:
            return
        script = Path(__file__).with_name('browse_h5.py')
        if not script.is_file():
            messagebox.showerror('Cannot open',
                                 f'browse_h5.py not found beside {Path(__file__).name}')
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
        # Release the previous file's archive handle before its widgets go away.
        if self._images is not None:
            self._images.close()
            self._images = None
        self._images_loaded = False
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
