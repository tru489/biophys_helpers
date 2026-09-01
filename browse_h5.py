"""
browse_h5.py

Generic structure browser for HDF5 (.h5 / .hdf5) files. Opens any HDF5 file and
displays its hierarchy as a lazily-expanded tree, reporting the kind, shape,
dtype and attributes of every node.

Unlike browse_images.py and browse_experiment.py — which each expect one specific
schema — this tool makes no assumptions about layout. It is meant for answering
"what is actually inside this file?" for files whose structure you no longer
remember.

Objects written by pandas (pd.HDFStore / DataFrame.to_hdf) are shown logically:
a stored DataFrame appears as a single leaf with its real row/column counts and
column dtypes, rather than as the PyTables internals (axis0, block0_values,
table) that back it. Tick "Show raw HDF5 nodes" to see the underlying hierarchy
verbatim.

Selecting a node with rows -- a pandas DataFrame/Series, or a raw compound
dataset such as a CELLGROUPED file's meta/frames -- fills the Data pane with a
paged, column-labelled table, the same way browse_parquet.py shows a parquet's
rows: Prev/Next and a "go to row" box page through the whole dataset without
loading it all into memory.

Usage:
    python browse_h5.py [<h5_path>] [--raw] [--dump]

    <h5_path>  Path to a .h5 / .hdf5 file. If omitted, a file picker opens.
    --raw      Start in raw-HDF5 mode (no pandas grouping).
    --dump     Print the structure to stdout and exit; no window is opened.
"""
import argparse
import os
import sys
import warnings
from pathlib import Path

# Both h5py and PyTables (via pandas) open the same file in this process, and
# the HDF5 library refuses a second open whose locking flag disagrees with the
# first ("file locking flag values don't match"). h5py takes a per-open
# locking= kwarg but PyTables does not, so the only way to make them agree is
# this library-wide env var. It must be set before h5py loads the HDF5 library.
# Disabling locking is also what lets us read files that are open elsewhere or
# that live on network / cloud-synced volumes.
os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

import h5py
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from fsutil import is_appledouble

# pandas/PyTables warn on every access for keys that aren't valid Python
# identifiers (e.g. sample names like "A_ctrl-1"), which is most of ours.
warnings.filterwarnings('ignore', message='object name is not a valid Python identifier')


_ROWS_PER_PAGE  = 200      # rows per page in the Data view
_FULL_READ_CAP  = 200_000  # don't fall back to a whole-frame read above this many rows
_ATTR_MAXLEN    = 200      # truncate attribute reprs to this many chars
_CELL_MAXLEN    = 120      # truncate long cell reprs to this many chars
_DUMP_MAXKIDS   = 50       # children printed per group in --dump mode

# PyTables/pandas internal node names, hidden in logical mode.
_PANDAS_KINDS = ('frame', 'frame_table', 'series', 'series_table',
                 'wide', 'wide_table')

# Bookkeeping attributes that PyTables and pandas write on their own nodes.
# Hidden in logical mode (raw mode shows everything). pandas_type and
# table_type are deliberately absent: those name the stored object and are
# worth seeing.
_INTERNAL_ATTRS = frozenset({
    # PyTables
    'CLASS', 'TITLE', 'VERSION', 'PYTABLES_FORMAT_VERSION', 'FILTERS',
    'EXTDIM', 'NROWS', 'nelements', 'DIRTY', 'blocksize', 'chunksize',
    'optlevel', 'reduction', 'slicesize', 'superblocksize',
    # pandas
    'pandas_version', 'encoding', 'errors', 'nan_rep', 'index_cols',
    'values_cols', 'non_index_axes', 'data_columns', 'info', 'levels',
    'index_variety', 'axis0_variety', 'axis1_variety', 'nblocks', 'ndim',
    'transposed',
})

# Leading bytes of the pickle formats PyTables uses for structured attributes.
# Matched for LABELLING ONLY — these values are never unpickled, since
# pickle.loads on file-supplied bytes would be arbitrary code execution.
_PICKLE_PREFIXES = (b'(lp', b'(dp', b'(I', b'(S', b'ccopy_reg', b'\x80')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Browse the structure of an HDF5 file.')
    parser.add_argument('h5_path', type=str, nargs='?',
                        help='Path to a .h5 / .hdf5 file (a picker opens if omitted)')
    parser.add_argument('--raw', action='store_true',
                        help='Start in raw-HDF5 mode (show pandas internals)')
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
            title='Select an HDF5 file',
            filetypes=[('HDF5 files', '*.h5 *.hdf5 *.he5'), ('All files', '*.*')])
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
            f'{p.name} is a macOS AppleDouble sidecar, not an HDF5 file. '
            f'Try {p.name[2:]} instead.')
    return p


# ---------------------------------------------------------------------------
# File opening
# ---------------------------------------------------------------------------

def _open_h5(path: Path) -> h5py.File:
    """
    Open the file read-only. Locking is disabled process-wide via
    HDF5_USE_FILE_LOCKING at import (see the note there), so this must NOT pass
    a locking= kwarg — a per-open flag that disagrees with PyTables' open would
    make one of the two handles fail.
    """
    return h5py.File(str(path), 'r')


def _open_store(path: Path):
    """
    Open a pandas HDFStore on the same file, or return None if that fails.
    A raw h5py-written file still opens fine as an HDFStore; failure just means
    we lose the logical pandas view, not the whole session.
    """
    try:
        return pd.HDFStore(str(path), mode='r')
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Node classification
# ---------------------------------------------------------------------------

def _pandas_type(obj) -> str | None:
    """
    Return the pandas_type attribute of a group written by pandas, else None.

    This is the same test pandas itself uses to find its objects in a store
    (see HDFStore.groups), so it stays correct without reaching into private
    pandas internals.
    """
    if not isinstance(obj, h5py.Group):
        return None
    try:
        raw = obj.attrs.get('pandas_type')
    except (OSError, KeyError):
        return None
    if raw is None:
        return None
    if isinstance(raw, bytes):
        try:
            raw = raw.decode('utf-8')
        except UnicodeDecodeError:
            return None
    raw = str(raw)
    return raw if raw in _PANDAS_KINDS else None


def _pandas_shape(grp: h5py.Group, kind: str) -> tuple[int | None, int | None]:
    """
    Row and column counts for a pandas-written group, read from metadata only
    (no data is loaded).

    Table format keeps one compound `table` dataset with one row per record.
    Fixed frames keep the column labels in `axis0` and the row index in `axis1`,
    with values split across `block*_values`. Fixed Series are different again:
    just `index` and `values`.
    """
    nrows = ncols = None
    try:
        if kind.endswith('_table'):
            tbl = grp.get('table')
            if isinstance(tbl, h5py.Dataset):
                nrows = int(tbl.shape[0])
                # Each values_block_N field is 2D and stands in for several
                # logical columns; everything else is one column. 'index' is
                # the frame index, not a column.
                ncols = 0
                for name in (tbl.dtype.names or ()):
                    if name == 'index':
                        continue
                    sub = tbl.dtype[name]
                    ncols += int(sub.shape[0]) if sub.shape else 1
        elif kind.startswith('series'):
            vals = grp.get('values')
            if isinstance(vals, h5py.Dataset) and vals.shape:
                nrows = int(vals.shape[0])
        else:
            ax0 = grp.get('axis0')
            ax1 = grp.get('axis1')
            if isinstance(ax0, h5py.Dataset):
                ncols = int(ax0.shape[0])
            if isinstance(ax1, h5py.Dataset):
                nrows = int(ax1.shape[0])
    except Exception:
        pass

    if kind.startswith('series'):
        ncols = 1
    return nrows, ncols


def _pandas_columns_meta(grp: h5py.Group, kind: str) -> list[tuple[str, str]]:
    """
    Column names and dtypes for a pandas group, read from metadata only.

    Used when no preview could be read (very large frames). Table format keeps
    the field types in the compound `table` dtype; fixed format keeps the labels
    in `axis0`, where dtypes aren't recoverable without reading the blocks.
    """
    out: list[tuple[str, str]] = []
    try:
        if kind.endswith('_table'):
            tbl = grp.get('table')
            if not isinstance(tbl, h5py.Dataset):
                return out
            for name in (tbl.dtype.names or ()):
                sub = tbl.dtype[name]
                if sub.shape:
                    # A values_block_N field packs several same-dtype columns
                    # whose names live in a pickled attr; report the group.
                    out.append((f'{name} [{sub.shape[0]} cols]', str(sub.base)))
                else:
                    out.append((name, str(sub)))
        else:
            ax0 = grp.get('axis0')
            if isinstance(ax0, h5py.Dataset):
                for v in ax0[()]:
                    label = v.decode('utf-8', 'replace') if isinstance(v, bytes) \
                        else str(v)
                    out.append((label, '?'))
    except Exception:
        pass
    return out


def classify(obj, logical: bool = True) -> dict:
    """
    Describe one HDF5 node.

    Returns a dict with:
        kind       short type name for the tree's Kind column
        shape      shape / row count as display text
        dtype      dtype as display text
        expandable whether the node has children to show in the current mode
        pandas     the pandas_type string, or None
        nrows      row count for pandas nodes, or None
    """
    info = {'kind': '?', 'shape': '', 'dtype': '', 'expandable': False,
            'pandas': None, 'nrows': None}

    pt = _pandas_type(obj) if logical else None
    if pt is not None:
        nrows, ncols = _pandas_shape(obj, pt)
        info['pandas'] = pt
        info['nrows'] = nrows
        info['kind'] = 'Series' if pt.startswith('series') else 'DataFrame'
        info['kind'] += ' (table)' if pt.endswith('_table') else ' (fixed)'
        rows = '?' if nrows is None else f'{nrows:,}'
        if pt.startswith('series'):
            info['shape'] = f'{rows} rows'
        else:
            cols = '?' if ncols is None else str(ncols)
            info['shape'] = f'{rows} rows x {cols} cols'
        # A pandas object is a leaf in logical mode: its PyTables children are
        # implementation detail.
        return info

    if isinstance(obj, h5py.Dataset):
        info['kind'] = 'Dataset'
        info['shape'] = 'scalar' if obj.shape == () else str(obj.shape)
        dt = obj.dtype
        info['dtype'] = 'compound' if dt.names else str(dt)
        return info

    if isinstance(obj, h5py.Group):
        n = len(obj)
        info['kind'] = 'Group'
        info['shape'] = f'{n} item(s)'
        info['expandable'] = n > 0
        return info

    info['kind'] = type(obj).__name__
    return info


def link_kind(parent: h5py.Group, name: str) -> str | None:
    """
    Return a description for soft / external links, or None for hard links.
    Dereferencing a broken link raises, so links are identified before access.
    """
    try:
        lnk = parent.get(name, getlink=True)
    except Exception:
        return 'link (unreadable)'
    if isinstance(lnk, h5py.SoftLink):
        return f'SoftLink -> {lnk.path}'
    if isinstance(lnk, h5py.ExternalLink):
        return f'ExternalLink -> {lnk.filename}:{lnk.path}'
    return None


def children(grp: h5py.Group) -> list[tuple[str, object, str | None]]:
    """
    List a group's children as (name, object_or_None, link_description).
    object_or_None is None when the child could not be dereferenced (a broken
    link), in which case link_description explains why.
    """
    out = []
    for name in sorted(grp.keys()):
        desc = link_kind(grp, name)
        try:
            obj = grp[name]
        except (KeyError, OSError):
            out.append((name, None, desc or 'broken link'))
            continue
        out.append((name, obj, desc))
    return out


def _fmt_attr(value) -> str:
    """Render an attribute value as bounded, readable text."""
    if isinstance(value, bytes):
        # PyTables stores structured attributes (non_index_axes, info, ...) as
        # pickles. Label them rather than printing the wire format as text; we
        # never unpickle, because that would execute code from the file.
        if value.startswith(_PICKLE_PREFIXES):
            return f'<pickled PyTables value, {len(value)} bytes>'
        try:
            text = value.decode('utf-8')
        except UnicodeDecodeError:
            text = f'<{len(value)} bytes, not text>'
    else:
        text = str(value)
    text = ' '.join(text.split())
    if len(text) > _ATTR_MAXLEN:
        text = text[:_ATTR_MAXLEN] + f'... ({len(text)} chars)'
    return text


def _attr_items(obj, logical: bool = False) -> list[tuple[str, str]]:
    """
    Attribute name/value pairs. In logical mode, PyTables and pandas
    bookkeeping attributes are omitted — they describe the storage layout, not
    the data, and they crowd out the user's own attributes.
    """
    items = []
    try:
        keys = sorted(obj.attrs.keys())
    except Exception:
        return items
    for k in keys:
        # block<N>_items_variety is per-block, so it can't be enumerated in
        # _INTERNAL_ATTRS the way the fixed-name attributes are.
        if logical and (k in _INTERNAL_ATTRS
                        or (k.startswith('block') and k.endswith('_variety'))):
            continue
        try:
            items.append((k, _fmt_attr(obj.attrs[k])))
        except Exception as exc:
            items.append((k, f'<unreadable: {exc.__class__.__name__}>'))
    return items


# ---------------------------------------------------------------------------
# Tabular data — pandas frames and raw compound datasets alike
# ---------------------------------------------------------------------------

def read_page(store, key: str, start: int, stop: int,
              nrows: int | None) -> tuple[pd.DataFrame | None, str]:
    """
    Read rows [start, stop) of a pandas-stored object.

    Returns (frame, message). frame is None when no page could be read, and
    message then explains why.
    """
    if store is None:
        return None, '(no pandas store available)'
    try:
        obj = store.select(key, start=start, stop=stop)
    except (TypeError, NotImplementedError, ValueError):
        # Some layouts (fixed format) don't honour start/stop; slicing a
        # whole-frame read is only acceptable when the frame is known to be
        # small enough to hold in memory in the first place.
        if nrows is not None and nrows > _FULL_READ_CAP:
            return None, f'(preview unavailable - {nrows:,} rows, not read)'
        try:
            obj = store.get(key).iloc[start:stop]
        except Exception as exc:
            return None, f'(preview failed: {exc.__class__.__name__}: {exc})'
    except Exception as exc:
        return None, f'(preview failed: {exc.__class__.__name__}: {exc})'

    if isinstance(obj, pd.Series):
        obj = obj.to_frame(name=obj.name if obj.name is not None else 'values')
    if not isinstance(obj, pd.DataFrame):
        return None, f'(unexpected preview type: {type(obj).__name__})'
    return obj, ''


def _fmt_cell(value) -> str:
    """Render one cell as bounded, single-line text."""
    if isinstance(value, bytes):
        try:
            value = value.decode('utf-8')
        except UnicodeDecodeError:
            return f'<{len(value)} bytes, not text>'
    if value is None:
        return ''
    text = value if isinstance(value, str) else str(value)
    text = ' '.join(text.split())
    if len(text) > _CELL_MAXLEN:
        text = text[:_CELL_MAXLEN] + f'... ({len(text)} chars)'
    return text


def read_dataset_page(dset: h5py.Dataset, start: int,
                      stop: int) -> tuple[list[str], list[list[str]], str]:
    """
    Field names and rows [start, stop) of a compound HDF5 dataset, as display
    strings. Returns (names, rows, message); message is non-empty only when the
    dataset has no named fields, in which case names/rows are both empty.
    """
    names = list(dset.dtype.names or ())
    if not names:
        return [], [], '(not a compound dataset -- no named fields)'
    stop = max(start, min(stop, dset.shape[0]))
    if start >= stop:
        return names, [], ''
    chunk = dset[start:stop]
    rows = [[_fmt_cell(rec[name]) for name in names] for rec in chunk]
    return names, rows, ''


# ---------------------------------------------------------------------------
# Text dump
# ---------------------------------------------------------------------------

def dump(h5: h5py.File, path: Path, logical: bool, out=sys.stdout) -> None:
    """Print the whole structure as an indented text tree."""
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f'{path}  ({size_mb:,.1f} MB)', file=out)
    print(f"mode: {'logical (pandas objects grouped)' if logical else 'raw HDF5'}",
          file=out)

    root_attrs = _attr_items(h5, logical)
    print('/', file=out)
    for k, v in root_attrs:
        print(f'  @{k} = {v}', file=out)

    def walk(grp: h5py.Group, depth: int) -> None:
        pad = '  ' * depth
        kids = children(grp)
        shown = kids[:_DUMP_MAXKIDS]
        for name, obj, link in shown:
            if obj is None:
                print(f'{pad}{name}  [{link}]', file=out)
                continue
            info = classify(obj, logical)
            bits = [info['kind']]
            if info['shape']:
                bits.append(info['shape'])
            if info['dtype']:
                bits.append(info['dtype'])
            suffix = f'  [{link}]' if link else ''
            print(f"{pad}{name}  {'  '.join(bits)}{suffix}", file=out)
            for k, v in _attr_items(obj, logical):
                print(f'{pad}  @{k} = {v}', file=out)
            # Don't recurse through links: an external link would pull in
            # another file's tree, and a soft link would print a subtree twice.
            if info['expandable'] and not link:
                walk(obj, depth + 1)
        if len(kids) > len(shown):
            print(f'{pad}... {len(kids) - len(shown)} more child(ren) not shown '
                  f'(--dump caps at {_DUMP_MAXKIDS} per group)', file=out)

    walk(h5, 1)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class BrowseH5App:

    _MONO      = ('TkFixedFont', 11)
    _HDR_FONT  = ('TkDefaultFont', 10, 'bold')
    _PLACEHOLDER = '__placeholder__'

    def __init__(self, root: tk.Tk, path: Path, h5: h5py.File, store, raw: bool):
        self._root  = root
        self._path  = path
        self._h5    = h5
        self._store = store

        self._logical = tk.BooleanVar(value=not raw)
        # Treeview item id -> HDF5 path. The display text alone can't be trusted
        # to rebuild a path (names may repeat at different depths).
        self._paths: dict[str, str] = {}
        # Describes the currently selected node's tabular data, or None when it
        # has none: {'kind': 'pandas', 'key': str, 'nrows': int|None} for a
        # pandas frame/series, {'kind': 'dataset', 'dset': h5py.Dataset} for a
        # raw compound dataset.
        self._data_source: dict | None = None
        self._page = 0

        root.title(f'browse_h5 - {path.name}')
        root.geometry('1200x800')

        self._build_ui()
        self._populate_root()

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
        tk.Button(top, text='Copy path', command=self._copy_path).pack(
            side=tk.RIGHT, padx=(6, 0))
        tk.Button(top, text='Collapse all', command=self._rebuild).pack(
            side=tk.RIGHT, padx=(6, 0))

        chk = tk.Checkbutton(top, text='Show raw HDF5 nodes',
                             variable=self._logical, onvalue=False, offvalue=True,
                             command=self._rebuild)
        chk.pack(side=tk.RIGHT, padx=(12, 0))
        if self._store is None:
            # Without a store there is no logical view to toggle away from.
            self._logical.set(False)
            chk.config(state=tk.DISABLED)

        self._path_label = tk.Label(top, text='', anchor='w', fg='#444')
        self._path_label.pack(side=tk.LEFT, padx=(16, 0))

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
        self._tree.heading('shape', text='Shape / rows', anchor='w')
        self._tree.heading('dtype', text='dtype', anchor='w')
        self._tree.column('#0',    width=300, minwidth=140, stretch=True)
        self._tree.column('kind',  width=150, minwidth=90,  stretch=False)
        self._tree.column('shape', width=170, minwidth=90,  stretch=False)
        self._tree.column('dtype', width=110, minwidth=70,  stretch=False)

        self._tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self._tree.bind('<<TreeviewOpen>>', self._on_open)
        self._tree.bind('<<TreeviewSelect>>', self._on_select)
        return frame

    def _build_details_pane(self, parent) -> tk.Frame:
        frame = tk.Frame(parent)

        tk.Label(frame, text='Details', font=self._HDR_FONT, anchor='w').pack(
            fill=tk.X)

        text_frame = tk.Frame(frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        tvsb = ttk.Scrollbar(text_frame, orient=tk.VERTICAL)
        self._text = tk.Text(text_frame, wrap=tk.NONE, font=self._MONO,
                             state=tk.DISABLED, height=20,
                             yscrollcommand=tvsb.set)
        tvsb.config(command=self._text.yview)
        self._text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tvsb.pack(side=tk.RIGHT, fill=tk.Y)

        self._data_label = tk.Label(frame, text='', font=self._HDR_FONT,
                                    anchor='w')
        self._data_label.pack(fill=tk.X, pady=(8, 0))

        data_frame = tk.Frame(frame)
        data_frame.pack(fill=tk.BOTH, expand=True)
        dvsb = ttk.Scrollbar(data_frame, orient=tk.VERTICAL)
        dhsb = ttk.Scrollbar(data_frame, orient=tk.HORIZONTAL)
        self._data = ttk.Treeview(data_frame, show='headings', height=8,
                                  yscrollcommand=dvsb.set,
                                  xscrollcommand=dhsb.set)
        dvsb.config(command=self._data.yview)
        dhsb.config(command=self._data.xview)
        self._data.grid(row=0, column=0, sticky='nsew')
        dvsb.grid(row=0, column=1, sticky='ns')
        dhsb.grid(row=1, column=0, sticky='ew')
        data_frame.rowconfigure(0, weight=1)
        data_frame.columnconfigure(0, weight=1)

        # ---- Nav bar (enabled only while the selection has pageable data) ----
        nav = tk.Frame(frame)
        nav.pack(fill=tk.X, pady=(6, 0))

        self._prev_btn = tk.Button(nav, text='<< Prev', command=self._prev_page)
        self._prev_btn.pack(side=tk.LEFT)
        self._next_btn = tk.Button(nav, text='Next >>', command=self._next_page)
        self._next_btn.pack(side=tk.LEFT, padx=(6, 0))
        self._page_label = tk.Label(nav, text='', anchor='w')
        self._page_label.pack(side=tk.LEFT, padx=(12, 0))

        tk.Button(nav, text='Go', command=self._goto_page).pack(side=tk.RIGHT)
        self._goto_entry = tk.Entry(nav, width=12)
        self._goto_entry.pack(side=tk.RIGHT, padx=(6, 6))
        self._goto_entry.bind('<Return>', self._goto_page)
        tk.Label(nav, text='Go to row:').pack(side=tk.RIGHT)

        return frame

    # ------------------------------------------------------------------
    # Tree population
    # ------------------------------------------------------------------

    def _is_logical(self) -> bool:
        return bool(self._logical.get())

    def _populate_root(self):
        root_id = self._tree.insert('', 'end', text='/', open=True,
                                    values=('File', f'{len(self._h5)} item(s)', ''))
        self._paths[root_id] = '/'
        self._insert_children(root_id, self._h5)
        self._tree.selection_set(root_id)

    def _insert_children(self, parent_id: str, grp: h5py.Group):
        """Insert one level of children, with placeholders under expandables."""
        logical = self._is_logical()
        base = self._paths[parent_id].rstrip('/')
        for name, obj, link in children(grp):
            if obj is None:
                item = self._tree.insert(parent_id, 'end', text=name,
                                         values=('broken link', link or '', ''))
                self._paths[item] = f'{base}/{name}'
                continue

            info = classify(obj, logical)
            kind = info['kind']
            if link:
                kind = f'{kind} ({link.split(" ->")[0]})'
            item = self._tree.insert(
                parent_id, 'end', text=name,
                values=(kind, info['shape'], info['dtype']))
            self._paths[item] = f'{base}/{name}'
            if info['expandable']:
                self._tree.insert(item, 'end', text='...',
                                  values=(self._PLACEHOLDER, '', ''))

    def _on_open(self, _event=None):
        item = self._tree.focus()
        if not item:
            return
        kids = self._tree.get_children(item)
        if len(kids) != 1:
            return
        if self._tree.set(kids[0], 'kind') != self._PLACEHOLDER:
            return

        self._tree.delete(kids[0])
        try:
            obj = self._h5[self._paths[item]]
        except (KeyError, OSError) as exc:
            self._tree.insert(item, 'end', text=f'<unreadable: {exc}>',
                              values=('error', '', ''))
            return
        if isinstance(obj, h5py.Group):
            self._insert_children(item, obj)

    def _rebuild(self):
        """Wipe and re-insert the tree — used by the raw/logical toggle."""
        self._tree.delete(*self._tree.get_children())
        self._paths.clear()
        self._clear_details()
        self._populate_root()

    # ------------------------------------------------------------------
    # Details pane
    # ------------------------------------------------------------------

    def _clear_details(self):
        self._text.config(state=tk.NORMAL)
        self._text.delete('1.0', tk.END)
        self._text.config(state=tk.DISABLED)
        self._data_label.config(text='')
        self._data.delete(*self._data.get_children())
        self._data['columns'] = ()
        self._path_label.config(text='')
        self._reset_data_source(None)

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
        key = self._paths.get(item)
        if key is None:
            return

        self._path_label.config(text=key)
        self._data_label.config(text='')
        self._data.delete(*self._data.get_children())
        self._data['columns'] = ()
        self._reset_data_source(None)

        try:
            obj = self._h5[key] if key != '/' else self._h5
        except (KeyError, OSError) as exc:
            self._write([key, '', f'unreadable: {exc.__class__.__name__}: {exc}'])
            return

        logical = self._is_logical()
        info = classify(obj, logical)
        lines = [key, '']

        if key == '/':
            lines.append(f'  {"kind":<12}File root')
            lines.append(f'  {"items":<12}{len(self._h5)}')
        else:
            lines.append(f'  {"kind":<12}{info["kind"]}')
            if info['shape']:
                lines.append(f'  {"shape":<12}{info["shape"]}')
            if info['dtype']:
                lines.append(f'  {"dtype":<12}{info["dtype"]}')

        link = None
        if key != '/':
            parent_path, _, name = key.rpartition('/')
            try:
                parent = self._h5[parent_path or '/']
                link = link_kind(parent, name)
            except (KeyError, OSError):
                link = None
        if link:
            lines.append(f'  {"link":<12}{link}')

        if isinstance(obj, h5py.Dataset):
            lines += self._dataset_lines(obj)
            if obj.dtype.names and obj.shape and obj.shape[0] > 0:
                self._data_source = {'kind': 'dataset', 'dset': obj,
                                     'nrows': int(obj.shape[0])}

        if info['pandas']:
            lines += self._pandas_lines(obj, key, info)

        lines += self._attr_lines(obj, logical)
        self._write(lines)
        self._show_data_page()

    def _dataset_lines(self, dset: h5py.Dataset) -> list[str]:
        lines = []
        if dset.chunks:
            lines.append(f'  {"chunks":<12}{dset.chunks}')
        if dset.compression:
            opts = '' if dset.compression_opts is None \
                else f' (opts={dset.compression_opts})'
            lines.append(f'  {"compression":<12}{dset.compression}{opts}')
        if dset.dtype.names:
            lines.append('')
            lines.append('  --- fields ---')
            for fname in dset.dtype.names:
                sub = dset.dtype[fname]
                shape = f' {sub.shape}' if sub.shape else ''
                lines.append(f'  {fname:<24}{sub.base}{shape}')
        return lines

    def _pandas_lines(self, grp: h5py.Group, key: str, info: dict) -> list[str]:
        """Column list from a first-page read, plus the data itself (via the
        Data pane's own paging, set up here for _show_data_page to use)."""
        frame, msg = read_page(self._store, key, 0, _ROWS_PER_PAGE, info['nrows'])

        lines = ['', '  --- columns ---']
        if frame is not None:
            lines.append(f'  {"(index)":<24}{frame.index.dtype}')
            for col in frame.columns:
                lines.append(f'  {str(col):<24}{frame.dtypes[col]}')
            self._data_source = {'kind': 'pandas', 'key': key,
                                 'nrows': info['nrows']}
        else:
            # No preview — fall back to names/dtypes recoverable from metadata.
            meta_cols = _pandas_columns_meta(grp, info['pandas'])
            if meta_cols:
                lines.append(f'  {msg}')
                for name, dt in meta_cols:
                    lines.append(f'  {name:<24}{dt}')
            else:
                lines.append(f'  {msg}')

        return lines

    # ------------------------------------------------------------------
    # Data pane paging
    # ------------------------------------------------------------------

    def _reset_data_source(self, source: dict | None = None):
        self._data_source = source
        self._page = 0

    def _max_page(self) -> int:
        source = self._data_source
        if not source or not source.get('nrows'):
            return 0
        return max(0, (source['nrows'] - 1) // _ROWS_PER_PAGE)

    def _fetch_page(self, start: int, stop: int) -> tuple[list[str], list[list[str]], str]:
        """(column names, rows, message) for the current data source's page."""
        source = self._data_source
        if source['kind'] == 'dataset':
            return read_dataset_page(source['dset'], start, stop)

        frame, msg = read_page(self._store, source['key'], start, stop,
                               source['nrows'])
        if frame is None:
            return [], [], msg
        names = ['(index)'] + [str(c) for c in frame.columns]
        rows = [[_fmt_cell(idx)] + [_fmt_cell(v) for v in row]
                for idx, row in zip(frame.index, frame.itertuples(index=False))]
        return names, rows, ''

    def _show_data_page(self):
        self._data.delete(*self._data.get_children())
        self._data['columns'] = ()

        if self._data_source is None:
            self._data_label.config(text='')
            self._page_label.config(text='')
            self._prev_btn.config(state=tk.DISABLED)
            self._next_btn.config(state=tk.DISABLED)
            return

        start = self._page * _ROWS_PER_PAGE
        stop = start + _ROWS_PER_PAGE
        names, rows, msg = self._fetch_page(start, stop)

        if msg or not names:
            self._data_label.config(text=f'Data - {msg or "(no data)"}')
            self._page_label.config(text='')
            self._prev_btn.config(state=tk.DISABLED)
            self._next_btn.config(state=tk.DISABLED)
            return

        # Use synthetic column ids: real field/column labels may repeat or
        # contain spaces, either of which breaks Treeview's Tcl column list.
        ids = [f'c{i}' for i in range(len(names))]
        self._data['columns'] = ids
        for cid, label in zip(ids, names):
            self._data.heading(cid, text=label, anchor='w')
            self._data.column(cid, width=110, minwidth=60, stretch=False,
                              anchor='w')
        for row in rows:
            self._data.insert('', 'end', values=row)

        total = 'unknown' if self._data_source['nrows'] is None \
            else f'{self._data_source["nrows"]:,}'
        if rows:
            self._data_label.config(
                text=f'Data - rows {start:,} - {start + len(rows) - 1:,} of {total}')
        else:
            self._data_label.config(text=f'Data - none (past end of {total} rows)')

        max_page = self._max_page()
        self._page_label.config(text=f'Page {self._page + 1} / {max_page + 1}')
        self._prev_btn.config(state=tk.NORMAL if self._page > 0 else tk.DISABLED)
        self._next_btn.config(
            state=tk.NORMAL if self._page < max_page else tk.DISABLED)

    def _prev_page(self):
        if self._data_source and self._page > 0:
            self._page -= 1
            self._show_data_page()

    def _next_page(self):
        if self._data_source and self._page < self._max_page():
            self._page += 1
            self._show_data_page()

    def _goto_page(self, _event=None):
        """Jump to the page holding a given absolute row number."""
        raw = self._goto_entry.get().strip().replace(',', '')
        self._goto_entry.delete(0, tk.END)
        if self._data_source is None:
            return
        try:
            row = int(raw)
        except ValueError:
            return
        nrows = self._data_source.get('nrows')
        if nrows is not None and not (0 <= row < nrows):
            return
        self._page = row // _ROWS_PER_PAGE
        self._show_data_page()

    def _attr_lines(self, obj, logical: bool = False) -> list[str]:
        items = _attr_items(obj, logical)
        if not items:
            return []
        lines = ['', '  --- attributes ---']
        for k, v in items:
            lines.append(f'  {k:<24}{v}')
        return lines

    # ------------------------------------------------------------------
    # Buttons
    # ------------------------------------------------------------------

    def _copy_path(self):
        sel = self._tree.selection()
        if not sel:
            return
        key = self._paths.get(sel[0], '')
        self._root.clipboard_clear()
        self._root.clipboard_append(key)

    def _open_other(self):
        chosen = filedialog.askopenfilename(
            title='Select an HDF5 file',
            filetypes=[('HDF5 files', '*.h5 *.hdf5 *.he5'), ('All files', '*.*')])
        if not chosen:
            return
        try:
            path = _resolve_path(chosen, allow_picker=False)
            h5 = _open_h5(path)
        except Exception as exc:
            messagebox.showerror('Cannot open file', str(exc))
            return

        store = _open_store(path)
        self._close_handles()

        self._path, self._h5, self._store = path, h5, store
        self._root.title(f'browse_h5 - {path.name}')
        # Rebuild the whole UI so the header, toggle state and panes all match
        # the new file.
        for child in self._root.winfo_children():
            child.destroy()
        self._paths.clear()
        self._logical = tk.BooleanVar(value=True)
        self._build_ui()
        self._populate_root()

    def _close_handles(self):
        for handle in (self._store, self._h5):
            try:
                if handle is not None:
                    handle.close()
            except Exception:
                pass

    def close(self):
        self._close_handles()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()

    try:
        path = _resolve_path(args.h5_path, allow_picker=not args.dump)
    except Exception as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)

    try:
        h5 = _open_h5(path)
    except OSError as exc:
        # 'file signature not found' means it isn't HDF5 at all — a wrong file,
        # or a cloud-storage placeholder that hasn't been downloaded.
        print(f'Error: cannot read {path} as HDF5.\n  {exc}', file=sys.stderr)
        sys.exit(1)

    store = _open_store(path)

    if args.dump:
        try:
            dump(h5, path, logical=(not args.raw and store is not None))
        finally:
            if store is not None:
                store.close()
            h5.close()
        return

    root = tk.Tk()
    app = BrowseH5App(root, path, h5, store, raw=args.raw)
    try:
        root.mainloop()
    finally:
        app.close()


if __name__ == '__main__':
    main()
