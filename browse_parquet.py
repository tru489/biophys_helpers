"""
browse_parquet.py

Tabular browser for Parquet (.parquet / .pq) files. Shows the file-level
metadata, the full column schema with per-column datatypes, and a scrollable,
paged view of the rows themselves.

Built directly on pyarrow.parquet rather than pandas.read_parquet, for the same
reason browse_h5.py reads HDF5 metadata before data: schema, row counts and row
group layout are all available without touching a single value, and the row view
then reads only the row groups it needs. A 10-million-row file opens as fast as a
10-row one.

Usage:
    python browse_parquet.py [<parquet_path>] [--rows N] [--dump]

    <parquet_path>  Path to a .parquet / .pq file. If omitted, a file picker opens.
    --rows N        Rows shown per page in the data view (default 100).
    --dump          Print schema and the first rows to stdout and exit; no window
                    is opened. Requires <parquet_path>.
"""
import argparse
import sys
from pathlib import Path

import pyarrow.parquet as pq
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from fsutil import is_appledouble


_ROWS_PER_PAGE = 100      # rows per page in the data view
_CELL_MAXLEN   = 120      # truncate long cell reprs to this many chars
_DUMP_ROWS     = 10       # rows printed in --dump mode
_KV_MAXLEN     = 200      # truncate key-value metadata reprs
_BATCH_ROWS    = 2_048    # rows decoded at a time when reading a page

# Written by pandas into the parquet key-value metadata. It is a large JSON blob
# describing the original pandas dtypes; the schema pane already shows the arrow
# types, so it is listed by size rather than printed in full.
_BULKY_KV_KEYS = frozenset({b'pandas', b'ARROW:schema'})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Browse the schema and rows of a Parquet file.')
    parser.add_argument('parquet_path', type=str, nargs='?',
                        help='Path to a .parquet / .pq file (a picker opens if omitted)')
    parser.add_argument('--rows', type=int, default=None,
                        help=f'Rows per page in the data view '
                             f'(default {_ROWS_PER_PAGE}, or {_DUMP_ROWS} with --dump)')
    parser.add_argument('--dump', action='store_true',
                        help='Print schema and the first rows to stdout and exit')
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
            title='Select a Parquet file',
            filetypes=[('Parquet files', '*.parquet *.pq'), ('All files', '*.*')])
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
            f'{p.name} is a macOS AppleDouble sidecar, not a Parquet file. '
            f'Try {p.name[2:]} instead.')
    return p


# ---------------------------------------------------------------------------
# File opening
# ---------------------------------------------------------------------------

def _open_parquet(path: Path) -> pq.ParquetFile:
    """
    Open the file for metadata-only access. Nothing but the footer is read here,
    so this stays fast regardless of file size.
    """
    return pq.ParquetFile(str(path))


# ---------------------------------------------------------------------------
# Schema and metadata
# ---------------------------------------------------------------------------

def _schema_rows(pf: pq.ParquetFile) -> list[tuple[str, str, str, str]]:
    """
    One (name, arrow type, parquet physical type, nullable) row per column.

    The arrow type is the logical one you get back when reading (timestamp,
    large_string, list<...>); the physical type is how parquet stored it on disk
    (INT64, BYTE_ARRAY). They differ often enough to be worth showing both.
    """
    out: list[tuple[str, str, str, str]] = []
    try:
        arrow_schema = pf.schema_arrow
    except Exception:
        arrow_schema = None
    try:
        parquet_schema = pf.schema
    except Exception:
        parquet_schema = None

    n = len(arrow_schema) if arrow_schema is not None else 0
    for i in range(n):
        field = arrow_schema.field(i)
        physical = ''
        try:
            # The parquet schema is flat over leaf columns, so its indices only
            # line up with the arrow fields when no column is nested.
            col = parquet_schema.column(i) if parquet_schema is not None else None
            if col is not None and col.name == field.name:
                logical = str(col.logical_type)
                physical = col.physical_type
                if logical and logical != 'None':
                    physical = f'{physical} / {logical}'
        except Exception:
            pass
        out.append((field.name, str(field.type), physical,
                    'yes' if field.nullable else 'no'))
    return out


def _kv_metadata(pf: pq.ParquetFile) -> list[tuple[str, str]]:
    """Key-value file metadata, with the bulky pandas blob summarised by size."""
    out: list[tuple[str, str]] = []
    try:
        kv = pf.metadata.metadata
    except Exception:
        return out
    if not kv:
        return out
    for raw_key in sorted(kv):
        try:
            key = raw_key.decode('utf-8', 'replace') if isinstance(raw_key, bytes) \
                else str(raw_key)
            value = kv[raw_key]
            if raw_key in _BULKY_KV_KEYS:
                out.append((key, f'<{len(value):,} bytes, not shown>'))
                continue
            text = value.decode('utf-8', 'replace') if isinstance(value, bytes) \
                else str(value)
            text = ' '.join(text.split())
            if len(text) > _KV_MAXLEN:
                text = text[:_KV_MAXLEN] + f'... ({len(text)} chars)'
            out.append((key, text))
        except Exception as exc:
            out.append((str(raw_key), f'<unreadable: {exc.__class__.__name__}>'))
    return out


def _file_lines(pf: pq.ParquetFile, path: Path) -> list[str]:
    """The file-level metadata block, as display lines."""
    size_mb = path.stat().st_size / (1024 * 1024)
    lines = [str(path), '']
    lines.append(f'  {"file size":<16}{size_mb:,.1f} MB')

    try:
        md = pf.metadata
    except Exception as exc:
        lines.append(f'  metadata unreadable: {exc.__class__.__name__}: {exc}')
        return lines

    try:
        lines.append(f'  {"rows":<16}{md.num_rows:,}')
        lines.append(f'  {"columns":<16}{md.num_columns}')
        lines.append(f'  {"row groups":<16}{md.num_row_groups}')
        lines.append(f'  {"format version":<16}{md.format_version}')
        lines.append(f'  {"created by":<16}{md.created_by}')
    except Exception:
        pass

    # Compressed/uncompressed totals and the codecs actually used have to be
    # summed over every column chunk; the footer holds no file-wide total.
    try:
        compressed = uncompressed = 0
        codecs = set()
        for g in range(md.num_row_groups):
            rg = md.row_group(g)
            compressed += rg.total_byte_size
            for c in range(rg.num_columns):
                chunk = rg.column(c)
                uncompressed += chunk.total_uncompressed_size
                codecs.add(str(chunk.compression))
        lines.append(f'  {"compression":<16}{", ".join(sorted(codecs)) or "(none)"}')
        lines.append(f'  {"stored bytes":<16}{compressed:,}')
        if uncompressed:
            ratio = uncompressed / compressed if compressed else 0
            lines.append(f'  {"uncompressed":<16}{uncompressed:,} '
                         f'({ratio:,.1f}x)')
    except Exception:
        pass

    kv = _kv_metadata(pf)
    lines.append('')
    lines.append('  --- key-value metadata ---')
    if kv:
        for key, value in kv:
            lines.append(f'  {key:<24}{value}')
    else:
        # polars writes none at all, so this is normal rather than a problem.
        lines.append('  (none)')

    lines.append('')
    lines.append('  --- row groups ---')
    try:
        for g in range(md.num_row_groups):
            rg = md.row_group(g)
            mb = rg.total_byte_size / (1024 * 1024)
            lines.append(f'  {("group " + str(g)):<24}{rg.num_rows:,} rows, '
                         f'{mb:,.1f} MB')
    except Exception:
        lines.append('  (unreadable)')
    return lines


# ---------------------------------------------------------------------------
# Row reading
# ---------------------------------------------------------------------------

def _row_group_bounds(pf: pq.ParquetFile) -> list[tuple[int, int, int]]:
    """
    (group index, first row, row count) for each row group.

    Parquet stores row counts per group but no cumulative offsets, so the
    running total has to be built here to map an absolute row number onto a
    group.
    """
    bounds = []
    start = 0
    try:
        md = pf.metadata
        for g in range(md.num_row_groups):
            n = md.row_group(g).num_rows
            bounds.append((g, start, n))
            start += n
    except Exception:
        pass
    return bounds


def _read_rows(pf: pq.ParquetFile, start: int,
               count: int) -> tuple[list[str], list[list[str]], str]:
    """
    Read rows [start, start + count) as display strings.

    Only the row groups spanning that window are read, so memory stays flat no
    matter how large the file is. Returns (column names, rows, message); message
    is non-empty only when the read failed, in which case the rows are empty.
    """
    try:
        names = [f.name for f in pf.schema_arrow]
    except Exception as exc:
        return [], [], f'(schema unreadable: {exc.__class__.__name__}: {exc})'

    if count <= 0:
        return names, [], ''

    stop = start + count
    spanning = [(g, first) for g, first, n in _row_group_bounds(pf)
                if first < stop and first + n > start]
    if not spanning:
        return names, [], ''
    wanted = [g for g, _ in spanning]
    origin = spanning[0][1]

    # Stream the spanning groups in batches rather than reading them whole: a
    # file written as one giant row group (pyarrow's default is 1M rows) would
    # otherwise pull hundreds of MB into memory to show a hundred rows. Batches
    # are decoded one at a time and dropped, so peak memory tracks the batch
    # size, not the group size.
    skip = max(0, start - origin)
    collected: list = []
    try:
        for batch in pf.iter_batches(batch_size=_BATCH_ROWS, row_groups=wanted):
            n = batch.num_rows
            if skip >= n:
                skip -= n
                continue
            take = batch.slice(skip, count - len(collected))
            skip = 0
            collected.append(take)
            if sum(b.num_rows for b in collected) >= count:
                break
    except Exception as exc:
        return names, [], f'(read failed: {exc.__class__.__name__}: {exc})'

    if not collected:
        return names, [], ''

    try:
        columns = [[] for _ in names]
        for batch in collected:
            for i, col in enumerate(batch.columns):
                columns[i].extend(col.to_pylist())
    except Exception as exc:
        return names, [], f'(read failed: {exc.__class__.__name__}: {exc})'

    nrows = len(columns[0]) if columns else 0
    rows = [[_fmt_cell(col[i]) for col in columns] for i in range(nrows)]
    return names, rows, ''


def _fmt_cell(value) -> str:
    """
    Render one cell as bounded, single-line text.

    Values are kept as their plain repr rather than prettied up, so anything
    copied out of the table pastes back into code unchanged.
    """
    if value is None:
        return ''
    text = value if isinstance(value, str) else str(value)
    text = ' '.join(text.split())
    if len(text) > _CELL_MAXLEN:
        text = text[:_CELL_MAXLEN] + f'... ({len(text)} chars)'
    return text


# ---------------------------------------------------------------------------
# Text dump
# ---------------------------------------------------------------------------

def dump(pf: pq.ParquetFile, path: Path, rows: int, out=sys.stdout) -> None:
    """Print the file metadata, the schema and the first rows."""
    for line in _file_lines(pf, path):
        print(line, file=out)

    print('', file=out)
    print('  --- schema ---', file=out)
    schema = _schema_rows(pf)
    for name, arrow_type, physical, nullable in schema:
        null = '' if nullable == 'yes' else '  not null'
        phys = f'  [{physical}]' if physical else ''
        print(f'  {name:<28}{arrow_type}{phys}{null}', file=out)

    n = max(0, rows)
    print('', file=out)
    print(f'  --- first {n} row(s) ---', file=out)
    names, data, msg = _read_rows(pf, 0, n)
    if msg:
        print(f'  {msg}', file=out)
        return
    if not data:
        print('  (no rows)', file=out)
        return

    # Size each column to its widest visible value so the dump lines up in a
    # terminal; values are already length-capped by _fmt_cell.
    widths = [len(name) for name in names]
    for row in data:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    header = '  '.join(name.ljust(widths[i]) for i, name in enumerate(names))
    print(f'  {header}', file=out)
    print(f'  {"-" * len(header)}', file=out)
    for j, row in enumerate(data):
        body = '  '.join(cell.ljust(widths[i]) for i, cell in enumerate(row))
        print(f'  {body}', file=out)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class BrowseParquetApp:

    _MONO      = ('TkFixedFont', 11)
    _HDR_FONT  = ('TkDefaultFont', 10, 'bold')

    def __init__(self, root: tk.Tk, path: Path, pf: pq.ParquetFile,
                 rows_per_page: int):
        self._root = root
        self._path = path
        self._pf   = pf
        self._rows = max(1, rows_per_page)
        self._page = 0

        root.title(f'browse_parquet - {path.name}')
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
        tk.Button(top, text='Copy row', command=self._copy_row).pack(
            side=tk.RIGHT, padx=(6, 0))
        tk.Button(top, text='Copy column', command=self._copy_column).pack(
            side=tk.RIGHT, padx=(6, 0))

        self._info_label = tk.Label(top, text='', anchor='w', fg='#444')
        self._info_label.pack(side=tk.LEFT, padx=(16, 0))

        # ---- Split: metadata+schema | rows ----
        split = ttk.PanedWindow(self._root, orient=tk.VERTICAL)
        split.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))

        split.add(self._build_schema_pane(split), weight=2)
        split.add(self._build_rows_pane(split), weight=3)

    def _build_schema_pane(self, parent) -> tk.Frame:
        """File metadata on the left, the column schema table on the right."""
        frame = tk.Frame(parent)

        inner = ttk.PanedWindow(frame, orient=tk.HORIZONTAL)
        inner.pack(fill=tk.BOTH, expand=True)

        # ---- File metadata ----
        meta = tk.Frame(inner)
        tk.Label(meta, text='File', font=self._HDR_FONT, anchor='w').pack(fill=tk.X)
        text_frame = tk.Frame(meta)
        text_frame.pack(fill=tk.BOTH, expand=True)
        tvsb = ttk.Scrollbar(text_frame, orient=tk.VERTICAL)
        self._text = tk.Text(text_frame, wrap=tk.NONE, font=self._MONO,
                             state=tk.DISABLED, height=14,
                             yscrollcommand=tvsb.set)
        tvsb.config(command=self._text.yview)
        self._text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tvsb.pack(side=tk.RIGHT, fill=tk.Y)

        # ---- Schema table ----
        sch = tk.Frame(inner)
        self._schema_label = tk.Label(sch, text='Columns', font=self._HDR_FONT,
                                      anchor='w')
        self._schema_label.pack(fill=tk.X)

        grid = tk.Frame(sch)
        grid.pack(fill=tk.BOTH, expand=True)
        svsb = ttk.Scrollbar(grid, orient=tk.VERTICAL)
        shsb = ttk.Scrollbar(grid, orient=tk.HORIZONTAL)
        self._schema = ttk.Treeview(
            grid, columns=('name', 'type', 'physical', 'null'),
            show='headings', selectmode='browse',
            yscrollcommand=svsb.set, xscrollcommand=shsb.set)
        svsb.config(command=self._schema.yview)
        shsb.config(command=self._schema.xview)

        self._schema.heading('name',     text='Column',        anchor='w')
        self._schema.heading('type',     text='Type (arrow)',  anchor='w')
        self._schema.heading('physical', text='Stored as',     anchor='w')
        self._schema.heading('null',     text='Null?',         anchor='w')
        self._schema.column('name',     width=200, minwidth=100, stretch=True)
        self._schema.column('type',     width=150, minwidth=80,  stretch=False)
        self._schema.column('physical', width=180, minwidth=80,  stretch=False)
        self._schema.column('null',     width=60,  minwidth=50,  stretch=False)

        self._schema.grid(row=0, column=0, sticky='nsew')
        svsb.grid(row=0, column=1, sticky='ns')
        shsb.grid(row=1, column=0, sticky='ew')
        grid.rowconfigure(0, weight=1)
        grid.columnconfigure(0, weight=1)

        inner.add(meta, weight=2)
        inner.add(sch, weight=3)
        return frame

    def _build_rows_pane(self, parent) -> tk.Frame:
        frame = tk.Frame(parent)

        self._rows_label = tk.Label(frame, text='Rows', font=self._HDR_FONT,
                                    anchor='w')
        self._rows_label.pack(fill=tk.X, pady=(8, 0))

        grid = tk.Frame(frame)
        grid.pack(fill=tk.BOTH, expand=True)
        rvsb = ttk.Scrollbar(grid, orient=tk.VERTICAL)
        rhsb = ttk.Scrollbar(grid, orient=tk.HORIZONTAL)
        self._data = ttk.Treeview(grid, show='headings', selectmode='browse',
                                  yscrollcommand=rvsb.set,
                                  xscrollcommand=rhsb.set)
        rvsb.config(command=self._data.yview)
        rhsb.config(command=self._data.xview)
        self._data.grid(row=0, column=0, sticky='nsew')
        rvsb.grid(row=0, column=1, sticky='ns')
        rhsb.grid(row=1, column=0, sticky='ew')
        grid.rowconfigure(0, weight=1)
        grid.columnconfigure(0, weight=1)

        # ---- Nav bar ----
        nav = tk.Frame(frame)
        nav.pack(fill=tk.X, pady=(6, 0))

        self._prev_btn = tk.Button(nav, text='<< Prev', command=self._prev)
        self._prev_btn.pack(side=tk.LEFT)
        self._next_btn = tk.Button(nav, text='Next >>', command=self._next)
        self._next_btn.pack(side=tk.LEFT, padx=(6, 0))
        self._page_label = tk.Label(nav, text='', anchor='w')
        self._page_label.pack(side=tk.LEFT, padx=(12, 0))

        tk.Button(nav, text='Go', command=self._goto).pack(side=tk.RIGHT)
        self._goto_entry = tk.Entry(nav, width=12)
        self._goto_entry.pack(side=tk.RIGHT, padx=(6, 6))
        self._goto_entry.bind('<Return>', self._goto)
        tk.Label(nav, text='Go to row:').pack(side=tk.RIGHT)

        return frame

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    def _populate(self):
        self._write(_file_lines(self._pf, self._path))

        schema = _schema_rows(self._pf)
        self._schema.delete(*self._schema.get_children())
        for name, arrow_type, physical, nullable in schema:
            self._schema.insert('', 'end',
                                values=(name, arrow_type, physical, nullable))
        self._schema_label.config(text=f'Columns  ({len(schema)})')

        self._info_label.config(text=f'{self._num_rows():,} rows x '
                                     f'{len(schema)} columns')
        self._show_page()

    def _write(self, lines: list[str]):
        self._text.config(state=tk.NORMAL)
        self._text.delete('1.0', tk.END)
        self._text.insert('1.0', '\n'.join(lines))
        self._text.config(state=tk.DISABLED)

    def _num_rows(self) -> int:
        try:
            return self._pf.metadata.num_rows
        except Exception:
            return 0

    def _max_page(self) -> int:
        total = self._num_rows()
        if total <= 0:
            return 0
        return (total - 1) // self._rows

    def _show_page(self):
        start = self._page * self._rows
        names, data, msg = _read_rows(self._pf, start, self._rows)

        # Use synthetic column ids: real column labels may repeat or contain
        # spaces, either of which breaks Treeview's Tcl column list.
        labels = ['(row)'] + names
        ids = [f'c{i}' for i in range(len(labels))]
        self._data.delete(*self._data.get_children())
        self._data['columns'] = ids
        for cid, label in zip(ids, labels):
            self._data.heading(cid, text=label, anchor='w')
            width = 70 if cid == 'c0' else 120
            self._data.column(cid, width=width, minwidth=50, stretch=False,
                              anchor='w')
        for i, row in enumerate(data):
            self._data.insert('', 'end', values=[f'{start + i:,}'] + row)

        total = self._num_rows()
        if msg:
            self._rows_label.config(text=f'Rows - {msg}')
        elif data:
            self._rows_label.config(
                text=f'Rows {start:,} - {start + len(data) - 1:,} of {total:,}')
        else:
            self._rows_label.config(text=f'Rows - none ({total:,} in file)')

        self._page_label.config(text=f'Page {self._page + 1} / {self._max_page() + 1}')
        self._prev_btn.config(state=tk.NORMAL if self._page > 0 else tk.DISABLED)
        self._next_btn.config(
            state=tk.NORMAL if self._page < self._max_page() else tk.DISABLED)

    # ------------------------------------------------------------------
    # Buttons
    # ------------------------------------------------------------------

    def _prev(self):
        if self._page > 0:
            self._page -= 1
            self._show_page()

    def _next(self):
        if self._page < self._max_page():
            self._page += 1
            self._show_page()

    def _goto(self, _event=None):
        """Jump to the page holding a given absolute row number."""
        raw = self._goto_entry.get().strip().replace(',', '')
        self._goto_entry.delete(0, tk.END)
        try:
            row = int(raw)
        except ValueError:
            return
        if not 0 <= row < self._num_rows():
            return
        self._page = row // self._rows
        self._show_page()

    def _copy_column(self):
        """Put the selected schema row's column name on the clipboard."""
        sel = self._schema.selection()
        if not sel:
            return
        name = self._schema.set(sel[0], 'name')
        self._root.clipboard_clear()
        self._root.clipboard_append(name)

    def _copy_row(self):
        """Put the selected data row on the clipboard, tab-separated."""
        sel = self._data.selection()
        if not sel:
            return
        values = self._data.item(sel[0], 'values')
        # Drop the leading synthetic row number: it isn't part of the data.
        self._root.clipboard_clear()
        self._root.clipboard_append('\t'.join(str(v) for v in values[1:]))

    def _open_other(self):
        chosen = filedialog.askopenfilename(
            title='Select a Parquet file',
            filetypes=[('Parquet files', '*.parquet *.pq'), ('All files', '*.*')])
        if not chosen:
            return
        try:
            path = _resolve_path(chosen, allow_picker=False)
            pf = _open_parquet(path)
        except Exception as exc:
            messagebox.showerror('Cannot open file', str(exc))
            return

        self._close_handles()
        self._path, self._pf = path, pf
        self._page = 0
        self._root.title(f'browse_parquet - {path.name}')
        # Rebuild the whole UI so the header and both panes match the new file.
        for child in self._root.winfo_children():
            child.destroy()
        self._build_ui()
        self._populate()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _close_handles(self):
        try:
            if self._pf is not None:
                self._pf.close()
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
        path = _resolve_path(args.parquet_path, allow_picker=not args.dump)
    except Exception as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)

    try:
        pf = _open_parquet(path)
    except Exception as exc:
        # A bad magic number means it isn't parquet at all — a wrong file, or a
        # cloud-storage placeholder that hasn't been downloaded.
        print(f'Error: cannot read {path} as Parquet.\n  {exc}', file=sys.stderr)
        sys.exit(1)

    if args.dump:
        # A dump goes to a terminal, where 100 rows is unhelpfully long; the GUI
        # paging default doesn't apply.
        try:
            dump(pf, path, args.rows if args.rows is not None else _DUMP_ROWS)
        finally:
            pf.close()
        return

    root = tk.Tk()
    app = BrowseParquetApp(root, path, pf,
                           args.rows if args.rows is not None else _ROWS_PER_PAGE)
    try:
        root.mainloop()
    finally:
        app.close()


if __name__ == '__main__':
    main()
