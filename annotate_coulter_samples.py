"""
annotate_coulter_samples.py

Interactive GUI for annotating Coulter Counter samples with arbitrary custom
metadata columns. Each sample is one column of a summary Coulter CSV (the same
format consumed by pair_bm_runs.py --coulter: each column is a sample name, each
row a single-cell volume measurement).

This is a stripped-down sibling of pair_bm_runs.py: it keeps the spreadsheet-like
table and the add/remove column dialogs, but has no run types, no Coulter
dropdown, and no grouping. It is purely for per-sample annotation.

Workflow:
    1. Load a summary Coulter CSV. One table row is created per column (sample).
    2. Add custom metadata columns (free-text or yes/no checkbox) via the
       "Add Column" dialog, and remove them via "Remove Column".
    3. Click any custom cell to edit it in place (checkbox cells toggle on click).
    4. Click "Done" to write output.

Output (written to <csv_dir>/YYYYMMDD_HHMMSS_coulter_sample_annotation/):
    metadata.csv   one row per sample; sample_name + all annotation columns.
    data.h5        pandas HDFStore, standalone with all relevant information:
                   /metadata           DataFrame (sample_name, h5_key, + annotations)
                   /data/{h5_key}      per-sample single-cell volume DataFrame

Usage:
    python annotate_coulter_samples.py <coulter_csv>
"""
import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import warnings
warnings.filterwarnings('ignore', message='object name is not a valid Python identifier')

import pandas as pd
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_cli_args() -> Path:
    """Returns the path to the summary Coulter CSV."""
    parser = argparse.ArgumentParser(
        description="Annotate Coulter Counter samples with custom metadata columns."
    )
    parser.add_argument('coulter_csv', type=str,
                        help='Path to a summary Coulter CSV whose columns are '
                             'sample names and rows are volume measurements')
    args = parser.parse_args()

    coulter_csv = Path(args.coulter_csv)
    if not coulter_csv.is_file():
        raise FileNotFoundError(f"Coulter file not found: {coulter_csv}")

    return coulter_csv


def _check_dependencies():
    """
    Verify runtime dependencies that otherwise fail silently or late, before
    the GUI opens. PyTables is the key one: pandas needs it to write data.h5,
    but it is an *optional* pandas backend, so a missing install only surfaces
    as an ImportError at write time — long after the annotation work is done.
    (Note: h5py is NOT a substitute for pytables here.)
    """
    missing = []
    try:
        import tables  # noqa: F401
    except ImportError:
        missing.append(
            "pytables (Python module 'tables') — required to write data.h5.\n"
            "    Install with:  conda install -n biophys_helpers pytables\n"
            "             (or:  pip install tables)"
        )

    if missing:
        print("ERROR: missing required dependencies:\n  - "
              + "\n  - ".join(missing), file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_coulter(path: Path) -> pd.DataFrame:
    """
    Loads a summary Coulter CSV where each column is a sample and each row is a
    single-cell volume measurement. Returns the full DataFrame.
    """
    return pd.read_csv(path)


def _safe_key(name: str, used: set) -> str:
    """
    Convert an arbitrary sample name into a unique, HDF5-safe path component.
    Non-alphanumeric characters become underscores; collisions get a numeric
    suffix. `used` is mutated to record the returned key.
    """
    base = re.sub(r'[^0-9A-Za-z]+', '_', name).strip('_') or 'sample'
    key = base
    i = 1
    while key in used:
        i += 1
        key = f'{base}_{i}'
    used.add(key)
    return key


# ---------------------------------------------------------------------------
# Main GUI window
# ---------------------------------------------------------------------------

class AnnotationWindow:
    """
    Spreadsheet-like tkinter window for annotating Coulter samples with custom
    metadata columns.

    Fixed column: [sample]. Plus any user-added custom columns.
    Clicking a custom column cell opens an in-place Entry; checkbox columns
    toggle on click.
    """

    def __init__(self, root: tk.Tk, coulter_csv: Path, coulter_df: pd.DataFrame):
        self._root = root
        self._coulter_csv = coulter_csv
        self._coulter_df = coulter_df
        self._samples = list(coulter_df.columns)

        self._custom_cols: list = []
        self._checkbox_cols: set = set()
        self._active_editor = None
        self._vlines: list = []
        self._last_col = None

        self._row_data: dict = {name: {} for name in self._samples}

        root.title(f"annotate_coulter_samples — {coulter_csv.name}")
        self._build_ui()
        self._populate_table()

    # ------------------------------------------------------------------
    # Column lists
    # ------------------------------------------------------------------

    def _all_cols(self) -> list:
        return ['sample'] + self._custom_cols

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        n = len(self._samples)
        tk.Label(
            self._root,
            text=(f"{n} sample(s) loaded from {self._coulter_csv.name}"),
            font=('TkDefaultFont', 10, 'bold'), anchor='w',
        ).pack(fill=tk.X, padx=10, pady=(8, 2))

        # 'clam' renders Treeview row-tag backgrounds reliably (the macOS aqua
        # theme ignores them), which is needed for the zebra striping below.
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except tk.TclError:
            pass

        tree_frame = tk.Frame(self._root)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 4))

        vsb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL)
        self._hsb = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)

        self._tree = ttk.Treeview(
            tree_frame, selectmode='extended',
            yscrollcommand=vsb.set, xscrollcommand=self._on_xscroll,
            show='headings', height=24,
        )
        vsb.config(command=self._tree.yview)
        self._hsb.config(command=self._tree.xview)

        self._tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        self._hsb.grid(row=1, column=0, sticky='ew')
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

        # Alternating row colors for visual separation between rows.
        self._tree.tag_configure('evenrow', background='#ffffff')
        self._tree.tag_configure('oddrow', background='#e8eef4')

        self._tree.bind('<ButtonRelease-1>', self._on_cell_click)
        # Keep the vertical column separators in sync with resizes and column
        # drags (drags settle on button release).
        self._tree.bind('<ButtonRelease-1>',
                        lambda e: self._draw_gridlines(), add='+')
        self._tree.bind('<Configure>',
                        lambda e: self._draw_gridlines(), add='+')

        self._setup_columns()

        btn_frame = tk.Frame(self._root)
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 8))

        tk.Button(btn_frame, text="Add Column",
                  command=self._add_column).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(btn_frame, text="Remove Column",
                  command=self._remove_column).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(btn_frame, text="Set Cells…",
                  command=self._set_cells).pack(side=tk.LEFT, padx=(0, 12))
        tk.Button(btn_frame, text="↑",
                  command=self._move_up).pack(side=tk.LEFT, padx=(0, 2))
        tk.Button(btn_frame, text="↓",
                  command=self._move_down).pack(side=tk.LEFT, padx=(0, 12))

        tk.Button(btn_frame, text="Done",
                  command=self._finish).pack(side=tk.RIGHT)

    def _setup_columns(self):
        cols = self._all_cols()
        self._tree['columns'] = cols
        for col in cols:
            w = 240 if col == 'sample' else 160
            if col in self._checkbox_cols:
                label = f'☑ {col}'
            else:
                label = col.replace('_', ' ').title()
            self._tree.heading(col, text=label)
            self._tree.column(col, width=w, minwidth=60, anchor='w',
                              stretch=tk.YES if col == 'sample' else tk.NO)

    def _row_values(self, sample: str) -> list:
        rd = self._row_data[sample]
        custom = []
        for c in self._custom_cols:
            if c in self._checkbox_cols:
                custom.append('✓' if rd.get(c) == 'yes' else '')
            else:
                custom.append(rd.get(c, ''))
        return [sample] + custom

    def _populate_table(self):
        for item in self._tree.get_children():
            self._tree.delete(item)
        for sample_name in self._samples:
            self._tree.insert('', tk.END, iid=sample_name,
                              values=self._row_values(sample_name))
        self._restripe()
        self._tree.after_idle(self._draw_gridlines)

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _restripe(self):
        """Reapply alternating row tags by current visual position."""
        for i, item in enumerate(self._tree.get_children()):
            self._tree.item(item, tags=('evenrow' if i % 2 == 0 else 'oddrow',))

    def _on_xscroll(self, *args):
        self._hsb.set(*args)
        self._draw_gridlines()

    def _draw_gridlines(self):
        """
        Overlay thin vertical separators at each visible column boundary.
        Uses bbox() so positions track horizontal scrolling and column widths.
        """
        for ln in self._vlines:
            ln.destroy()
        self._vlines = []

        children = self._tree.get_children()
        if not children:
            return
        first = children[0]
        cols = self._tree['columns']
        for col in cols[:-1]:
            bbox = self._tree.bbox(first, col)
            if not bbox:
                continue
            bx, _, bw, _ = bbox
            line = tk.Frame(self._tree, width=1, bg='#b0b0b0')
            line.place(x=bx + bw, y=0, relheight=1.0)
            self._vlines.append(line)

    def _refresh_row(self, sample: str):
        tags = self._tree.item(sample, 'tags')
        self._tree.item(sample, values=self._row_values(sample), tags=tags)

    # ------------------------------------------------------------------
    # In-place cell editing
    # ------------------------------------------------------------------

    def _on_cell_click(self, event):
        if self._active_editor:
            try:
                self._active_editor.destroy()
            except tk.TclError:
                pass
            self._active_editor = None

        region = self._tree.identify_region(event.x, event.y)
        if region != 'cell':
            return

        col_id = self._tree.identify_column(event.x)
        item = self._tree.identify_row(event.y)
        if not item:
            return

        cols = self._tree['columns']
        col_idx = int(col_id[1:]) - 1
        if col_idx < 0 or col_idx >= len(cols):
            return
        col_name = cols[col_idx]

        if col_name == 'sample':
            return
        self._last_col = col_name

        if col_name in self._checkbox_cols:
            current = self._row_data[item].get(col_name, '')
            self._commit_edit(item, col_name, '' if current == 'yes' else 'yes', None)
            return

        self._open_editor(item, col_name)

    def _open_editor(self, item: str, col_name: str):
        """
        Open an in-place text Entry on (item, col_name). Enter or Tab commits
        and advances the editor to the same column of the next row; FocusOut
        commits without advancing; Escape cancels.
        """
        if self._active_editor:
            try:
                self._active_editor.destroy()
            except tk.TclError:
                pass
            self._active_editor = None

        bbox = self._tree.bbox(item, col_name)
        if not bbox:
            return
        x, y, w, h = bbox

        current_val = self._tree.set(item, col_name)

        var = tk.StringVar(value=current_val)
        widget = tk.Entry(self._tree, textvariable=var)

        def _commit(advance):
            self._commit_edit(item, col_name, var.get(), widget)
            if advance:
                self._edit_next_row(item, col_name)

        widget.bind('<Return>', lambda e: (_commit(True), 'break')[1])
        widget.bind('<Tab>',    lambda e: (_commit(True), 'break')[1])
        widget.bind('<FocusOut>', lambda e: _commit(False))
        widget.bind('<Escape>', lambda e, ww=widget: ww.destroy())
        widget.select_range(0, tk.END)

        widget.place(x=x, y=y, width=w, height=h)
        widget.focus_set()
        self._active_editor = widget

    def _edit_next_row(self, item: str, col_name: str):
        """Open the editor on the same column of the row below `item`, if any."""
        children = self._tree.get_children()
        try:
            idx = children.index(item)
        except ValueError:
            return
        if idx + 1 < len(children):
            nxt = children[idx + 1]
            self._tree.see(nxt)
            self._tree.update_idletasks()
            self._open_editor(nxt, col_name)

    def _commit_edit(self, item: str, col_name: str, value: str, widget):
        if widget is not None:
            try:
                widget.destroy()
            except tk.TclError:
                pass
            if self._active_editor is widget:
                self._active_editor = None

        self._row_data[item][col_name] = value
        self._refresh_row(item)

    # ------------------------------------------------------------------
    # Button actions
    # ------------------------------------------------------------------

    def _add_column(self):
        result = {'name': None, 'checkbox': False}
        top = tk.Toplevel(self._root)
        top.title("Add column")
        top.grab_set()
        top.resizable(False, False)

        tk.Label(top, text="Column name:").grid(
            row=0, column=0, padx=10, pady=(10, 4), sticky='w')
        name_var = tk.StringVar()
        tk.Entry(top, textvariable=name_var, width=24).grid(
            row=0, column=1, padx=10, pady=(10, 4))

        checkbox_var = tk.BooleanVar(value=False)
        tk.Checkbutton(top, text="Checkbox values (yes / no)",
                       variable=checkbox_var).grid(
            row=1, column=0, columnspan=2, padx=10, pady=(0, 4), sticky='w')

        def _ok():
            result['name']     = name_var.get().strip()
            result['checkbox'] = checkbox_var.get()
            top.destroy()

        def _cancel():
            top.destroy()

        btn_frame = tk.Frame(top)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=(4, 10))
        tk.Button(btn_frame, text="OK",     command=_ok,     width=8).pack(side=tk.LEFT, padx=6)
        tk.Button(btn_frame, text="Cancel", command=_cancel, width=8).pack(side=tk.LEFT, padx=6)

        top.protocol('WM_DELETE_WINDOW', _cancel)
        self._root.wait_window(top)

        name = result['name']
        if not name:
            return
        if name in self._all_cols():
            messagebox.showwarning("Duplicate",
                                   f"Column '{name}' already exists.",
                                   parent=self._root)
            return
        self._custom_cols.append(name)
        if result['checkbox']:
            self._checkbox_cols.add(name)
        for rd in self._row_data.values():
            rd[name] = ''
        self._setup_columns()
        self._populate_table()

    def _remove_column(self):
        if not self._custom_cols:
            messagebox.showinfo("No columns", "No custom columns to remove.",
                                parent=self._root)
            return

        result = {'name': None}
        top = tk.Toplevel(self._root)
        top.title("Remove column")
        top.grab_set()
        top.resizable(False, False)

        tk.Label(top, text="Select column to remove:").pack(
            padx=14, pady=(12, 4), anchor='w')

        listbox = tk.Listbox(top, selectmode=tk.SINGLE, height=min(8, len(self._custom_cols)),
                             width=30, exportselection=False)
        for col in self._custom_cols:
            listbox.insert(tk.END, col)
        listbox.select_set(0)
        listbox.pack(padx=14, pady=(0, 8))

        def _ok():
            sel = listbox.curselection()
            if sel:
                result['name'] = self._custom_cols[sel[0]]
            top.destroy()

        def _cancel():
            top.destroy()

        btn_frame = tk.Frame(top)
        btn_frame.pack(pady=(0, 12))
        tk.Button(btn_frame, text="Remove", command=_ok,     width=8).pack(side=tk.LEFT, padx=6)
        tk.Button(btn_frame, text="Cancel", command=_cancel, width=8).pack(side=tk.LEFT, padx=6)

        top.protocol('WM_DELETE_WINDOW', _cancel)
        self._root.wait_window(top)

        name = result['name']
        if not name:
            return
        self._custom_cols.remove(name)
        self._checkbox_cols.discard(name)
        for rd in self._row_data.values():
            rd.pop(name, None)
        self._setup_columns()
        self._populate_table()

    def _set_cells(self):
        if not self._custom_cols:
            messagebox.showinfo("No columns", "Add a custom column first.",
                                parent=self._root)
            return

        selected = self._tree.selection()

        top = tk.Toplevel(self._root)
        top.title("Set cells")
        top.grab_set()
        top.resizable(False, False)

        tk.Label(top, text="Column:").grid(
            row=0, column=0, padx=10, pady=(10, 4), sticky='w')
        default_col = (self._last_col if self._last_col in self._custom_cols
                       else self._custom_cols[0])
        col_var = tk.StringVar(value=default_col)
        col_box = ttk.Combobox(top, values=self._custom_cols, textvariable=col_var,
                               state='readonly', width=22)
        col_box.grid(row=0, column=1, padx=10, pady=(10, 4))

        tk.Label(top, text="Value:").grid(
            row=1, column=0, padx=10, pady=4, sticky='w')
        value_frame = tk.Frame(top)
        value_frame.grid(row=1, column=1, padx=10, pady=4, sticky='w')

        value_var = tk.StringVar()

        def _build_value_widget(*_):
            for child in value_frame.winfo_children():
                child.destroy()
            if col_var.get() in self._checkbox_cols:
                value_var.set('yes')
                ttk.Combobox(value_frame, values=['yes', 'no'],
                             textvariable=value_var, state='readonly',
                             width=20).pack()
            else:
                value_var.set('')
                tk.Entry(value_frame, textvariable=value_var, width=23).pack()

        col_box.bind('<<ComboboxSelected>>', _build_value_widget)
        _build_value_widget()

        all_var = tk.BooleanVar(value=False)
        tk.Checkbutton(top, text="Apply to all rows (ignore selection)",
                       variable=all_var).grid(
            row=2, column=0, columnspan=2, padx=10, pady=(4, 0), sticky='w')

        tk.Label(top, text=f"{len(selected)} row(s) selected",
                 fg='#555').grid(
            row=3, column=0, columnspan=2, padx=10, pady=(0, 4), sticky='w')

        result = {'ok': False}

        def _ok():
            result.update(ok=True, col=col_var.get(),
                          value=value_var.get(), all=all_var.get())
            top.destroy()

        def _cancel():
            top.destroy()

        btn_frame = tk.Frame(top)
        btn_frame.grid(row=4, column=0, columnspan=2, pady=(4, 10))
        tk.Button(btn_frame, text="OK",     command=_ok,     width=8).pack(side=tk.LEFT, padx=6)
        tk.Button(btn_frame, text="Cancel", command=_cancel, width=8).pack(side=tk.LEFT, padx=6)

        top.protocol('WM_DELETE_WINDOW', _cancel)
        self._root.wait_window(top)

        if not result['ok']:
            return

        targets = self._tree.get_children() if result['all'] else selected
        if not targets:
            messagebox.showwarning(
                "No selection",
                "Select one or more rows, or check 'Apply to all rows'.",
                parent=self._root)
            return

        col = result['col']
        value = result['value']
        if col in self._checkbox_cols:
            value = 'yes' if value == 'yes' else ''

        for item in targets:
            self._row_data[item][col] = value
            self._refresh_row(item)

    def _move_up(self):
        for item in self._tree.selection():
            idx = self._tree.index(item)
            if idx > 0:
                self._tree.move(item, '', idx - 1)
        self._restripe()

    def _move_down(self):
        children = self._tree.get_children()
        n = len(children)
        for item in reversed(self._tree.selection()):
            idx = self._tree.index(item)
            if idx < n - 1:
                self._tree.move(item, '', idx + 1)
        self._restripe()

    def _finish(self):
        ordered = [self._tree.set(item, 'sample')
                   for item in self._tree.get_children()]
        out_dir = _write_output(
            self._coulter_csv, self._coulter_df, self._row_data, ordered,
            self._custom_cols, self._checkbox_cols)
        messagebox.showinfo("Done", f"Output written to:\n{out_dir}",
                            parent=self._root)
        self._root.destroy()


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _write_output(coulter_csv: Path, coulter_df: pd.DataFrame, row_data: dict,
                  ordered_samples: list, custom_cols: list,
                  checkbox_cols: set) -> Path:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = coulter_csv.parent / f'{timestamp}_coulter_sample_annotation'
    out_dir.mkdir()

    used_keys: set = set()
    sample_keys = {s: _safe_key(s, used_keys) for s in ordered_samples}

    meta_rows = []
    for sample in ordered_samples:
        rd = row_data[sample]
        row = {'sample_name': sample, 'h5_key': sample_keys[sample]}
        for col in custom_cols:
            val = rd.get(col, '')
            if col in checkbox_cols:
                val = 'yes' if val == 'yes' else 'no'
            row[col] = val
        meta_rows.append(row)

    meta_df = pd.DataFrame(meta_rows)

    meta_csv = out_dir / 'metadata.csv'
    meta_df.to_csv(meta_csv, index=False)
    print(f"Written: {meta_csv}")

    h5_path = out_dir / 'data.h5'
    try:
        with pd.HDFStore(str(h5_path), mode='w') as store:
            store.put('/metadata', meta_df, format='table', data_columns=True)

            for sample in ordered_samples:
                key = sample_keys[sample]
                vols = coulter_df[[sample]].dropna().reset_index(drop=True)
                vols.columns = [key]   # original sample name may be HDF5-unsafe
                store.put(f'/data/{key}', vols, format='table',
                          data_columns=True)
                print(f"Written HDF5 key: /data/{key}  "
                      f"({len(vols)} rows, sample='{sample}')")

        print(f"Written: {h5_path}")
    except ImportError:
        print("[warn] 'tables' package not installed — data.h5 not written. "
              "Install with: pip install tables")

    print(f"\n[done] Output: {out_dir}")
    return out_dir


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    coulter_csv = parse_cli_args()
    _check_dependencies()

    coulter_df = _load_coulter(coulter_csv)
    print(f"Loaded Coulter CSV: {coulter_csv.name}  "
          f"({len(coulter_df.columns)} columns, {len(coulter_df)} rows)")

    root = tk.Tk()
    AnnotationWindow(root, coulter_csv, coulter_df)
    root.mainloop()


if __name__ == '__main__':
    main()
