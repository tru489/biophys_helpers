"""
pair_bm_runs.py

Interactive GUI for organizing buoyant mass runs from multiple fluid conditions
(h2o, d2o, optiprep) into paired/triplet groups for population-level SMR
density analysis. Optionally associates Coulter Counter volume data with each
sample group.

Workflow:
    1. The script discovers all sample subdirs in the given superdir that contain
       a *_mass_results folder with a mass_pg column CSV.
    2. If a *_bm_gating folder is also present, the gate thresholds (lower/upper)
       are pre-loaded and saved to the output automatically.
    3. If --coulter is given, column names from that CSV are available as a
       "Coulter Col" dropdown per row. Setting a Coulter column on any run in a
       group auto-fills the same column for all other runs in that group.
    4. A spreadsheet-like table is shown with all discovered samples. For each
       sample, assign a run type (h2o / d2o / optiprep), optional Coulter column,
       and group ID. Custom metadata columns can be added and edited.
    5. Click "Group Selected" to associate multiple runs as a single biological
       sample. Click "Done" to write output.

Output (written to <superdir>/YYYYMMDD_HHMMSS_populationlevel_smr_pairing/):
    metadata.csv   one row per run; all assigned attributes + gate thresholds
    data.h5        pandas HDFStore; /metadata DataFrame + /data/{gid}/{run_type}
                   DataFrames containing the full mass CSV contents per run;
                   /data/{gid}/coulter if a Coulter column was assigned

Usage:
    python pair_bm_runs.py <superdir> [--coulter <csv_path>]
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
from tkinter import filedialog, messagebox
from tkinter import ttk
import yaml


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RUN_TYPES = ['h2o', 'd2o', 'optiprep']

_GROUP_COLORS = [
    '#fff2cc',  # yellow
    '#dae8fc',  # blue
    '#d5e8d4',  # green
    '#f8cecc',  # pink
    '#e1d5e7',  # purple
    '#ffe6cc',  # orange
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_cli_args() -> tuple:
    """Returns (superdir: Path, coulter_path: Path | None)."""
    parser = argparse.ArgumentParser(
        description="Organize buoyant mass runs into paired groups for "
                    "population-level SMR density analysis."
    )
    parser.add_argument('superdir', type=str,
                        help='Path to the experiment superdir')
    parser.add_argument('--coulter', type=str, default=None,
                        metavar='CSV',
                        help='Path to a Coulter Counter CSV whose columns '
                             'are sample names (optional)')
    args = parser.parse_args()

    superdir = Path(args.superdir)
    if not superdir.is_dir():
        raise FileNotFoundError(f"Directory not found: {superdir}")

    coulter_path = None
    if args.coulter is not None:
        coulter_path = Path(args.coulter)
        if not coulter_path.is_file():
            raise FileNotFoundError(f"Coulter file not found: {coulter_path}")

    return superdir, coulter_path


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _collect_sample_dirs(superdir: Path, pattern: re.Pattern) -> list:
    """
    Return all dirs that contain at least one *_mass_results subdir,
    searching at depth 1 then depth 2 from superdir.

    Handles both flat layouts (superdir/sample/mass_results) and
    grouped layouts (superdir/group/sample/mass_results).
    """
    sample_dirs = []
    for d in sorted(superdir.iterdir()):
        if not d.is_dir():
            continue
        if any(pattern.match(s.name) for s in d.iterdir() if s.is_dir()):
            sample_dirs.append(d)
        else:
            for sub in sorted(d.iterdir()):
                if sub.is_dir() and any(pattern.match(s.name)
                                        for s in sub.iterdir() if s.is_dir()):
                    sample_dirs.append(sub)
    return sample_dirs


def _discover_runs(superdir: Path) -> dict:
    """
    Finds BM run data for each sample subdir.

    Returns:
        {sample_name: {'csv_path': Path, 'data': DataFrame,
                       'bm_gate': (lower, upper) | None}}
    """
    run_dir_pattern = re.compile(r'.+_mass_results$')
    result = {}

    for sample_dir in _collect_sample_dirs(superdir, run_dir_pattern):
        run_dirs = sorted(
            d for d in sample_dir.iterdir()
            if d.is_dir() and run_dir_pattern.match(d.name)
        )
        run_dir = run_dirs[-1]

        csv_path = None
        for f in sorted(run_dir.iterdir()):
            if (f.is_file() and f.suffix == '.csv'
                    and not f.name.startswith('curation_index')):
                try:
                    header = pd.read_csv(f, nrows=0)
                except Exception:
                    continue
                if 'mass_pg' in header.columns:
                    csv_path = f
                    break

        if csv_path is None:
            print(f"  [skip] {sample_dir.name}: no valid mass_pg CSV in {run_dir.name}")
            continue

        df = pd.read_csv(csv_path)

        bm_gate = None
        bm_gate_dirs = sorted(
            d for d in sample_dir.iterdir()
            if d.is_dir() and re.search(r'_bm_gating$', d.name)
        )
        if bm_gate_dirs:
            yaml_files = list(bm_gate_dirs[-1].glob('*.yaml'))
            if yaml_files:
                try:
                    gate = yaml.safe_load(yaml_files[0].read_text(encoding='utf-8'))
                    bm_gate = (gate['lower'], gate['upper'])
                except Exception as exc:
                    print(f"  [warn] {sample_dir.name}: could not read bm_gate YAML: {exc}")

        result[sample_dir.name] = {
            'csv_path': csv_path,
            'data':     df,
            'bm_gate':  bm_gate,
        }

    return result


def _load_coulter(path: Path) -> pd.DataFrame:
    """
    Loads a Coulter Counter CSV where each column is a sample and each row is
    a volume measurement. Returns the full DataFrame.
    """
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Main GUI window
# ---------------------------------------------------------------------------

class PairingWindow:
    """
    Spreadsheet-like tkinter window for assigning run types, optional Coulter
    column associations, custom metadata, and group IDs to discovered sample runs.

    Fixed columns: [sample, run_type, group, (coulter_col if Coulter CSV given)]
    Plus any user-added custom columns.

    Clicking a run_type or coulter_col cell opens an in-place Combobox.
    Clicking a custom column cell opens an in-place Entry.
    Setting a coulter_col on any run auto-fills the same value for all other
    runs in the same group.
    """

    def __init__(self, root: tk.Tk, superdir: Path, runs: dict,
                 coulter_df: pd.DataFrame = None, on_back=None):
        self._root = root
        self._superdir = superdir
        self._runs = runs
        self._coulter_df = coulter_df
        self._coulter_cols = list(coulter_df.columns) if coulter_df is not None else []
        self._on_back = on_back

        self._custom_cols: list = []
        self._shared_cols: set = set()
        self._checkbox_cols: set = set()
        self._group_counter: int = 0
        self._group_color_map: dict = {}
        self._active_editor = None
        self._last_col = None

        init_row = {'run_type': '', 'group': ''}
        if coulter_df is not None:
            init_row['coulter_col'] = ''
        self._row_data: dict = {name: dict(init_row) for name in runs}

        root.title(f"pair_bm_runs — {superdir.name}")
        self._build_ui()
        self._populate_table()

    # ------------------------------------------------------------------
    # Column lists
    # ------------------------------------------------------------------

    def _fixed_cols(self) -> list:
        cols = ['sample', 'run_type', 'group']
        if self._coulter_df is not None:
            cols.append('coulter_col')
        return cols

    def _all_cols(self) -> list:
        return self._fixed_cols() + self._custom_cols

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        n = len(self._runs)
        n_gated = sum(1 for v in self._runs.values() if v['bm_gate'])
        coulter_info = (f"   Coulter: {len(self._coulter_cols)} columns"
                        if self._coulter_df is not None else "")
        tk.Label(
            self._root,
            text=(f"{n} sample(s) discovered in {self._superdir.name}   "
                  f"({n_gated} with BM gate){coulter_info}"),
            font=('TkDefaultFont', 10, 'bold'), anchor='w',
        ).pack(fill=tk.X, padx=10, pady=(8, 2))

        tree_frame = tk.Frame(self._root)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 4))

        vsb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL)
        hsb = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)

        self._tree = ttk.Treeview(
            tree_frame, selectmode='extended',
            yscrollcommand=vsb.set, xscrollcommand=hsb.set,
            show='headings', height=24,
        )
        vsb.config(command=self._tree.yview)
        hsb.config(command=self._tree.xview)

        self._tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

        for i, color in enumerate(_GROUP_COLORS):
            self._tree.tag_configure(f'grp{i}', background=color)
        self._tree.tag_configure('ungrouped', background='')

        self._tree.bind('<ButtonRelease-1>', self._on_cell_click)

        self._setup_columns()

        btn_frame = tk.Frame(self._root)
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 8))

        if self._on_back is not None:
            tk.Button(btn_frame, text="← Back",
                      command=self._do_back).pack(side=tk.LEFT, padx=(0, 12))

        tk.Button(btn_frame, text="Add Column",
                  command=self._add_column).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(btn_frame, text="Remove Column",
                  command=self._remove_column).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(btn_frame, text="Group Selected",
                  command=self._group_selected).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(btn_frame, text="Clear Group",
                  command=self._clear_group).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(btn_frame, text="Set Cells…",
                  command=self._set_cells).pack(side=tk.LEFT, padx=(0, 12))
        tk.Button(btn_frame, text="↑",
                  command=self._move_up).pack(side=tk.LEFT, padx=(0, 2))
        tk.Button(btn_frame, text="↓",
                  command=self._move_down).pack(side=tk.LEFT, padx=(0, 12))

        self._done_btn = tk.Button(btn_frame, text="Done",
                                   state=tk.DISABLED, command=self._finish)
        self._done_btn.pack(side=tk.RIGHT)

    def _setup_columns(self):
        cols = self._all_cols()
        self._tree['columns'] = cols
        col_widths = {'sample': 200, 'run_type': 90, 'group': 70,
                      'coulter_col': 220}
        for col in cols:
            w = col_widths.get(col, 160)
            if col == 'coulter_col':
                label = 'Coulter Col'
            elif col in self._checkbox_cols and col in self._shared_cols:
                label = f'★ ☑ {col}'
            elif col in self._checkbox_cols:
                label = f'☑ {col}'
            elif col in self._shared_cols:
                label = f'★ {col}'
            else:
                label = col.replace('_', ' ').title()
            self._tree.heading(col, text=label)
            self._tree.column(col, width=w, minwidth=60, anchor='w',
                              stretch=tk.YES if col == 'sample' else tk.NO)

    def _row_values(self, sample: str) -> list:
        rd = self._row_data[sample]
        base = [sample, rd.get('run_type', ''), rd.get('group', '')]
        if self._coulter_df is not None:
            base.append(rd.get('coulter_col', ''))
        custom = []
        for c in self._custom_cols:
            if c in self._checkbox_cols:
                custom.append('✓' if rd.get(c) == 'yes' else '')
            else:
                custom.append(rd.get(c, ''))
        return base + custom

    def _populate_table(self):
        for item in self._tree.get_children():
            self._tree.delete(item)
        for sample_name in self._runs:
            self._tree.insert('', tk.END, iid=sample_name,
                              values=self._row_values(sample_name))
        self._refresh_colors()

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _refresh_row(self, sample: str):
        self._tree.item(sample, values=self._row_values(sample))

    def _refresh_colors(self):
        for item in self._tree.get_children():
            sample = self._tree.set(item, 'sample')
            gid = self._row_data[sample].get('group', '')
            if gid:
                if gid not in self._group_color_map:
                    self._group_color_map[gid] = (
                        len(self._group_color_map) % len(_GROUP_COLORS))
                tag = f'grp{self._group_color_map[gid]}'
            else:
                tag = 'ungrouped'
            self._tree.item(item, tags=(tag,))

    def _refresh_all(self):
        for sample in self._runs:
            self._refresh_row(sample)
        self._refresh_colors()
        self._check_done()

    def _check_done(self):
        all_typed   = all(self._row_data[n].get('run_type', '') for n in self._runs)
        all_grouped = all(self._row_data[n].get('group', '')    for n in self._runs)
        state = tk.NORMAL if (all_typed and all_grouped) else tk.DISABLED
        self._done_btn.config(state=state)

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

        if col_name in ('sample', 'group'):
            return
        self._last_col = col_name

        if col_name in self._checkbox_cols:
            current = self._row_data[item].get(col_name, '')
            self._commit_edit(item, col_name, '' if current == 'yes' else 'yes', None)
            return

        bbox = self._tree.bbox(item, col_id)
        if not bbox:
            return
        x, y, w, h = bbox

        if col_name == 'run_type':
            self._open_combo_editor(item, col_name, _RUN_TYPES, 12, x, y, w, h)
            return
        if col_name == 'coulter_col':
            self._open_combo_editor(item, col_name, self._coulter_cols, 30,
                                    x, y, w, h)
            return

        current_val = self._tree.set(item, col_name)
        var = tk.StringVar(value=current_val)
        widget = tk.Entry(self._tree, textvariable=var)
        widget.bind('<Return>',
                    lambda e, i=item, c=col_name, v=var, ww=widget:
                    self._commit_edit(i, c, v.get(), ww))
        widget.bind('<FocusOut>',
                    lambda e, i=item, c=col_name, v=var, ww=widget:
                    self._commit_edit(i, c, v.get(), ww))
        widget.bind('<Escape>', lambda e, ww=widget: ww.destroy())
        widget.select_range(0, tk.END)

        widget.place(x=x, y=y, width=w, height=h)
        widget.focus_set()
        self._active_editor = widget

    def _open_combo_editor(self, item, col_name, values, width, x, y, w, h):
        """
        Open a readonly in-place Combobox on (item, col_name).

        Commits on selection (<<ComboboxSelected>>) and cancels on Escape.
        Deliberately NO <FocusOut> commit: a readonly combobox's dropdown is a
        separate popup, so opening it makes the combobox lose focus — a
        FocusOut-destroys-editor handler would tear the widget down just as the
        list appears, forcing repeated clicks. Stale editors are instead cleaned
        up by the next _on_cell_click.
        """
        current_val = self._tree.set(item, col_name)
        widget = ttk.Combobox(self._tree, values=values,
                              state='readonly', width=width)
        widget.set(current_val)
        widget.bind('<<ComboboxSelected>>',
                    lambda e, ww=widget: self._commit_edit(item, col_name,
                                                           ww.get(), ww))
        widget.bind('<Escape>', lambda e, ww=widget: ww.destroy())
        widget.place(x=x, y=y, width=w, height=h)
        widget.focus_set()
        self._active_editor = widget

        # Auto-post the dropdown so the cell-opening click also reveals the
        # list (one click instead of the extra activation click macOS/aqua
        # otherwise requires). Deferred so the creating click finishes first;
        # falls back silently if the private Tk proc is unavailable.
        def _post(ww=widget):
            try:
                ww.tk.call('ttk::combobox::Post', ww)
            except tk.TclError:
                pass
        widget.after_idle(_post)

    def _commit_edit(self, item: str, col_name: str, value: str, widget):
        if widget is not None:
            try:
                widget.destroy()
            except tk.TclError:
                pass
            if self._active_editor is widget:
                self._active_editor = None

        self._row_data[item][col_name] = value

        # Coulter column and shared custom cols are group-level; propagate to all members.
        if col_name == 'coulter_col' and value:
            gid = self._row_data[item].get('group', '')
            if gid:
                for name in self._runs:
                    if self._row_data[name].get('group') == gid:
                        self._row_data[name]['coulter_col'] = value
                self._refresh_all()
                return

        elif col_name in self._shared_cols and value:
            gid = self._row_data[item].get('group', '')
            if gid:
                for name in self._runs:
                    if self._row_data[name].get('group') == gid:
                        self._row_data[name][col_name] = value
                self._refresh_all()
                return

        self._refresh_row(item)
        self._refresh_colors()
        self._check_done()

    # ------------------------------------------------------------------
    # Button actions
    # ------------------------------------------------------------------

    def _group_selected(self):
        selected = self._tree.selection()
        if not selected:
            messagebox.showwarning("No selection", "Select rows to group together.")
            return
        self._group_counter += 1
        gid = f'G{self._group_counter:02d}'
        for item in selected:
            self._row_data[item]['group'] = gid

        for col in self._shared_cols:
            vals = [self._row_data[it][col] for it in selected
                    if self._row_data[it][col]]
            distinct = list(dict.fromkeys(vals))
            if len(distinct) > 1:
                kept = self._resolve_shared_conflict(col, distinct)
                for it in selected:
                    self._row_data[it][col] = kept
            elif len(distinct) == 1:
                for it in selected:
                    self._row_data[it][col] = distinct[0]

        self._refresh_all()

    def _clear_group(self):
        for item in self._tree.selection():
            self._row_data[item]['group'] = ''
        self._refresh_all()

    def _settable_cols(self) -> list:
        """Columns the bulk 'Set Cells' dialog may target (everything editable
        except the read-only 'sample' and the group-managed 'group')."""
        cols = ['run_type']
        if self._coulter_df is not None:
            cols.append('coulter_col')
        return cols + self._custom_cols

    def _set_cells(self):
        selected = self._tree.selection()
        settable = self._settable_cols()

        top = tk.Toplevel(self._root)
        top.title("Set cells")
        top.grab_set()
        top.resizable(False, False)

        tk.Label(top, text="Column:").grid(
            row=0, column=0, padx=10, pady=(10, 4), sticky='w')
        default_col = (self._last_col if self._last_col in settable
                       else settable[0])
        col_var = tk.StringVar(value=default_col)
        col_box = ttk.Combobox(top, values=settable, textvariable=col_var,
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
            col = col_var.get()
            if col == 'run_type':
                value_var.set('')
                ttk.Combobox(value_frame, values=_RUN_TYPES,
                             textvariable=value_var, state='readonly',
                             width=20).pack()
            elif col == 'coulter_col':
                value_var.set('')
                ttk.Combobox(value_frame, values=self._coulter_cols,
                             textvariable=value_var, state='readonly',
                             width=20).pack()
            elif col in self._checkbox_cols:
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

        # Route through _commit_edit so group-level propagation (coulter_col and
        # shared custom columns) and Done-button re-evaluation happen per cell.
        for item in targets:
            self._commit_edit(item, col, value, None)

    def _resolve_shared_conflict(self, col: str, distinct_vals: list) -> str:
        result = {'value': distinct_vals[0]}
        top = tk.Toplevel(self._root)
        top.title(f"Conflict in '{col}'")
        top.grab_set()
        top.resizable(False, False)

        tk.Label(top,
                 text=f"Column '{col}' has conflicting values in this group:",
                 wraplength=320).pack(padx=14, pady=(12, 4), anchor='w')

        choice_var = tk.StringVar(value=distinct_vals[0])
        tk.Radiobutton(top, text="Clear all (leave blank for all members)",
                       variable=choice_var, value='__clear__').pack(padx=14, anchor='w')
        for v in distinct_vals:
            tk.Radiobutton(top, text=f'Keep "{v}" for all members',
                           variable=choice_var, value=v).pack(padx=14, anchor='w')

        def _ok():
            chosen = choice_var.get()
            result['value'] = '' if chosen == '__clear__' else chosen
            top.destroy()

        tk.Button(top, text="OK", command=_ok, width=10).pack(pady=(8, 12))
        top.protocol('WM_DELETE_WINDOW', _ok)
        self._root.wait_window(top)
        return result['value']

    def _add_column(self):
        result = {'name': None, 'shared': False, 'checkbox': False}
        top = tk.Toplevel(self._root)
        top.title("Add column")
        top.grab_set()
        top.resizable(False, False)

        tk.Label(top, text="Column name:").grid(
            row=0, column=0, padx=10, pady=(10, 4), sticky='w')
        name_var = tk.StringVar()
        tk.Entry(top, textvariable=name_var, width=24).grid(
            row=0, column=1, padx=10, pady=(10, 4))

        shared_var = tk.BooleanVar(value=False)
        tk.Checkbutton(top, text="Shared within groups",
                       variable=shared_var).grid(
            row=1, column=0, columnspan=2, padx=10, pady=(4, 0), sticky='w')

        checkbox_var = tk.BooleanVar(value=False)
        tk.Checkbutton(top, text="Checkbox values (yes / no)",
                       variable=checkbox_var).grid(
            row=2, column=0, columnspan=2, padx=10, pady=(0, 4), sticky='w')

        def _ok():
            result['name']     = name_var.get().strip()
            result['shared']   = shared_var.get()
            result['checkbox'] = checkbox_var.get()
            top.destroy()

        def _cancel():
            top.destroy()

        btn_frame = tk.Frame(top)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=(4, 10))
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
        if result['shared']:
            self._shared_cols.add(name)
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
        self._shared_cols.discard(name)
        self._checkbox_cols.discard(name)
        for rd in self._row_data.values():
            rd.pop(name, None)
        self._setup_columns()
        self._populate_table()

    def _move_up(self):
        for item in self._tree.selection():
            idx = self._tree.index(item)
            if idx > 0:
                self._tree.move(item, '', idx - 1)

    def _move_down(self):
        children = self._tree.get_children()
        n = len(children)
        for item in reversed(self._tree.selection()):
            idx = self._tree.index(item)
            if idx < n - 1:
                self._tree.move(item, '', idx + 1)

    def _do_back(self):
        if self._on_back:
            self._on_back()

    def _finish(self):
        ordered = [self._tree.set(item, 'sample')
                   for item in self._tree.get_children()]
        out_dir = _write_output(
            self._superdir, self._runs, self._row_data, ordered,
            self._custom_cols, self._coulter_df)
        messagebox.showinfo("Done", f"Output written to:\n{out_dir}",
                            parent=self._root)
        self._root.destroy()


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _write_output(superdir: Path, runs: dict, row_data: dict,
                  ordered_samples: list, custom_cols: list,
                  coulter_df: pd.DataFrame = None) -> Path:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = superdir / f'{timestamp}_populationlevel_smr_pairing'
    out_dir.mkdir()

    meta_rows = []
    for sample in ordered_samples:
        rd = row_data[sample]
        gate = runs[sample]['bm_gate']
        row = {
            'group_id':       rd.get('group', ''),
            'sample_name':    sample,
            'run_type':       rd.get('run_type', ''),
            'bm_gate_lower':  gate[0] if gate else float('nan'),
            'bm_gate_upper':  gate[1] if gate else float('nan'),
        }
        if coulter_df is not None:
            row['coulter_column'] = rd.get('coulter_col', '')
        for col in custom_cols:
            row[col] = rd.get(col, '')
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
                rd = row_data[sample]
                gid = rd.get('group', '')
                run_type = rd.get('run_type', '')
                if not gid or not run_type:
                    continue
                df = runs[sample]['data']
                key = f'/data/{gid}/{run_type}'
                store.put(key, df, format='table', data_columns=True)
                print(f"Written HDF5 key: {key}  ({len(df)} rows)")

            # Coulter data: one entry per group, written once using the first
            # run in the group that has a coulter_col assigned.
            if coulter_df is not None:
                written_groups = set()
                for sample in ordered_samples:
                    rd = row_data[sample]
                    gid = rd.get('group', '')
                    col_name = rd.get('coulter_col', '')
                    if not gid or not col_name or gid in written_groups:
                        continue
                    if col_name not in coulter_df.columns:
                        print(f"  [warn] Coulter column '{col_name}' not found — skipping")
                        continue
                    coulter_series = coulter_df[[col_name]].dropna()
                    coulter_series = coulter_series.reset_index(drop=True)
                    key = f'/data/{gid}/coulter'
                    store.put(key, coulter_series, format='table',
                              data_columns=True)
                    print(f"Written HDF5 key: {key}  "
                          f"({len(coulter_series)} rows, column='{col_name}')")
                    written_groups.add(gid)

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
    superdir, coulter_path = parse_cli_args()
    state = {'superdir': superdir}

    coulter_df = None
    if coulter_path is not None:
        coulter_df = _load_coulter(coulter_path)
        print(f"Loaded Coulter CSV: {coulter_path.name}  "
              f"({len(coulter_df.columns)} columns, {len(coulter_df)} rows)")

    while True:
        restart = [False]
        superdir = state['superdir']

        print(f"Discovering BM runs in {superdir.name}...")
        runs = _discover_runs(superdir)

        if not runs:
            print(f"No BM data found in {superdir}")
            sys.exit(1)

        n_gated = sum(1 for v in runs.values() if v['bm_gate'])
        print(f"Found {len(runs)} sample(s), {n_gated} with BM gate thresholds.")

        root = tk.Tk()

        def on_back(root=root, restart=restart):
            new = filedialog.askdirectory(title="Select experiment superdir")
            if new:
                state['superdir'] = Path(new)
                restart[0] = True
                root.destroy()

        PairingWindow(root, superdir, runs, coulter_df=coulter_df,
                      on_back=on_back)
        root.mainloop()

        if not restart[0]:
            break


if __name__ == '__main__':
    main()
