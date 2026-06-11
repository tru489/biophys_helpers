"""
pair_bm_runs.py

Interactive GUI for organizing buoyant mass runs from multiple fluid conditions
(h2o, d2o, optiprep) into paired/triplet groups for population-level SMR
density analysis.

Workflow:
    1. The script discovers all sample subdirs in the given superdir that contain
       a *_mass_results folder with a mass_pg column CSV.
    2. If a *_bm_gating folder is also present, the gate thresholds (lower/upper)
       are pre-loaded and saved to the output automatically.
    3. A spreadsheet-like table is shown with all discovered samples. For each
       sample, assign a run type (h2o / d2o / optiprep) and group ID. Custom
       metadata columns (e.g. timepoint, drug) can be added and edited.
    4. Click "Group Selected" to associate multiple runs as a single biological
       sample. Click "Done" to write output.

Output (written to <superdir>/YYYYMMDD_HHMMSS_populationlevel_smr_pairing/):
    metadata.csv   one row per run; all assigned attributes + gate thresholds
    data.h5        pandas HDFStore; /metadata DataFrame + /data/{gid}/{run_type}
                   DataFrames containing the full mass CSV contents per run

Usage:
    python pair_bm_runs.py <superdir>
"""
import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
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

def parse_cli_args() -> Path:
    parser = argparse.ArgumentParser(
        description="Organize buoyant mass runs into paired groups for "
                    "population-level SMR density analysis."
    )
    parser.add_argument('superdir', type=str,
                        help='Path to the experiment superdir')
    args = parser.parse_args()
    p = Path(args.superdir)
    if not p.is_dir():
        raise FileNotFoundError(f"Directory not found: {p}")
    return p


# ---------------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------------

def _discover_runs(superdir: Path) -> dict:
    """
    Finds BM run data for each sample subdir.

    Returns:
        {sample_name: {'csv_path': Path, 'data': DataFrame,
                       'bm_gate': (lower, upper) | None}}
    """
    run_dir_pattern = re.compile(r'.+_mass_results$')
    result = {}

    for sample_dir in sorted(superdir.iterdir()):
        if not sample_dir.is_dir():
            continue

        run_dirs = sorted(
            d for d in sample_dir.iterdir()
            if d.is_dir() and run_dir_pattern.match(d.name)
        )
        if not run_dirs:
            continue
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


# ---------------------------------------------------------------------------
# Main GUI window
# ---------------------------------------------------------------------------

class PairingWindow:
    """
    Spreadsheet-like tkinter window for assigning run types, custom metadata,
    and group IDs to discovered sample runs.

    The Treeview has fixed columns [sample, run_type, group] plus any
    user-added custom columns. Clicking a run_type cell opens an in-place
    Combobox; clicking a custom column cell opens an in-place Entry.
    """

    def __init__(self, root: tk.Tk, superdir: Path, runs: dict, on_back=None):
        self._root = root
        self._superdir = superdir
        self._runs = runs
        self._on_back = on_back

        self._custom_cols: list = []
        self._group_counter: int = 0
        self._group_color_map: dict = {}   # {gid: color_index}
        self._active_editor = None

        # row data keyed by sample_name
        self._row_data: dict = {name: {'run_type': '', 'group': ''}
                                for name in runs}

        root.title(f"pair_bm_runs — {superdir.name}")
        self._build_ui()
        self._populate_table()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        n = len(self._runs)
        n_gated = sum(1 for v in self._runs.values() if v['bm_gate'])
        tk.Label(
            self._root,
            text=(f"{n} sample(s) discovered in {self._superdir.name}   "
                  f"({n_gated} with BM gate)"),
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
        tk.Button(btn_frame, text="Group Selected",
                  command=self._group_selected).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(btn_frame, text="Clear Group",
                  command=self._clear_group).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(btn_frame, text="↑",
                  command=self._move_up).pack(side=tk.LEFT, padx=(0, 2))
        tk.Button(btn_frame, text="↓",
                  command=self._move_down).pack(side=tk.LEFT, padx=(0, 12))

        self._done_btn = tk.Button(btn_frame, text="Done",
                                   state=tk.DISABLED, command=self._finish)
        self._done_btn.pack(side=tk.RIGHT)

    def _setup_columns(self):
        cols = ['sample', 'run_type', 'group'] + self._custom_cols
        self._tree['columns'] = cols
        widths = {'sample': 200, 'run_type': 100, 'group': 80}
        for col in cols:
            w = widths.get(col, 160)
            self._tree.heading(col, text=col.replace('_', ' ').title())
            self._tree.column(col, width=w, minwidth=60, anchor='w',
                              stretch=tk.YES if col == 'sample' else tk.NO)

    def _populate_table(self):
        for item in self._tree.get_children():
            self._tree.delete(item)
        for sample_name in self._runs:
            rd = self._row_data[sample_name]
            values = ([sample_name, rd.get('run_type', ''), rd.get('group', '')]
                      + [rd.get(c, '') for c in self._custom_cols])
            self._tree.insert('', tk.END, iid=sample_name, values=values)
        self._refresh_colors()

    # ------------------------------------------------------------------
    # Row / state helpers
    # ------------------------------------------------------------------

    def _refresh_row(self, sample: str):
        rd = self._row_data[sample]
        values = ([sample, rd.get('run_type', ''), rd.get('group', '')]
                  + [rd.get(c, '') for c in self._custom_cols])
        self._tree.item(sample, values=values)
        self._refresh_colors()
        self._check_done()

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

    def _check_done(self):
        all_typed = all(self._row_data[n].get('run_type', '') for n in self._runs)
        any_grouped = any(self._row_data[n].get('group', '') for n in self._runs)
        state = tk.NORMAL if (all_typed and any_grouped) else tk.DISABLED
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

        bbox = self._tree.bbox(item, col_id)
        if not bbox:
            return
        x, y, w, h = bbox

        current_val = self._tree.set(item, col_name)

        if col_name == 'run_type':
            widget = ttk.Combobox(
                self._tree, values=_RUN_TYPES, state='readonly', width=12)
            widget.set(current_val)
            widget.bind('<<ComboboxSelected>>',
                        lambda e, i=item, c=col_name, ww=widget:
                        self._commit_edit(i, c, ww.get(), ww))
            widget.bind('<FocusOut>',
                        lambda e, i=item, c=col_name, ww=widget:
                        self._commit_edit(i, c, ww.get(), ww))
        else:
            var = tk.StringVar(value=current_val)
            widget = tk.Entry(self._tree, textvariable=var)
            widget.bind('<Return>',
                        lambda e, i=item, c=col_name, v=var, ww=widget:
                        self._commit_edit(i, c, v.get(), ww))
            widget.bind('<FocusOut>',
                        lambda e, i=item, c=col_name, v=var, ww=widget:
                        self._commit_edit(i, c, v.get(), ww))
            widget.bind('<Escape>', lambda e, ww=widget: ww.destroy())

        widget.place(x=x, y=y, width=w, height=h)
        widget.focus_set()
        if col_name != 'run_type':
            widget.select_range(0, tk.END)
        self._active_editor = widget

    def _commit_edit(self, item: str, col_name: str, value: str, widget):
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

    def _group_selected(self):
        selected = self._tree.selection()
        if not selected:
            messagebox.showwarning("No selection", "Select rows to group together.")
            return
        self._group_counter += 1
        gid = f'G{self._group_counter:02d}'
        for item in selected:
            self._row_data[item]['group'] = gid
            self._refresh_row(item)

    def _clear_group(self):
        for item in self._tree.selection():
            self._row_data[item]['group'] = ''
            self._refresh_row(item)

    def _add_column(self):
        name = simpledialog.askstring(
            "Add Column", "Column name:", parent=self._root)
        if not name or not name.strip():
            return
        name = name.strip()
        all_cols = ['sample', 'run_type', 'group'] + self._custom_cols
        if name in all_cols:
            messagebox.showwarning("Duplicate",
                                   f"Column '{name}' already exists.",
                                   parent=self._root)
            return
        self._custom_cols.append(name)
        for rd in self._row_data.values():
            rd[name] = ''
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
            self._superdir, self._runs, self._row_data,
            ordered, self._custom_cols)
        messagebox.showinfo("Done", f"Output written to:\n{out_dir}",
                            parent=self._root)
        self._root.destroy()


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _write_output(superdir: Path, runs: dict, row_data: dict,
                  ordered_samples: list, custom_cols: list) -> Path:
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
    initial_superdir = parse_cli_args()
    state = {'superdir': initial_superdir}

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

        PairingWindow(root, superdir, runs, on_back=on_back)
        root.mainloop()

        if not restart[0]:
            break


if __name__ == '__main__':
    main()
