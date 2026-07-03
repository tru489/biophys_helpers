"""
compile_experiment.py

Automatically discovers and compiles all per-sample data for an experiment
into a pair of HDF5 files. Given a superdir, it walks each sample subdir,
finds every known data type (BM mass, iFXM volume, pairing, gating, images),
loads them, and writes a structured output.

Recognised sub-subdir types (all optional; most-recent used if multiple exist):
    *_mass_results          — mass CSV with mass_pg column
    *_imaging_fxm_results   — stage2_analysis/*_ProcessedVolumes.csv;
                               stage1_image_processing/*_CELLGROUPED.hdf5;
                               stage2_analysis/*_Hdf5PathIndex.csv
    *_pairing_results       — *_PairedSMRVolumes.csv
    *_bm_gating             — YAML with lower/upper thresholds
    *_ifxm-vol_gating       — YAML with lower/upper thresholds

Pairing resolution (priority):
    1. *_pairing_results/*_PairedSMRVolumes.csv (if dir present)
    2. ProcessedVolumes rows where matched_mass is not NaN
    3. None — no pairing key written

Optional Coulter calibration (--coulter <csv>):
    A GUI pairs each sample that has volume data with a Coulter Counter
    column, then a per-sample calibration window finds a scaling factor
    (vol_au → fL) by percentile-matching against the Coulter distribution.
    Calibrated volumes are stored alongside the raw volume data.

Output:
    <superdir>/YYYYMMDD_HHMMSS_compiled/
        experiment_data.h5   — DataFrames (pandas HDFStore)
        images.h5            — per-transit BF image stacks (h5py)

experiment_data.h5 key layout:
    /metadata                               — one row per sample; includes
                                              coulter_column and
                                              calibration_factor when --coulter
                                              is used, plus any custom
                                              annotation columns added in the GUI
    /samples/{safe_name}/mass               — full mass CSV DataFrame
    /samples/{safe_name}/volume             — full ProcessedVolumes DataFrame
                                              (volume column is in vol_au)
    /samples/{safe_name}/volume_calibrated  — same as /volume plus volume_fL
                                              column (in fL); written only when
                                              a calibration factor was accepted
    /samples/{safe_name}/pairing            — paired rows (if available)

images.h5 key layout:
    /{safe_name}/{transit_idx:05d}/bf  — (n_frames, H, W) uint8

{safe_name} replaces - and . with _ so HDF5 key rules are satisfied.
The original sample directory name is preserved in /metadata.sample_name.

Usage:
    python compile_experiment.py <superdir>
    python compile_experiment.py <superdir> --coulter <coulter_csv>
"""
import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox, ttk
import warnings
import h5py
import yaml

# Sample dirs that start with digits (e.g. "0h_pt1") trigger a benign
# NaturalNameWarning from PyTables — suppress it; we always use
# store[key] notation, not attribute-style natural naming.
warnings.filterwarnings('ignore', message='object name is not a valid Python identifier')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_cli_args() -> tuple[Path, Path | None]:
    parser = argparse.ArgumentParser(
        description="Compile per-sample experiment data into a single HDF5 file."
    )
    parser.add_argument('superdir', type=str,
                        help='Path to the experiment superdir')
    parser.add_argument('--coulter', type=str, default=None, metavar='CSV',
                        help='Coulter Counter CSV (columns = sample names, '
                             'rows = per-cell volumes in fL). Triggers pairing '
                             'and calibration GUIs.')
    args = parser.parse_args()
    p = Path(args.superdir)
    if not p.is_dir():
        raise FileNotFoundError(f"Directory not found: {p}")
    coulter_path = None
    if args.coulter is not None:
        coulter_path = Path(args.coulter)
        if not coulter_path.is_file():
            raise FileNotFoundError(f"Coulter file not found: {coulter_path}")
    return p, coulter_path


# ---------------------------------------------------------------------------
# Key sanitisation
# ---------------------------------------------------------------------------

def _safe_key(name: str) -> str:
    """Replace characters invalid in HDF5 key names (- and .) with _."""
    return re.sub(r'[-.]', '_', name)


# ---------------------------------------------------------------------------
# Per-sample discovery
# ---------------------------------------------------------------------------

def _last_matching_dir(parent: Path, pattern: re.Pattern) -> Path | None:
    """Return the lexicographically last subdir whose name matches pattern."""
    matches = sorted(
        d for d in parent.iterdir()
        if d.is_dir() and pattern.search(d.name)
    )
    return matches[-1] if matches else None


def _discover_sample(sample_dir: Path) -> dict:
    """
    Locate the relevant file path for each known data type inside sample_dir.

    Returns a dict with keys:
        mass_path, volume_path, pairing_path, bm_gate_path, ifxm_gate_path,
        hdf5_src_path, hdf5_index_path
    Any key whose source is absent is set to None.
    """
    paths = {
        'mass_path':       None,
        'volume_path':     None,
        'pairing_path':    None,
        'bm_gate_path':    None,
        'ifxm_gate_path':  None,
        'hdf5_src_path':   None,
        'hdf5_index_path': None,
    }

    # --- mass_results ---
    mass_dir = _last_matching_dir(sample_dir, re.compile(r'_mass_results$'))
    if mass_dir is not None:
        for f in sorted(mass_dir.iterdir()):
            if (f.is_file() and f.suffix == '.csv'
                    and not f.name.startswith('curation_index')):
                try:
                    hdr = pd.read_csv(f, nrows=0)
                except Exception:
                    continue
                if 'mass_pg' in hdr.columns:
                    paths['mass_path'] = f
                    break

    # --- imaging_fxm_results ---
    fxm_dir = _last_matching_dir(sample_dir, re.compile(r'_imaging_fxm_results$'))
    if fxm_dir is not None:
        stage2 = fxm_dir / 'stage2_analysis'
        if stage2.is_dir():
            for f in stage2.iterdir():
                if f.is_file() and f.name.endswith('_ProcessedVolumes.csv'):
                    paths['volume_path'] = f
                    break
            idx_files = list(stage2.glob('*_Hdf5PathIndex.csv'))
            if idx_files:
                paths['hdf5_index_path'] = idx_files[0]

        stage1 = fxm_dir / 'stage1_image_processing'
        if stage1.is_dir():
            hdf5_files = list(stage1.glob('*.hdf5'))
            if hdf5_files:
                paths['hdf5_src_path'] = hdf5_files[0]

    # --- pairing_results ---
    pair_dir = _last_matching_dir(sample_dir, re.compile(r'_pairing_results$'))
    if pair_dir is not None:
        for f in pair_dir.iterdir():
            if f.is_file() and f.name.endswith('_PairedSMRVolumes.csv'):
                paths['pairing_path'] = f
                break

    # --- bm_gating ---
    bm_gate_dir = _last_matching_dir(sample_dir, re.compile(r'_bm_gating$'))
    if bm_gate_dir is not None:
        yaml_files = sorted(bm_gate_dir.glob('*.yaml'))
        if yaml_files:
            paths['bm_gate_path'] = yaml_files[0]

    # --- ifxm-vol_gating ---
    ifxm_gate_dir = _last_matching_dir(sample_dir, re.compile(r'_ifxm-vol_gating$'))
    if ifxm_gate_dir is not None:
        yaml_files = sorted(ifxm_gate_dir.glob('*.yaml'))
        if yaml_files:
            paths['ifxm_gate_path'] = yaml_files[0]

    return paths


# ---------------------------------------------------------------------------
# Image stack helpers
# ---------------------------------------------------------------------------

def _pad_stack(frames: list) -> np.ndarray:
    """Stack a list of 2D arrays; zero-pad to max shape if sizes differ."""
    if len(set(f.shape for f in frames)) == 1:
        return np.stack(frames)
    max_h = max(f.shape[0] for f in frames)
    max_w = max(f.shape[1] for f in frames)
    out = np.zeros((len(frames), max_h, max_w), dtype=frames[0].dtype)
    for i, f in enumerate(frames):
        out[i, :f.shape[0], :f.shape[1]] = f
    return out


def _save_images_for_sample(hdf5_src: Path, index_csv: Path,
                             out_grp, sample_name: str) -> int:
    """
    Read per-transit BF frames from the CELLGROUPED source and write stacks
    into out_grp (an open h5py group for this sample).

    Key layout inside out_grp:
        {transit_idx:05d}/bf  — (n_frames, H, W) uint8
    """
    idx_df = pd.read_csv(index_csv)
    n_transits = idx_df['TransitIndex'].nunique()

    with h5py.File(hdf5_src, 'r') as src:
        for transit_id, rows in idx_df.groupby('TransitIndex'):
            bf_frames = [src[p][()] for p in rows['Hdf5PathsBF']]
            bf_stack  = _pad_stack(bf_frames)
            key = f'{int(transit_id):05d}'
            out_grp.create_dataset(f'{key}/bf', data=bf_stack,
                                   compression='gzip', compression_opts=4)

    return n_transits


# ---------------------------------------------------------------------------
# Per-type loaders
# ---------------------------------------------------------------------------

def _load_gate(path: Path) -> tuple[float, float] | None:
    """Read a gating YAML and return (lower, upper), or None on failure."""
    try:
        data = yaml.safe_load(path.read_text(encoding='utf-8'))
        return (float(data['lower']), float(data['upper']))
    except Exception as exc:
        print(f"  [warn] could not read gate YAML {path.name}: {exc}")
        return None


def _load_coulter(path: Path) -> pd.DataFrame:
    """Load Coulter CSV: columns = sample names, rows = per-cell volumes in fL."""
    return pd.read_csv(path)


def _resolve_pairing(volume_df: pd.DataFrame | None,
                     pairing_path: Path | None) -> tuple[pd.DataFrame | None, str]:
    """
    Determine the pairing DataFrame and source label.

    Priority:
        1. pairing_path (PairedSMRVolumes.csv from pairing_results dir)
        2. Non-NaN matched_mass rows in volume_df
        3. None
    Returns (df_or_None, source_label).
    """
    if pairing_path is not None:
        try:
            df = pd.read_csv(pairing_path)
            return df, 'pairing_results'
        except Exception as exc:
            print(f"  [warn] could not read {pairing_path.name}: {exc}")

    if volume_df is not None and 'matched_mass' in volume_df.columns:
        paired = volume_df[volume_df['matched_mass'].notna()].copy()
        if not paired.empty:
            return paired, 'volume_cols'

    return None, 'none'


# ---------------------------------------------------------------------------
# Main compilation
# ---------------------------------------------------------------------------

def compile_experiment(superdir: Path) -> list[dict]:
    """
    Walk every sample subdir, discover data, and return a list of sample
    records with loaded DataFrames ready for writing.
    """
    sample_records = []

    for sample_dir in sorted(superdir.iterdir()):
        if not sample_dir.is_dir():
            continue
        paths = _discover_sample(sample_dir)

        # Skip dirs that look like output dirs (no recognised data)
        if all(v is None for v in paths.values()):
            continue

        name = sample_dir.name

        # Load each data type
        mass_df = None
        if paths['mass_path'] is not None:
            try:
                mass_df = pd.read_csv(paths['mass_path'])
            except Exception as exc:
                print(f"  [warn] {name}: could not read mass CSV: {exc}")

        volume_df = None
        if paths['volume_path'] is not None:
            try:
                volume_df = pd.read_csv(paths['volume_path'])
            except Exception as exc:
                print(f"  [warn] {name}: could not read volume CSV: {exc}")

        pairing_df, pairing_src = _resolve_pairing(
            volume_df, paths['pairing_path'])

        bm_gate = (_load_gate(paths['bm_gate_path'])
                   if paths['bm_gate_path'] else None)
        ifxm_gate = (_load_gate(paths['ifxm_gate_path'])
                     if paths['ifxm_gate_path'] else None)

        has_images = (paths['hdf5_src_path'] is not None
                      and paths['hdf5_index_path'] is not None)

        def _tick(val, label=''):
            return f'ok({label})' if (val is not None and label) else ('ok' if val is not None else '--')

        print(
            f"[{name}]"
            f"  mass={_tick(mass_df)}"
            f"  volume={_tick(volume_df)}"
            f"  pairing={_tick(pairing_df, pairing_src)}"
            f"  bm_gate={_tick(bm_gate)}"
            f"  ifxm_gate={_tick(ifxm_gate)}"
            f"  images={'ok' if has_images else '--'}"
        )

        sample_records.append({
            'name':            name,
            'mass_df':         mass_df,
            'volume_df':       volume_df,
            'pairing_df':      pairing_df,
            'bm_gate':         bm_gate,
            'ifxm_gate':       ifxm_gate,
            'hdf5_src_path':   paths['hdf5_src_path'],
            'hdf5_index_path': paths['hdf5_index_path'],
        })

    return sample_records


# ---------------------------------------------------------------------------
# Annotation + Coulter pairing GUI
# ---------------------------------------------------------------------------

class CompileAnnotationWindow:
    """
    Spreadsheet-style GUI over every discovered sample.

    Fixed columns:
        sample       — sample directory name (read-only).
        coulter_col  — present only when a Coulter CSV was supplied. An in-place
                       readonly Combobox pairs each *volume* sample with a
                       Coulter column; mass-only rows show '—' and are skipped.
                       When present, Done stays disabled until every volume
                       sample is assigned (the original pairing requirement).

    Plus any number of user-added annotation columns (free-text or yes/no
    checkbox), edited in place, with bulk "Set Cells", add/remove-column and
    row reordering — mirroring annotate_coulter_samples.py.

    On Done (self.completed = True):
        self.result           {sample_name: coulter_col}  ('' if unset)
        self.annotations      {sample_name: {col: value}}
        self.custom_cols      ordered annotation column names
        self.checkbox_cols    set of annotation columns that are checkboxes
        self.ordered_samples  sample names in final (possibly reordered) order

    Closing without Done leaves self.completed = False and empty defaults, so
    compilation still proceeds (without annotations or calibration).
    """

    def __init__(self, root: tk.Tk, sample_names: list,
                 volume_names, coulter_cols: list):
        self._root = root
        self._samples = list(sample_names)
        self._volume_names = set(volume_names)
        self._coulter_cols = list(coulter_cols)
        self._has_coulter = bool(coulter_cols)

        self._custom_cols: list = []
        self._checkbox_cols: set = set()
        self._active_editor = None
        self._vlines: list = []
        self._last_col = None

        # Per-sample cell values; the Coulter assignment lives here too, under
        # the 'coulter_col' key.
        self._row_data: dict = {name: {} for name in self._samples}

        # Outputs — these defaults hold if the window is closed without Done.
        self.completed = False
        self.result: dict = {}
        self.annotations: dict = {}
        self.custom_cols: list = []
        self.checkbox_cols: set = set()
        self.ordered_samples: list = list(self._samples)

        root.title('Annotate & pair samples' if self._has_coulter
                   else 'Annotate samples')
        self._build_ui()
        self._populate_table()
        root.protocol('WM_DELETE_WINDOW', self._on_close)

    # ------------------------------------------------------------------
    # Column lists
    # ------------------------------------------------------------------

    def _fixed_cols(self) -> list:
        return ['sample', 'coulter_col'] if self._has_coulter else ['sample']

    def _all_cols(self) -> list:
        return self._fixed_cols() + self._custom_cols

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        n = len(self._samples)
        header = (f"{n} sample(s) discovered — "
                  + ("assign a Coulter column to every volume sample, "
                     "then annotate" if self._has_coulter else "annotate"))
        tk.Label(
            self._root, text=header,
            font=('TkDefaultFont', 10, 'bold'), anchor='w',
        ).pack(fill=tk.X, padx=10, pady=(8, 2))

        # 'clam' renders Treeview row-tag backgrounds reliably (the macOS aqua
        # theme ignores them), which the zebra striping below relies on.
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

        self._tree.tag_configure('evenrow', background='#ffffff')
        self._tree.tag_configure('oddrow', background='#e8eef4')

        self._tree.bind('<ButtonRelease-1>', self._on_cell_click)
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

        self._done_btn = tk.Button(btn_frame, text="Done", command=self._finish)
        self._done_btn.pack(side=tk.RIGHT)

    def _setup_columns(self):
        cols = self._all_cols()
        self._tree['columns'] = cols
        for col in cols:
            if col == 'sample':
                label, w, stretch = 'Sample', 240, tk.YES
            elif col == 'coulter_col':
                label, w, stretch = 'Coulter Column', 200, tk.NO
            elif col in self._checkbox_cols:
                label, w, stretch = f'☑ {col}', 160, tk.NO
            else:
                label, w, stretch = col.replace('_', ' ').title(), 160, tk.NO
            self._tree.heading(col, text=label)
            self._tree.column(col, width=w, minwidth=60, anchor='w',
                              stretch=stretch)

    def _row_values(self, sample: str) -> list:
        rd = self._row_data[sample]
        vals = [sample]
        if self._has_coulter:
            vals.append(rd.get('coulter_col', '')
                        if sample in self._volume_names else '—')
        for c in self._custom_cols:
            if c in self._checkbox_cols:
                vals.append('✓' if rd.get(c) == 'yes' else '')
            else:
                vals.append(rd.get(c, ''))
        return vals

    def _populate_table(self):
        for item in self._tree.get_children():
            self._tree.delete(item)
        for sample_name in self._samples:
            self._tree.insert('', tk.END, iid=sample_name,
                              values=self._row_values(sample_name))
        self._restripe()
        self._check_done()
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

    def _check_done(self):
        """Enabled unless a Coulter CSV is loaded with volume samples unpaired."""
        if not self._has_coulter:
            self._done_btn.config(state=tk.NORMAL)
            return
        all_assigned = all(
            self._row_data[s].get('coulter_col')
            for s in self._samples if s in self._volume_names)
        self._done_btn.config(state=tk.NORMAL if all_assigned else tk.DISABLED)

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
        if col_name == 'coulter_col':
            self._open_coulter_editor(item)
            return

        self._last_col = col_name
        if col_name in self._checkbox_cols:
            current = self._row_data[item].get(col_name, '')
            self._commit_edit(item, col_name, '' if current == 'yes' else 'yes', None)
            return

        self._open_editor(item, col_name)

    def _open_coulter_editor(self, item: str):
        """Readonly Combobox to pick a Coulter column (volume samples only)."""
        if item not in self._volume_names:
            return
        bbox = self._tree.bbox(item, 'coulter_col')
        if not bbox:
            return
        x, y, w, h = bbox

        current = self._row_data[item].get('coulter_col', '')
        widget = ttk.Combobox(self._tree, values=self._coulter_cols,
                              state='readonly', width=30)
        widget.set(current)
        widget.bind('<<ComboboxSelected>>',
                    lambda e, ww=widget: self._commit_coulter(item, ww.get(), ww))
        widget.bind('<Escape>', lambda _, ww=widget: ww.destroy())
        widget.place(x=x, y=y, width=w, height=h)
        widget.focus_set()
        self._active_editor = widget

        def _post(ww=widget):
            try:
                ww.tk.call('ttk::combobox::Post', ww)
            except tk.TclError:
                pass
        widget.after_idle(_post)

    def _commit_coulter(self, item: str, value: str, widget):
        try:
            widget.destroy()
        except tk.TclError:
            pass
        self._active_editor = None
        self._row_data[item]['coulter_col'] = value
        self._tree.set(item, 'coulter_col', value)
        self._check_done()

    def _open_editor(self, item: str, col_name: str):
        """
        Open an in-place text Entry on (item, col_name). Enter or Tab commits
        and advances to the same column of the next row; FocusOut commits
        without advancing; Escape cancels.
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

    # ------------------------------------------------------------------
    # Finish / close
    # ------------------------------------------------------------------

    def _finish(self):
        self.ordered_samples = list(self._tree.get_children())
        self.result = {s: self._row_data[s].get('coulter_col', '')
                       for s in self._samples}
        self.annotations = {
            s: {c: self._row_data[s].get(c, '') for c in self._custom_cols}
            for s in self._samples}
        self.custom_cols = list(self._custom_cols)
        self.checkbox_cols = set(self._checkbox_cols)
        self.completed = True
        self._root.destroy()

    def _on_close(self):
        self.completed = False
        self.result = {}
        self.annotations = {}
        self.custom_cols = []
        self.checkbox_cols = set()
        self.ordered_samples = list(self._samples)
        self._root.destroy()


# ---------------------------------------------------------------------------
# Calibration algorithm
# ---------------------------------------------------------------------------

def _find_calibration_factor(
        ifxm_vols_au: np.ndarray,
        coulter_vols_fL: np.ndarray,
        vol_low: float,
        vol_high: float,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Percentile-matching calibration — Python port of coulter_counter_calibration.m.

    Steps:
      1. Filter Coulter to [vol_low, vol_high].
      2. Compute initial factor = median(Coulter_range) / median(iFXM_all).
      3. Sweep factor ± 3 fL/AU around initial, step 0.01.
      4. For each candidate factor, scale iFXM and score by
         sum(|pct(scaled_iFXM) - pct(Coulter_range)|) across pct 5–95.
      5. Return the factor with the lowest score.

    Returns (best_factor, initial_factor, sweep_factors, sweep_scores).
    Raises ValueError if the Coulter range contains fewer than 10 points.
    """
    if ifxm_vols_au.size < 10:
        raise ValueError(
            f'Too few iFXM data points ({ifxm_vols_au.size}) for calibration. '
            'If an iFXM gate is set, check that the bounds are not too tight.')

    cc_in_range = coulter_vols_fL[
        (coulter_vols_fL >= vol_low) & (coulter_vols_fL <= vol_high)]
    if cc_in_range.size < 10:
        raise ValueError(
            f'Fewer than 10 Coulter points fall within {vol_low}–{vol_high} fL '
            '— widen the calibration range.')

    initial_factor = float(np.median(cc_in_range) / np.median(ifxm_vols_au))

    if not np.isfinite(initial_factor) or initial_factor <= 0:
        raise ValueError(
            f'Initial calibration factor is not finite ({initial_factor:.4g}). '
            'Check that the iFXM and Coulter distributions overlap.')

    sweep_low  = max(0.001, initial_factor - 3.0)
    sweep_high = initial_factor + 3.0 + 1e-9
    factors    = np.arange(sweep_low, sweep_high, 0.01)

    cc_pct  = np.percentile(cc_in_range, range(5, 96))
    scores  = np.full(len(factors), np.inf)
    pct_idx = list(range(5, 96))

    for i, f in enumerate(factors):
        scaled = ifxm_vols_au * f
        if scaled.size < 10:
            continue
        scores[i] = float(np.sum(np.abs(np.percentile(scaled, pct_idx) - cc_pct)))

    best_idx = int(np.argmin(scores))
    return factors[best_idx], initial_factor, factors, scores


# ---------------------------------------------------------------------------
# Calibration GUI (per-sample, sequential)
# ---------------------------------------------------------------------------

class CalibrationWindow:
    """
    Per-sample calibration Toplevel.

    Shows a Coulter histogram with a user-selectable calibration range. On
    "Compute", runs _find_calibration_factor() and draws the score curve and
    scaled-iFXM vs Coulter overlay.

    self.result  — calibration factor (float) on Accept; None on Skip.
    self.window  — the Toplevel; destroyed on either Accept or Skip, which
                   unblocks the caller's cal_root.wait_window(cw.window).
    """

    def __init__(self, parent: tk.Tk, rec: dict, coulter_vols_fL: np.ndarray):
        self.result: float | None = None
        self._rec     = rec
        self._cc_vols = coulter_vols_fL

        # Apply iFXM gate to raw volumes before calibration
        raw_vols = rec['volume_df']['volume'].dropna().values
        gate     = rec.get('ifxm_gate')
        if gate is not None:
            lo, hi = gate
            self._ifxm_vols = raw_vols[(raw_vols >= lo) & (raw_vols <= hi)]
        else:
            self._ifxm_vols = raw_vols

        # Default range: 5th–95th percentile of Coulter, rounded to integers
        default_low  = round(float(np.percentile(coulter_vols_fL, 5)))
        default_high = round(float(np.percentile(coulter_vols_fL, 95)))

        self.window = tk.Toplevel(parent)
        self.window.title(f"{rec['name']} — Coulter Calibration")
        self.window.protocol('WM_DELETE_WINDOW', self._on_skip)
        self._maximise()

        _F = ('TkDefaultFont', 12)      # base font for all tk controls
        _FB = ('TkDefaultFont', 12, 'bold')

        # ---- Range controls (top strip) ----
        ctrl = tk.Frame(self.window)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=12, pady=(10, 4))
        tk.Label(ctrl, text='Calibration range (fL):', font=_F).pack(side=tk.LEFT)
        tk.Label(ctrl, text='  Low:', font=_F).pack(side=tk.LEFT)
        self._low_var  = tk.StringVar(value=str(default_low))
        tk.Entry(ctrl, textvariable=self._low_var,  width=8, font=_F).pack(side=tk.LEFT, padx=(2, 8))
        tk.Label(ctrl, text='High:', font=_F).pack(side=tk.LEFT)
        self._high_var = tk.StringVar(value=str(default_high))
        tk.Entry(ctrl, textvariable=self._high_var, width=8, font=_F).pack(side=tk.LEFT, padx=(2, 12))
        tk.Button(ctrl, text='Compute', font=_F, command=self._compute).pack(side=tk.LEFT)

        # ---- Bottom controls (bottom strip) ----
        # Reserve this strip BEFORE packing the canvas so the Skip / Accept &
        # Next buttons are never pushed off-screen by the expanding figure —
        # that clipping is what hid them on macOS (where 'zoomed' is a no-op).
        bot = tk.Frame(self.window)
        bot.pack(side=tk.BOTTOM, fill=tk.X, padx=12, pady=(2, 10))
        self._factor_label = tk.Label(bot, text='Calibration factor:  —  fL/AU',
                                      font=_FB)
        self._factor_label.pack(side=tk.LEFT)
        tk.Button(bot, text='Skip', font=_F,
                  command=self._on_skip).pack(side=tk.RIGHT, padx=(8, 0))
        self._accept_btn = tk.Button(bot, text='Accept & Next', font=_FB,
                                     state=tk.DISABLED, command=self._on_accept)
        self._accept_btn.pack(side=tk.RIGHT)

        # ---- Matplotlib 2×2 figure (fills the space between the strips) ----
        self._fig, self._axs = plt.subplots(2, 2, figsize=(16, 9))
        self._fig.subplots_adjust(hspace=0.45, wspace=0.35)
        canvas = FigureCanvasTkAgg(self._fig, master=self.window)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True,
                                    padx=8, pady=4)
        self._canvas = canvas

        # Initial state: draw Coulter histogram; blank the other three panels
        for ax in self._axs.flat:
            ax.axis('off')
        self._draw_coulter_hist(default_low, default_high)
        canvas.draw()

        self._best_factor: float | None = None

    def _maximise(self):
        """
        Size the window to fill the screen on every platform.

        'zoomed' only maximises on Windows; on macOS/Linux Tk it is either a
        no-op or raises, leaving the window at the figure's natural size (taller
        than the screen), which clips the bottom button bar. So always set an
        explicit screen-sized geometry, leaving a margin for the menu bar/dock.
        """
        self.window.update_idletasks()
        sw = self.window.winfo_screenwidth()
        sh = self.window.winfo_screenheight()
        self.window.geometry(f'{sw}x{sh - 120}+0+0')
        if sys.platform.startswith('win'):
            try:
                self.window.state('zoomed')
            except tk.TclError:
                pass

    # ------------------------------------------------------------------

    def _parse_range(self) -> tuple[float, float] | None:
        try:
            low  = float(self._low_var.get())
            high = float(self._high_var.get())
            if low >= high or low <= 0:
                raise ValueError
            return low, high
        except (ValueError, tk.TclError):
            messagebox.showerror(
                'Invalid range',
                'Enter positive numbers with Low < High.',
                parent=self.window)
            return None

    def _draw_coulter_hist(self, vol_low: float, vol_high: float):
        ax = self._axs[0, 0]
        ax.cla()
        ax.axis('on')
        cc   = self._cc_vols
        cmin = max(cc.min(), 1e-3)
        bins = np.logspace(np.log10(cmin), np.log10(cc.max()), 60)
        in_range = cc[(cc >= vol_low) & (cc <= vol_high)]
        ax.hist(cc,       bins=bins, color='steelblue', alpha=0.5, label='All Coulter')
        ax.hist(in_range, bins=bins, color='orange',    alpha=0.7, label='Cal. range')
        ax.set_xscale('log')
        ax.set_xlabel('Volume (fL)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Coulter distribution', fontsize=12)
        ax.tick_params(labelsize=10)
        ax.legend(fontsize=10)

    def _compute(self):
        rng = self._parse_range()
        if rng is None:
            return
        vol_low, vol_high = rng

        try:
            best, initial, factors, scores = _find_calibration_factor(
                self._ifxm_vols, self._cc_vols, vol_low, vol_high)
        except ValueError as exc:
            messagebox.showerror('Calibration error', str(exc), parent=self.window)
            return

        self._best_factor = best

        # [0,0] Coulter histogram — redraw with current range
        self._draw_coulter_hist(vol_low, vol_high)

        # [0,1] Score vs calibration factor
        ax01 = self._axs[0, 1]
        ax01.cla()
        ax01.axis('on')
        ax01.plot(factors, scores, lw=1, color='steelblue')
        ax01.axvline(initial, color='orange', lw=1.2, ls='--',
                     label=f'Median: {initial:.3f}')
        ax01.axvline(best,    color='red',    lw=1.2, ls='--',
                     label=f'Best:   {best:.3f}')
        ax01.set_xlabel('Calibration factor (fL/AU)', fontsize=11)
        ax01.set_ylabel('Σ|percentile diff|', fontsize=11)
        ax01.set_title('Calibration score', fontsize=12)
        ax01.tick_params(labelsize=10)
        ax01.legend(fontsize=10)

        # [1,0] Overlay: scaled iFXM vs Coulter (density-normalised)
        ax10 = self._axs[1, 0]
        ax10.cla()
        ax10.axis('on')
        cc_rng     = self._cc_vols[(self._cc_vols >= vol_low) & (self._cc_vols <= vol_high)]
        ifxm_scaled = self._ifxm_vols * best
        vmin = max(vol_low,  1e-3)
        bins = np.logspace(np.log10(vmin), np.log10(vol_high), 50)
        ax10.hist(cc_rng,      bins=bins, density=True, alpha=0.6,
                  color='orange',    label='Coulter')
        ax10.hist(ifxm_scaled, bins=bins, density=True, alpha=0.6,
                  color='steelblue', label='Scaled iFXM')
        ax10.set_xscale('log')
        ax10.set_xlabel('Volume (fL)', fontsize=11)
        ax10.set_ylabel('Density', fontsize=11)
        ax10.set_title('Overlay (calibration range)', fontsize=12)
        ax10.tick_params(labelsize=10)
        ax10.legend(fontsize=10)

        # [1,1] Text summary
        ax11 = self._axs[1, 1]
        ax11.cla()
        ax11.axis('off')
        gate = self._rec.get('ifxm_gate')
        gate_str = (f'{gate[0]:.3g} – {gate[1]:.3g} AU' if gate else 'none')
        summary = (
            f"Sample:          {self._rec['name']}\n\n"
            f"iFXM gate:       {gate_str}\n"
            f"iFXM N (gated):  {len(self._ifxm_vols)}\n"
            f"Coulter N:       {len(self._cc_vols)}\n\n"
            f"Range:           {vol_low:.4g} – {vol_high:.4g} fL\n\n"
            f"Initial factor:  {initial:.4f} fL/AU\n"
            f"Refined factor:  {best:.4f} fL/AU"
        )
        ax11.text(0.05, 0.92, summary, transform=ax11.transAxes,
                  va='top', ha='left', fontsize=12, family='monospace')

        self._canvas.draw()
        self._factor_label.config(
            text=f'Calibration factor:  {best:.4f} fL/AU')
        self._accept_btn.config(state=tk.NORMAL)

    def _on_accept(self):
        self.result   = self._best_factor
        self.vol_low  = float(self._low_var.get())
        self.vol_high = float(self._high_var.get())
        self.window.destroy()

    def _on_skip(self):
        self.result = None
        self.window.destroy()


# ---------------------------------------------------------------------------
# Calibration diagnostic plots
# ---------------------------------------------------------------------------

def _write_calibration_plots(out_dir: Path, cal_plot_data: dict) -> None:
    """
    Write calibration diagnostic plots into out_dir/calibration_plots/.

    One overlay histogram PNG per sample (Coulter vs scaled iFXM, density-
    normalised, log x-axis) plus a single bar chart of all calibration factors.

    cal_plot_data: {sample_name: {'factor': float, 'ifxm_vols': ndarray,
                                   'cc_vols': ndarray,
                                   'vol_low': float, 'vol_high': float}}
    """
    plot_dir = out_dir / 'calibration_plots'
    plot_dir.mkdir()

    names   = list(cal_plot_data.keys())
    factors = [cal_plot_data[n]['factor'] for n in names]

    # Per-sample overlay histograms
    for name, d in cal_plot_data.items():
        factor    = d['factor']
        cc        = d['cc_vols']
        ifxm_sc   = d['ifxm_vols'] * factor
        vol_low   = d['vol_low']
        vol_high  = d['vol_high']

        fig, ax = plt.subplots(figsize=(6, 4))
        vmin = max(vol_low, 1e-3)
        bins = np.logspace(np.log10(vmin), np.log10(vol_high), 50)
        cc_rng = cc[(cc >= vol_low) & (cc <= vol_high)]
        ax.hist(cc_rng,  bins=bins, density=True, alpha=0.6,
                color='orange',    label='Coulter')
        ax.hist(ifxm_sc, bins=bins, density=True, alpha=0.6,
                color='steelblue', label='Scaled iFXM')
        ax.set_xscale('log')
        ax.set_xlabel('Volume (fL)')
        ax.set_ylabel('Density')
        ax.set_title(f'{name}\nfactor = {factor:.4f} fL/AU')
        ax.legend()
        fig.tight_layout()
        safe = re.sub(r'[^\w\-.]', '_', name)
        fig.savefig(plot_dir / f'{safe}_calibration.png', dpi=150)
        plt.close(fig)

    # Bar chart of all calibration factors
    fig, ax = plt.subplots(figsize=(max(5, len(names) * 0.8 + 1.5), 4))
    x = range(len(names))
    ax.bar(x, factors, color='steelblue', edgecolor='white', linewidth=0.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Calibration factor (fL/AU)')
    ax.set_title('Calibration factors per sample')
    fig.tight_layout()
    fig.savefig(plot_dir / 'calibration_factors.png', dpi=150)
    plt.close(fig)

    print(f"Calibration plots written -> {plot_dir}")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _write_output(superdir: Path, sample_records: list,
                  pairing: dict | None = None,
                  calibration: dict | None = None,
                  cal_plot_data: dict | None = None,
                  annotations: dict | None = None,
                  custom_cols: list | None = None,
                  checkbox_cols: set | None = None,
                  ordered_samples: list | None = None) -> Path:
    pairing       = pairing       or {}
    calibration   = calibration   or {}
    cal_plot_data = cal_plot_data or {}
    annotations   = annotations   or {}
    custom_cols   = custom_cols   or []
    checkbox_cols = checkbox_cols or set()

    # Honour the row order chosen in the annotation window, if provided.
    if ordered_samples:
        order = {name: i for i, name in enumerate(ordered_samples)}
        sample_records = sorted(
            sample_records, key=lambda r: order.get(r['name'], len(order)))

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = superdir / f'{timestamp}_compiled'
    out_dir.mkdir()
    h5_path        = out_dir / 'experiment_data.h5'
    images_h5_path = out_dir / 'images.h5'

    meta_rows = []
    for rec in sample_records:
        name = rec['name']
        bm   = rec['bm_gate']
        ifxm = rec['ifxm_gate']
        entry = {
            'sample_name':       name,
            'hdf5_key':          _safe_key(name),
            'has_mass':          rec['mass_df'] is not None,
            'has_volume':        rec['volume_df'] is not None,
            'has_pairing':       rec['pairing_df'] is not None,
            'has_bm_gate':       bm is not None,
            'has_ifxm_gate':     ifxm is not None,
            'has_images':        rec['hdf5_src_path'] is not None,
            'bm_gate_lower':     bm[0]   if bm   else float('nan'),
            'bm_gate_upper':     bm[1]   if bm   else float('nan'),
            'ifxm_gate_lower':   ifxm[0] if ifxm else float('nan'),
            'ifxm_gate_upper':   ifxm[1] if ifxm else float('nan'),
            'coulter_column':    pairing.get(name, ''),
            'calibration_factor': calibration.get(name) or float('nan'),
        }
        # Append custom annotation columns (checkbox cols normalised to yes/no).
        row_ann = annotations.get(name, {})
        for col in custom_cols:
            val = row_ann.get(col, '')
            if col in checkbox_cols:
                val = 'yes' if val == 'yes' else 'no'
            entry[col] = val
        meta_rows.append(entry)

    meta_df = pd.DataFrame(meta_rows)

    with pd.HDFStore(str(h5_path), mode='w') as store:
        store.put('/metadata', meta_df, format='table', data_columns=True)

        for rec in sample_records:
            name     = rec['name']
            key_base = f'/samples/{_safe_key(name)}'

            if rec['mass_df'] is not None:
                store.put(f'{key_base}/mass', rec['mass_df'],
                          format='table', data_columns=True)

            if rec['volume_df'] is not None:
                store.put(f'{key_base}/volume', rec['volume_df'],
                          format='table', data_columns=True)

                factor = calibration.get(name)
                if factor is not None:
                    cal_df = rec['volume_df'].copy()
                    cal_df['volume_fL'] = cal_df['volume'] * factor
                    store.put(f'{key_base}/volume_calibrated', cal_df,
                              format='table', data_columns=True)
                    print(f"  [{name}] volume_calibrated written "
                          f"(factor = {factor:.4f} fL/AU)")

            if rec['pairing_df'] is not None:
                store.put(f'{key_base}/pairing', rec['pairing_df'],
                          format='table', data_columns=True)

    print(f"\nCompiled {len(sample_records)} sample(s) -> {h5_path}")

    image_candidates = [r for r in sample_records
                        if r['hdf5_src_path'] and r['hdf5_index_path']]
    if image_candidates:
        print(f"Writing image stacks for {len(image_candidates)} sample(s) "
              f"-> {images_h5_path}")
        with h5py.File(str(images_h5_path), 'w') as img_store:
            for rec in image_candidates:
                grp = img_store.require_group(_safe_key(rec['name']))
                n   = _save_images_for_sample(
                    rec['hdf5_src_path'], rec['hdf5_index_path'],
                    grp, rec['name'])
                print(f"  [{rec['name']}] {n} transits")

    if cal_plot_data:
        _write_calibration_plots(out_dir, cal_plot_data)

    return out_dir


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    superdir, coulter_path = parse_cli_args()
    print(f"Compiling {superdir.name}...")
    records = compile_experiment(superdir)
    if not records:
        print("No sample data found.")
        sys.exit(1)

    coulter_df = None
    coulter_cols: list = []
    if coulter_path is not None:
        coulter_df = _load_coulter(coulter_path)
        coulter_cols = list(coulter_df.columns)
        print(f"Loaded Coulter CSV: {coulter_path.name}  "
              f"({len(coulter_df.columns)} columns, {len(coulter_df)} rows)")

    # --- Phase 1: annotation + (optional) Coulter pairing GUI ---
    ann_root = tk.Tk()
    volume_names = {r['name'] for r in records if r['volume_df'] is not None}
    aw = CompileAnnotationWindow(
        ann_root, [r['name'] for r in records], volume_names, coulter_cols)
    ann_root.mainloop()   # blocks until the window destroys ann_root

    if not aw.completed:
        print("[warn] annotation window closed without Done — "
              "compiling without annotations or calibration.")

    pairing         = aw.result
    annotations     = aw.annotations
    custom_cols     = aw.custom_cols
    checkbox_cols   = aw.checkbox_cols
    ordered_samples = aw.ordered_samples

    calibration:  dict = {}
    cal_plot_data: dict = {}

    # --- Phase 2: per-sample calibration (only with --coulter + assignments) ---
    if coulter_path is not None and any(pairing.values()):
        records_by_name = {r['name']: r for r in records}
        cal_root = tk.Tk()
        cal_root.withdraw()

        paired_items = [(n, c) for n, c in pairing.items() if c]
        print(f"\nStarting calibration for {len(paired_items)} sample(s)...")
        for sample_name, cc_col in paired_items:
            rec     = records_by_name[sample_name]
            cc_vols = coulter_df[cc_col].dropna().values
            print(f"  {sample_name} vs Coulter column '{cc_col}'  "
                  f"(N={len(cc_vols)})")
            cw = CalibrationWindow(cal_root, rec, cc_vols)
            cal_root.wait_window(cw.window)
            calibration[sample_name] = cw.result
            if cw.result is not None:
                print(f"    -> accepted  factor = {cw.result:.4f} fL/AU")
                cal_plot_data[sample_name] = {
                    'factor':    cw.result,
                    'ifxm_vols': cw._ifxm_vols,
                    'cc_vols':   cw._cc_vols,
                    'vol_low':   cw.vol_low,
                    'vol_high':  cw.vol_high,
                }
            else:
                print(f"    -> skipped")

        cal_root.destroy()

    _write_output(superdir, records, pairing, calibration, cal_plot_data,
                  annotations, custom_cols, checkbox_cols, ordered_samples)


if __name__ == '__main__':
    main()
