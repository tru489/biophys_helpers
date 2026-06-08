"""
gate_experiment_subfolder.py

Interactive GUI for gating buoyant mass (BM) or iFXM volume data across all
sample subfolders in an experiment superdir. One column per sample is shown in
the GUI. After gating, a YAML file recording the gate bounds is written into
each sample subfolder, and a timestamped summary folder is written into the
superdir.

Workflow:
    1. A data-type selection dialog asks whether you are gating Buoyant Mass
       or iFXM Volume data.
    2. The script discovers the relevant data file for each sample subdir and
       loads the target column (mass_pg for BM, volume for iFXM).
    3. A scrollable list of sample names is shown. The user multi-selects a
       group and clicks "Set cutoffs for selection".
    4. A histogram window opens showing all selected samples overlaid with
       shared bin edges. The user clicks to set lower then upper cutoffs.
    5. Steps 3-4 repeat until all samples are assigned. "Done" becomes available.
    6. On "Done":
         - A YAML gate file is written into each sample subfolder:
             <superdir_name>_<mode>_gate.yaml
         - A summary folder is written into the superdir:
             <YYMMDD.HHMMSS>_<mode>_gating_summary/
               cutoff_log.txt
               cutoff_stats.csv
               histograms/group_NN.png

Expected directory structure:

    BM:
        <superdir>/<sample_subdir>/<name>_mass_results/<date>_<name>.csv
        Target column: mass_pg

    iFXM:
        <superdir>/<sample_subdir>/<YYYYMMDD_HHMMSS>_imaging_fxm_results/
            stage2_analysis/<sample>_ProcessedVolumes.csv
        Target column: volume

Usage:
    python gate_experiment_subfolder.py <superdir>
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
from tkinter import messagebox
import yaml


# ---------------------------------------------------------------------------
# Mode configuration
# ---------------------------------------------------------------------------

_MODE = {
    'bm': {
        'label':       'Buoyant Mass',
        'unit':        'pg',
        'xlabel':      'Buoyant Mass (pg)',
        'scale':       'linear',
        'bins':        lambda vals: np.linspace(vals.min(), vals.max(), 201),
        'xlim':        None,
        'data_type':    'bm',
        'dir_suffix':   '_bm_gating_summary',
        'yaml_suffix':  '_bm_gate.yaml',
        'yaml_dir_tag': 'bm_gating',
    },
    'ifxm': {
        'label':        'iFXM Volume',
        'unit':         'fL',
        'xlabel':       'Volume (fL)',
        'scale':        'linear',
        'bins':         lambda vals: np.linspace(vals.min(), vals.max(), 201),
        'xlim':         None,
        'data_type':    'ifxm_volume',
        'dir_suffix':   '_ifxm_volume_gating_summary',
        'yaml_suffix':  '_ifxm_volume_gate.yaml',
        'yaml_dir_tag': 'ifxm-vol_gating',
    },
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_cli_args() -> Path:
    """
    Parses CLI args, returning the path to the experiment superdir.

    Raises:
        FileNotFoundError: directory does not exist

    Returns:
        Path: path to the experiment superdir
    """
    parser = argparse.ArgumentParser(
        description="Interactively gate BM or iFXM data across sample "
                    "subfolders of an experiment superdir."
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

def _discover_bm(superdir: Path) -> dict:
    """
    Finds buoyant mass data for each sample subdir in superdir.

    Searches two levels deep (superdir → sample_subdir → *_mass_results) for
    CSVs containing a mass_pg column. If a sample has multiple mass_results
    dirs, the lexicographically last one is used (newest timestamp).
    Files named curation_index*.csv are skipped.

    Args:
        superdir (Path): experiment superdir

    Returns:
        dict: {sample_name (str): np.ndarray of mass_pg values}
    """
    run_dir_pattern = re.compile(r'.+_mass_results$')
    data = {}

    for sample_dir in sorted(superdir.iterdir()):
        if not sample_dir.is_dir():
            continue
        run_dirs = sorted(
            d for d in sample_dir.iterdir()
            if d.is_dir() and run_dir_pattern.match(d.name)
        )
        if not run_dirs:
            continue
        run_dir = run_dirs[-1]      # most recent if multiple

        for f in sorted(run_dir.iterdir()):
            if (f.is_file() and f.suffix == '.csv'
                    and not f.name.startswith('curation_index')):
                df = pd.read_csv(f)
                if 'mass_pg' not in df.columns:
                    continue
                vals = df['mass_pg'].dropna().values
                if len(vals) > 0:
                    data[sample_dir.name] = vals
                break

    return data


def _discover_ifxm(superdir: Path) -> dict:
    """
    Finds iFXM volume data for each sample subdir in superdir.

    Searches two levels deep (superdir → sample_subdir →
    *_imaging_fxm_results/stage2_analysis/*_ProcessedVolumes.csv) for CSVs
    containing a volume column.

    Args:
        superdir (Path): experiment superdir

    Returns:
        dict: {sample_name (str): np.ndarray of volume values}
    """
    run_dir_pattern = re.compile(r'\d{8}_\d{6}_imaging_fxm_results$')
    data = {}

    for sample_dir in sorted(superdir.iterdir()):
        if not sample_dir.is_dir():
            continue
        run_dirs = sorted(
            d for d in sample_dir.iterdir()
            if d.is_dir() and run_dir_pattern.match(d.name)
        )
        if not run_dirs:
            continue
        run_dir = run_dirs[-1]      # most recent if multiple
        stage2 = run_dir / 'stage2_analysis'
        if not stage2.is_dir():
            continue
        for f in stage2.iterdir():
            if f.is_file() and f.name.endswith('_ProcessedVolumes.csv'):
                df = pd.read_csv(f)
                if 'volume' not in df.columns:
                    continue
                vals = df['volume'].dropna().values
                if len(vals) > 0:
                    data[sample_dir.name] = vals
                break

    return data


# ---------------------------------------------------------------------------
# Data type selection dialog
# ---------------------------------------------------------------------------

def _ask_data_type(root: tk.Tk) -> str:
    """
    Shows a modal dialog asking whether the user is gating Buoyant Mass or
    iFXM Volume data. Blocks until one button is clicked.

    If the user closes the window without selecting, the application exits.

    Args:
        root (tk.Tk): parent window

    Returns:
        str: 'bm' or 'ifxm'
    """
    result = {'mode': None}

    top = tk.Toplevel(root)
    top.title("Select data type")
    top.grab_set()
    top.resizable(False, False)

    tk.Label(top, text="What type of data are you gating?",
             font=('TkDefaultFont', 11), pady=12, padx=20).pack()

    btn_frame = tk.Frame(top)
    btn_frame.pack(padx=20, pady=(0, 16))

    def _select(mode):
        result['mode'] = mode
        top.destroy()

    tk.Button(btn_frame, text="Buoyant Mass", width=22,
              command=lambda: _select('bm')).pack(side=tk.LEFT, padx=8)
    tk.Button(btn_frame, text="iFXM Volume", width=22,
              command=lambda: _select('ifxm')).pack(side=tk.LEFT, padx=8)

    top.protocol('WM_DELETE_WINDOW', lambda: sys.exit(0))
    root.wait_window(top)

    if result['mode'] is None:
        sys.exit(0)
    return result['mode']


# ---------------------------------------------------------------------------
# Cutoff window
# ---------------------------------------------------------------------------

class CutoffWindow:
    """
    Modal tkinter Toplevel containing an embedded matplotlib histogram.

    Displays all selected samples as overlaid histograms with shared bin edges.
    Bin layout and axis scale are determined by the active data mode ('bm' or
    'ifxm'). The user clicks once to set a lower cutoff (red dashed line) and
    again to set an upper cutoff (blue dashed line), with the accepted region
    shaded green. Reset clears both lines and restarts. Accept is only enabled
    once both cutoffs are set.

    Result stored in self.result as (lower, upper), or None if closed without
    accepting.
    """

    def __init__(self, parent: tk.Tk, selected_cols: list,
                 data: dict, mode: str):
        self.result = None
        self._lower = None
        self._upper = None
        self._state = 0         # 0=awaiting lower, 1=awaiting upper, 2=both set
        self._vlines = []
        self._patch = None
        self._mode = mode

        self._top = tk.Toplevel(parent)
        self._top.title("Set cutoffs")
        self._top.grab_set()

        self._fig, self._ax = plt.subplots(figsize=(14, 5))
        self._draw_histograms(selected_cols, data)

        canvas = FigureCanvasTkAgg(self._fig, master=self._top)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._canvas = canvas
        self._cid = canvas.mpl_connect('button_press_event', self._on_click)

        info_frame = tk.Frame(self._top)
        info_frame.pack(fill=tk.X, padx=10, pady=(4, 0))
        self._status_var = tk.StringVar(value="Click to set lower cutoff.")
        tk.Label(info_frame, textvariable=self._status_var,
                 anchor='w').pack(side=tk.LEFT)

        btn_frame = tk.Frame(self._top)
        btn_frame.pack(fill=tk.X, padx=10, pady=6)
        tk.Button(btn_frame, text="Reset",
                  command=self._reset).pack(side=tk.LEFT, padx=(0, 8))
        self._accept_btn = tk.Button(btn_frame, text="Accept",
                                     state=tk.DISABLED, command=self._accept)
        self._accept_btn.pack(side=tk.RIGHT)

        self._top.update_idletasks()
        pw = parent.winfo_width();  ph = parent.winfo_height()
        px = parent.winfo_rootx();  py = parent.winfo_rooty()
        tw = self._top.winfo_width(); th = self._top.winfo_height()
        self._top.geometry(f"+{px + (pw - tw) // 2}+{py + (ph - th) // 2}")

        parent.wait_window(self._top)

    def _draw_histograms(self, selected_cols: list, data: dict):
        """
        Renders overlaid histograms for all selected samples onto self._ax.
        Bin edges and axis scale are determined by _MODE[self._mode]. Bins are
        shared across all columns so bars align for easy visual comparison.

        Args:
            selected_cols: sample names to plot
            data: mapping of sample name to value array
        """
        cfg = _MODE[self._mode]
        ax = self._ax
        ax.clear()

        arrays = [data[col][~np.isnan(data[col])] for col in selected_cols]
        arrays = [a for a in arrays if len(a) > 0]
        if not arrays:
            return

        all_vals = np.concatenate(arrays)
        shared_bins = cfg['bins'](all_vals)

        for col, vals in zip(selected_cols, arrays):
            ax.hist(vals, bins=shared_bins, alpha=0.5, edgecolor='black',
                    linewidth=0.3, label=col)
        ax.set_xscale(cfg['scale'])
        if cfg['xlim']:
            ax.set_xlim(*cfg['xlim'])
        ax.set_xlabel(cfg['xlabel'])
        ax.set_ylabel('count')
        ax.legend(fontsize=7, loc='upper right')
        self._fig.tight_layout()

    def _on_click(self, event):
        """
        Handles matplotlib mouse click events.
        State 0: first click sets lower cutoff (red dashed line).
        State 1: second click sets upper cutoff (blue dashed line) and shades
                 the accepted region. Accept button is then enabled.
        Clicks outside the axes or with upper <= lower are ignored/warned.
        """
        if event.inaxes is None or event.xdata is None:
            return
        x = event.xdata

        if self._state == 0:
            self._lower = x
            line = self._ax.axvline(x, color='red', linestyle='--', linewidth=1.2)
            self._ax.text(x, self._ax.get_ylim()[1], f'{x:.3g}',
                          color='red', fontsize=8, va='top', ha='right')
            self._vlines.append(line)
            self._state = 1
            self._status_var.set(f"Lower: {x:.4g}  —  Click to set upper cutoff.")

        elif self._state == 1:
            if x <= self._lower:
                self._status_var.set("Upper must be greater than lower. Click again.")
                return
            self._upper = x
            line = self._ax.axvline(x, color='steelblue', linestyle='--', linewidth=1.2)
            self._ax.text(x, self._ax.get_ylim()[1], f'{x:.3g}',
                          color='steelblue', fontsize=8, va='top', ha='left')
            self._vlines.append(line)
            self._patch = self._ax.axvspan(
                self._lower, self._upper, alpha=0.12, color='green')
            self._state = 2
            self._status_var.set(
                f"Lower: {self._lower:.4g}   Upper: {self._upper:.4g}")
            self._accept_btn.config(state=tk.NORMAL)

        self._canvas.draw()

    def _reset(self):
        """Clears all cutoff lines and shading; resets state to awaiting lower cutoff."""
        for line in self._vlines:
            line.remove()
        self._vlines.clear()
        if self._patch is not None:
            self._patch.remove()
            self._patch = None
        self._lower = None
        self._upper = None
        self._state = 0
        self._accept_btn.config(state=tk.DISABLED)
        self._status_var.set("Click to set lower cutoff.")
        self._canvas.draw()

    def _accept(self):
        """Records the current (lower, upper) as self.result and closes the window."""
        self.result = (self._lower, self._upper)
        plt.close(self._fig)
        self._top.destroy()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow:
    """
    Primary application window. Shows all discovered sample names in a
    scrollable multi-select listbox. The user repeatedly selects groups,
    sets cutoffs via CutoffWindow, and repeats until all samples are assigned.
    "Done" is only enabled once every sample has been processed, at which
    point YAML gate files and a summary folder are written.
    """

    def __init__(self, root: tk.Tk, superdir: Path, sample_dirs: dict,
                 columns: list, data: dict, mode: str):
        self._root = root
        self._superdir = superdir
        self._sample_dirs = sample_dirs     # {sample_name: Path to sample subdir}
        self._columns = columns
        self._data = data
        self._mode = mode
        self._cutoffs: dict = {}
        self._groups: list = []
        self._remaining: list = list(columns)

        root.title(f"gate_experiment_subfolder  [{_MODE[mode]['label']}]")

        self._header_var = tk.StringVar()
        tk.Label(root, textvariable=self._header_var,
                 font=('TkDefaultFont', 11, 'bold'),
                 anchor='w').pack(fill=tk.X, padx=10, pady=(10, 4))

        list_frame = tk.Frame(root)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10)
        scrollbar = tk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self._listbox = tk.Listbox(
            list_frame, selectmode=tk.MULTIPLE, exportselection=False,
            yscrollcommand=scrollbar.set, height=24, width=80)
        scrollbar.config(command=self._listbox.yview)
        self._listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._listbox.bind('<<ListboxSelect>>', self._on_select)

        btn_frame = tk.Frame(root)
        btn_frame.pack(fill=tk.X, padx=10, pady=8)
        self._set_btn = tk.Button(
            btn_frame, text="Set cutoffs for selection",
            state=tk.DISABLED, command=self._open_cutoff_window)
        self._set_btn.pack(side=tk.LEFT, padx=(0, 8))
        self._done_btn = tk.Button(
            btn_frame, text="Done", state=tk.DISABLED, command=self._finish)
        self._done_btn.pack(side=tk.RIGHT)

        self._refresh_list()

    def _refresh_list(self):
        """Rebuilds the listbox from remaining unprocessed samples and updates the header."""
        self._listbox.delete(0, tk.END)
        for col in self._remaining:
            self._listbox.insert(tk.END, col)
        n_done = len(self._cutoffs)
        n_total = len(self._columns)
        self._header_var.set(
            f"Samples remaining: {n_total - n_done} / {n_total}   "
            f"[{self._superdir.name}]")
        self._done_btn.config(
            state=tk.NORMAL if n_done == n_total else tk.DISABLED)
        self._set_btn.config(state=tk.DISABLED)

    def _on_select(self, _event):
        """Enables/disables "Set cutoffs" button based on whether anything is selected."""
        has_selection = bool(self._listbox.curselection())
        self._set_btn.config(state=tk.NORMAL if has_selection else tk.DISABLED)

    def _open_cutoff_window(self):
        """
        Opens a CutoffWindow for the currently selected samples. On acceptance,
        records the cutoffs, appends to _groups, removes processed samples from
        the listbox, and refreshes the header counter.
        """
        indices = self._listbox.curselection()
        if not indices:
            return
        selected = [self._remaining[i] for i in indices]
        win = CutoffWindow(self._root, selected, self._data, self._mode)
        if win.result is None:
            return
        lower, upper = win.result
        for col in selected:
            self._cutoffs[col] = (lower, upper)
            self._remaining.remove(col)
        self._groups.append((lower, upper, selected))
        self._refresh_list()

    def _finish(self):
        """Writes YAML gate files and summary folder, then closes the application."""
        summary_dir = _write_output(
            self._superdir, self._sample_dirs, self._columns,
            self._data, self._cutoffs, self._groups, self._mode)
        messagebox.showinfo("Done", f"Output written to:\n{summary_dir}")
        self._root.destroy()


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _write_yaml_files(superdir: Path, sample_dirs: dict,
                      cutoffs: dict, mode: str, timestamp: str):
    """
    Creates a gating subdir inside each sample subfolder and writes a YAML
    gate file recording the bounds applied to that sample.

    Subdir:    <sample_subdir>/<YYMMDD_HHMMSS>_<mode_tag>_gating/
    File:      <sample_subdir_name>_<mode>_gate.yaml
    Content:
        experiment: <superdir_name>
        data_type:  bm | ifxm_volume
        lower:      <float>
        upper:      <float>

    Args:
        superdir:    experiment superdir (name used in YAML and filename)
        sample_dirs: mapping of sample_name to sample subdir Path
        cutoffs:     mapping of sample_name to (lower, upper)
        mode:        'bm' or 'ifxm'
        timestamp:   YYMMDD_HHMMSS string used to name the subdir
    """
    cfg = _MODE[mode]
    subdir_name = f"{timestamp}_{cfg['yaml_dir_tag']}"

    for sample, (lo, hi) in cutoffs.items():
        sample_dir = sample_dirs[sample]
        gate_dir = sample_dir / subdir_name
        gate_dir.mkdir(exist_ok=True)
        fname = f"{sample_dir.name}_{cfg['yaml_suffix'].lstrip('_')}"
        out_path = gate_dir / fname
        payload = {
            'experiment': superdir.name,
            'data_type':  cfg['data_type'],
            'lower':      float(lo),
            'upper':      float(hi),
        }
        with open(out_path, 'w') as fh:
            yaml.dump(payload, fh, default_flow_style=False, sort_keys=False)
        print(f"Written: {out_path}")


def _write_output(superdir: Path, sample_dirs: dict, columns: list,
                  data: dict, cutoffs: dict,
                  groups: list, mode: str) -> Path:
    """
    Writes per-sample YAML gate files and a timestamped summary folder into
    the superdir.

    Summary folder structure:
        <superdir>/<YYMMDD.HHMMSS>_<mode>_gating_summary/
          cutoff_log.txt
          cutoff_stats.csv
          histograms/
            group_01.png
            group_02.png
            ...

    Args:
        superdir:    experiment superdir
        sample_dirs: mapping of sample_name to sample subdir Path
        columns:     all sample names in original order
        data:        mapping of sample_name to full (ungated) value array
        cutoffs:     mapping of sample_name to (lower, upper)
        groups:      ordered list of (lower, upper, [sample_names])
        mode:        'bm' or 'ifxm'

    Returns:
        Path: the created summary directory
    """
    timestamp = datetime.now().strftime('%y%m%d.%H%M%S')
    suffix = _MODE[mode]['dir_suffix']
    summary_dir = superdir / f'{timestamp}{suffix}'
    summary_dir.mkdir()

    hist_dir = summary_dir / 'histograms'
    hist_dir.mkdir()

    ts_yaml = datetime.now().strftime('%Y%m%d_%H%M%S')
    _write_yaml_files(superdir, sample_dirs, cutoffs, mode, ts_yaml)
    _save_group_histograms(hist_dir, data, groups, mode)
    _write_log(summary_dir, superdir, data, groups, timestamp, mode)
    _write_stats_csv(summary_dir, superdir, data, cutoffs, groups, mode)

    return summary_dir


def _save_group_histograms(hist_dir: Path, data: dict,
                           groups: list, mode: str):
    """
    Saves one histogram PNG per cutoff group into hist_dir.

    Each plot mirrors what was shown in the GUI: overlaid histograms with
    shared bin edges, cutoff lines, and shaded accepted region. Files are
    named group_01.png, group_02.png, etc.

    Args:
        hist_dir: directory to write PNG files into
        data:     mapping of sample name to full (ungated) value array
        groups:   ordered list of (lower, upper, [sample_names])
        mode:     'bm' or 'ifxm'
    """
    cfg = _MODE[mode]
    for i, (lo, hi, cols) in enumerate(groups, 1):
        arrays = [data[col][~np.isnan(data[col])] for col in cols]
        arrays = [a for a in arrays if len(a) > 0]
        if not arrays:
            continue

        all_vals = np.concatenate(arrays)
        shared_bins = cfg['bins'](all_vals)

        fig, ax = plt.subplots(figsize=(14, 5))
        for col, vals in zip(cols, arrays):
            ax.hist(vals, bins=shared_bins, alpha=0.5, edgecolor='black',
                    linewidth=0.3, label=col)
        ax.axvline(lo, color='red', linestyle='--', linewidth=1.2,
                   label=f'lower = {lo:.4g}')
        ax.axvline(hi, color='steelblue', linestyle='--', linewidth=1.2,
                   label=f'upper = {hi:.4g}')
        ax.axvspan(lo, hi, alpha=0.08, color='green')
        ax.set_xscale(cfg['scale'])
        if cfg['xlim']:
            ax.set_xlim(*cfg['xlim'])
        ax.set_xlabel(cfg['xlabel'])
        ax.set_ylabel('count')
        ax.set_title(
            f'Group {i}   lower = {lo:.4g} {cfg["unit"]}   '
            f'upper = {hi:.4g} {cfg["unit"]}', fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
        fig.tight_layout()
        out_path = hist_dir / f'group_{i:02d}.png'
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Written: {out_path}")


def _write_log(summary_dir: Path, superdir: Path,
               data: dict, groups: list, timestamp: str, mode: str):
    """
    Writes a plain-text log file documenting the cutoffs applied and the
    number of values removed per sample.

    Written as UTF-8 to support the arrow character used in per-sample lines.

    Args:
        summary_dir: directory to write the log into
        superdir:    experiment superdir (name used in header)
        data:        mapping of sample name to full (ungated) value array
        groups:      ordered list of (lower, upper, [sample_names])
        timestamp:   YYMMDD.HHMMSS string
        mode:        'bm' or 'ifxm'
    """
    cfg = _MODE[mode]
    log_path = summary_dir / 'cutoff_log.txt'

    ts = timestamp  # YYMMDD.HHMMSS
    run_str = f"20{ts[:2]}-{ts[2:4]}-{ts[4:6]} {ts[7:9]}:{ts[9:11]}:{ts[11:13]}"

    total_before = total_after = 0
    lines = []

    lines.append(f"gate_experiment_subfolder — {cfg['label']} Cutoff Log")
    lines.append("=" * 60)
    lines.append(f"Experiment:  {superdir.name}")
    lines.append(f"Superdir:    {superdir}")
    lines.append(f"Run:         {run_str}")
    lines.append("")
    lines.append("Cutoff groups")
    lines.append("-" * 60)

    for i, (lo, hi, cols) in enumerate(groups, 1):
        lines.append(
            f"Group {i}   lower = {lo:.4g} {cfg['unit']}   "
            f"upper = {hi:.4g} {cfg['unit']}")
        for col in cols:
            n_before = len(data[col])
            n_after = int(np.sum((data[col] >= lo) & (data[col] <= hi)))
            n_removed = n_before - n_after
            pct = 100 * n_removed / n_before if n_before else 0
            total_before += n_before
            total_after += n_after
            lines.append(
                f"  {col:<50s}  {n_before} → {n_after}"
                f"  ({n_removed} removed, {pct:.1f}%)"
            )
        lines.append("")

    total_removed = total_before - total_after
    total_pct = 100 * total_removed / total_before if total_before else 0
    n_cols = sum(len(g[2]) for g in groups)
    lines.append(
        f"Total: {total_before} → {total_after} values retained across "
        f"{n_cols} sample(s)  ({total_removed} removed, {total_pct:.1f}%)"
    )

    log_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f"Written: {log_path}")


def _write_stats_csv(summary_dir: Path, superdir: Path, data: dict,
                     cutoffs: dict, groups: list, mode: str):
    """
    Writes a CSV of descriptive statistics on the gated data, one row per sample.

    Metrics: sample, n, mean, median, mode (midpoint of highest-count bin),
    std, cv_pct (std/mean * 100), lower_cutoff, upper_cutoff.

    Args:
        summary_dir: directory to write the stats CSV into
        superdir:    experiment superdir (name used in filename)
        data:        mapping of sample name to full (ungated) value array
        cutoffs:     mapping of sample name to (lower, upper)
        groups:      ordered list of (lower, upper, [sample_names])
        mode:        'bm' or 'ifxm'
    """
    cfg = _MODE[mode]
    rows = []

    for _, _, cols in groups:
        all_vals = np.concatenate([
            data[col][~np.isnan(data[col])] for col in cols
            if len(data[col][~np.isnan(data[col])]) > 0
        ])
        shared_bins = cfg['bins'](all_vals)

        for col in cols:
            lo_col, hi_col = cutoffs[col]
            raw = data[col]
            gated = raw[(raw >= lo_col) & (raw <= hi_col)]
            gated = gated[~np.isnan(gated)]

            if len(gated) == 0:
                rows.append({
                    'sample': col, 'n': 0,
                    'mean': np.nan, 'median': np.nan, 'mode': np.nan,
                    'std': np.nan, 'cv_pct': np.nan,
                    'lower_cutoff': lo_col, 'upper_cutoff': hi_col,
                })
                continue

            counts, edges = np.histogram(gated, bins=shared_bins)
            peak_idx = counts.argmax()
            mode_val = (edges[peak_idx] + edges[peak_idx + 1]) / 2

            mean = gated.mean()
            std = gated.std()
            rows.append({
                'sample':       col,
                'n':            len(gated),
                'mean':         mean,
                'median':       float(np.median(gated)),
                'mode':         mode_val,
                'std':          std,
                'cv_pct':       100 * std / mean if mean != 0 else np.nan,
                'lower_cutoff': lo_col,
                'upper_cutoff': hi_col,
            })

    stats_path = summary_dir / 'cutoff_stats.csv'
    pd.DataFrame(rows).to_csv(stats_path, index=False)
    print(f"Written: {stats_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    superdir = parse_cli_args()

    root = tk.Tk()
    root.withdraw()
    mode = _ask_data_type(root)

    print(f"Discovering {_MODE[mode]['label']} data in {superdir.name}...")
    if mode == 'bm':
        data = _discover_bm(superdir)
    else:
        data = _discover_ifxm(superdir)

    if not data:
        print(f"No {_MODE[mode]['label']} data found in {superdir}")
        sys.exit(1)

    # sample_dirs: map sample name back to its subdir Path for YAML writing
    sample_dirs = {
        name: superdir / name for name in data
    }

    columns = list(data.keys())
    print(f"Found {len(columns)} sample(s): {', '.join(columns)}")

    root.deiconify()
    MainWindow(root, superdir, sample_dirs, columns, data, mode)
    root.mainloop()


if __name__ == '__main__':
    main()
