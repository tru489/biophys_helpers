"""
gating/common.py

Shared GUI components and output utilities used by gate_bm_coulter.py and
gate_experiments_inplace.py.

Public API
----------
CutoffWindow          Modal histogram window; click-to-set lower/upper cutoffs.
MainWindow            Scrollable sample/column list that drives CutoffWindow per group.
ask_data_type_dialog  Modal mode-selection dialog (parameterised button labels).
save_group_histograms Save one PNG per cutoff group.
write_stats_csv       Write descriptive statistics CSV for all gated samples.
write_log             Write plain-text cutoff log (caller provides header lines).
"""

import sys
import tkinter as tk
from tkinter import messagebox

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# CutoffWindow
# ---------------------------------------------------------------------------

class CutoffWindow:
    """
    Modal tkinter Toplevel containing an embedded matplotlib histogram.

    Displays selected samples/columns as overlaid histograms with shared bin
    edges. The user clicks once to set a lower cutoff (red dashed line) and
    again to set an upper cutoff (blue dashed line). Accept is only enabled
    once both are set.

    Args:
        parent:       parent tkinter window
        selected_cols: names of columns/samples to display
        data:         mapping of name → value array (ungated)
        mode_cfg:     entry from a _MODE dict, e.g. _MODE['bm']

    Result:
        self.result  (lower, upper) tuple, or None if closed without accepting.
    """

    def __init__(self, parent: tk.Tk, selected_cols: list,
                 data: dict, mode_cfg: dict):
        self.result = None
        self._lower = None
        self._upper = None
        self._state = 0         # 0=awaiting lower, 1=awaiting upper, 2=both set
        self._vlines = []
        self._patch = None
        self._cfg = mode_cfg

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
        cfg = self._cfg
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
        self.result = (self._lower, self._upper)
        plt.close(self._fig)
        self._top.destroy()


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------

class MainWindow:
    """
    Primary application window. Shows all sample/column names in a scrollable
    multi-select listbox. The user selects groups, gates via CutoffWindow, and
    repeats until all items are assigned. Done is only enabled once all items
    are processed.

    Args:
        root:          tkinter root window
        columns:       ordered list of all column/sample names
        data:          mapping of name → ungated value array
        mode_cfg:      entry from a _MODE dict
        on_finish:     callable(cutoffs, groups) → Path | str | None
                       Called when Done is clicked; return value shown in dialog.
        context_label: text appended to the header, e.g. filename or superdir name
        listbox_width: character width of the listbox widget (default 120)
    """

    def __init__(self, root: tk.Tk, columns: list, data: dict, mode_cfg: dict,
                 on_finish,
                 context_label: str = '', listbox_width: int = 120):
        self._root = root
        self._columns = columns
        self._data = data
        self._cfg = mode_cfg
        self._on_finish = on_finish
        self._cutoffs: dict = {}
        self._groups: list = []
        self._remaining: list = list(columns)
        self._history: list = []

        label = mode_cfg['label']
        ctx = f"  [{context_label}]" if context_label else ''
        root.title(f"Gating — {label}{ctx}")

        self._header_var = tk.StringVar()
        tk.Label(root, textvariable=self._header_var,
                 font=('TkDefaultFont', 11, 'bold'),
                 anchor='w').pack(fill=tk.X, padx=10, pady=(10, 4))

        list_frame = tk.Frame(root)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10)
        scrollbar = tk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self._listbox = tk.Listbox(
            list_frame, selectmode=tk.MULTIPLE, exportselection=False,
            yscrollcommand=scrollbar.set, height=24, width=listbox_width)
        scrollbar.config(command=self._listbox.yview)
        self._listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._listbox.bind('<<ListboxSelect>>', self._on_select)

        btn_frame = tk.Frame(root)
        btn_frame.pack(fill=tk.X, padx=10, pady=8)

        self._back_btn = tk.Button(btn_frame, text="← Back",
                                   state=tk.DISABLED, command=self._do_back)
        self._back_btn.pack(side=tk.LEFT, padx=(0, 16))

        self._set_btn = tk.Button(
            btn_frame, text="Set cutoffs for selection",
            state=tk.DISABLED, command=self._open_cutoff_window)
        self._set_btn.pack(side=tk.LEFT, padx=(0, 8))
        self._done_btn = tk.Button(
            btn_frame, text="Done", state=tk.DISABLED, command=self._finish)
        self._done_btn.pack(side=tk.RIGHT)

        self._refresh_list()

    def _refresh_list(self):
        self._listbox.delete(0, tk.END)
        for col in self._remaining:
            self._listbox.insert(tk.END, col)
        n_done = len(self._cutoffs)
        n_total = len(self._columns)
        self._header_var.set(
            f"Remaining: {n_total - n_done} / {n_total}   "
            f"[{self._cfg['label']}]")
        self._done_btn.config(
            state=tk.NORMAL if n_done == n_total else tk.DISABLED)
        self._set_btn.config(state=tk.DISABLED)
        self._back_btn.config(
            state=tk.NORMAL if self._history else tk.DISABLED)

    def _on_select(self, _event):
        has_selection = bool(self._listbox.curselection())
        self._set_btn.config(state=tk.NORMAL if has_selection else tk.DISABLED)

    def _open_cutoff_window(self):
        indices = self._listbox.curselection()
        if not indices:
            return
        selected = [self._remaining[i] for i in indices]
        win = CutoffWindow(self._root, selected, self._data, self._cfg)
        if win.result is None:
            return
        self._history.append((
            self._cutoffs.copy(),
            list(self._groups),
            list(self._remaining),
        ))
        lower, upper = win.result
        for col in selected:
            self._cutoffs[col] = (lower, upper)
            self._remaining.remove(col)
        self._groups.append((lower, upper, selected))
        self._refresh_list()

    def _finish(self):
        result = self._on_finish(self._cutoffs, self._groups)
        msg = f"Output written to:\n{result}" if result else "Done."
        messagebox.showinfo("Done", msg)
        self._root.destroy()

    def _do_back(self):
        if not self._history:
            return
        self._cutoffs, self._groups, self._remaining = self._history.pop()
        self._refresh_list()


# ---------------------------------------------------------------------------
# Data type selection dialog
# ---------------------------------------------------------------------------

def ask_data_type_dialog(root: tk.Tk, options: list) -> str:
    """
    Show a modal dialog letting the user choose a data type.

    Args:
        root:    parent tkinter window
        options: list of (button_label, mode_key) tuples, e.g.
                 [('Buoyant Mass', 'bm'), ('iFXM Volume', 'ifxm')]

    Returns:
        str: the selected mode_key; calls sys.exit(0) if the window is closed.
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

    for label, key in options:
        tk.Button(btn_frame, text=label, width=22,
                  command=lambda k=key: (result.__setitem__('mode', k),
                                        top.destroy())).pack(side=tk.LEFT, padx=8)

    top.protocol('WM_DELETE_WINDOW', lambda: sys.exit(0))
    root.wait_window(top)

    if result['mode'] is None:
        sys.exit(0)
    return result['mode']


# ---------------------------------------------------------------------------
# Shared output functions
# ---------------------------------------------------------------------------

def save_group_histograms(hist_dir: Path, data: dict,
                          groups: list, mode_cfg: dict):
    """
    Save one histogram PNG per cutoff group into hist_dir.

    Each plot mirrors the GUI view: overlaid histograms with shared bin edges,
    cutoff lines, and shaded accepted region. Files are named group_01.png, etc.

    Args:
        hist_dir: directory to write PNG files into
        data:     mapping of name → ungated value array
        groups:   ordered list of (lower, upper, [names])
        mode_cfg: entry from a _MODE dict
    """
    cfg = mode_cfg
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


def write_stats_csv(stats_path: Path, data: dict, cutoffs: dict,
                    groups: list, mode_cfg: dict):
    """
    Write a CSV of descriptive statistics for gated data, one row per sample.

    Metrics: sample, n, mean, median, mode (midpoint of peak histogram bin),
    std, cv_pct, lower_cutoff, upper_cutoff.

    Args:
        stats_path: full path of the CSV file to write
        data:       mapping of name → ungated value array
        cutoffs:    mapping of name → (lower, upper)
        groups:     ordered list of (lower, upper, [names])
        mode_cfg:   entry from a _MODE dict
    """
    cfg = mode_cfg
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
                rows.append({'sample': col, 'n': 0,
                             'mean': np.nan, 'median': np.nan, 'mode': np.nan,
                             'std': np.nan, 'cv_pct': np.nan,
                             'lower_cutoff': lo_col, 'upper_cutoff': hi_col})
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

    pd.DataFrame(rows).to_csv(stats_path, index=False)
    print(f"Written: {stats_path}")


def write_log(log_path: Path, header_lines: list, data: dict,
              groups: list, mode_cfg: dict):
    """
    Write a plain-text cutoff log documenting applied bounds and per-column
    removal statistics.

    The caller is responsible for providing header_lines (script-specific
    context: input/output paths, run timestamp, etc.). The shared body covers
    per-group cutoff entries and an overall summary.

    Written as UTF-8 to support the → character in per-sample lines.

    Args:
        log_path:     full path of the log file to write
        header_lines: list of strings written verbatim before the groups section
        data:         mapping of name → ungated value array
        groups:       ordered list of (lower, upper, [names])
        mode_cfg:     entry from a _MODE dict
    """
    cfg = mode_cfg
    total_before = total_after = 0
    lines = list(header_lines)
    lines.append("")
    lines.append("Cutoff groups")
    lines.append("-" * 60)

    for i, (lo, hi, cols) in enumerate(groups, 1):
        lines.append(
            f"Group {i}   lower = {lo:.4g} {cfg['unit']}   "
            f"upper = {hi:.4g} {cfg['unit']}")
        for col in cols:
            n_before = len(data[col][~np.isnan(data[col])])
            raw = data[col]
            n_after = len(raw[(raw >= lo) & (raw <= hi) & ~np.isnan(raw)])
            n_removed = n_before - n_after
            pct = 100 * n_removed / n_before if n_before else 0.0
            total_before += n_before
            total_after += n_after
            lines.append(
                f"  {col:<60s}  {n_before} → {n_after}"
                f"  ({n_removed} removed, {pct:.1f}%)"
            )
        lines.append("")

    total_removed = total_before - total_after
    total_pct = 100 * total_removed / total_before if total_before else 0.0
    n_items = sum(len(g[2]) for g in groups)
    lines.append(
        f"Total: {total_before} → {total_after} values retained across "
        f"{n_items} item(s)  ({total_removed} removed, {total_pct:.1f}%)"
    )

    log_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f"Written: {log_path}")
