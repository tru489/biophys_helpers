"""
apply_bm_cutoffs.py

Interactive GUI for applying upper/lower cutoffs to columns of a mass_pg CSV file
(typically produced by aggregate_bm_vol_files.py). The user groups columns, sets
shared cutoffs visually via overlaid linear-scale histograms, and repeats until all
columns are processed. Outputs are written to a timestamped directory.

Workflow:
    1. A scrollable list of all column names is shown. The user multi-selects a group
       and clicks "Set cutoffs for selection".
    2. A histogram window opens showing all selected columns overlaid with shared bin
       edges. The user clicks once to set a lower cutoff (red dashed line) and again
       to set an upper cutoff (blue dashed line). "Accept" records the cutoffs.
    3. Steps 1-2 repeat until all columns are assigned. "Done" then becomes available.
    4. On "Done", a timestamped output directory is created containing:
         - <stem>_cutoff.csv        gated data (values outside cutoffs removed)
         - <stem>_cutoff_log.txt    per-column removal statistics
         - histograms/group_NN.png  saved histogram for each group

Usage:
    python apply_bm_cutoffs.py <mass_pg.csv>

    <mass_pg.csv>   Path to a CSV file where each column is a dataset (no row index).
                    Typically produced by aggregate_bm_vol_files.py.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_cli_args() -> Path:
    """
    Parses CLI args, returning the path to the input CSV file.

    Raises:
        FileNotFoundError: file does not exist

    Returns:
        Path: path to the CSV file to process
    """
    parser = argparse.ArgumentParser(
        description="Interactively apply cutoffs to columns of a mass_pg CSV file."
    )
    parser.add_argument('csv_file', type=str, help='Path to the CSV file')
    args = parser.parse_args()
    p = Path(args.csv_file)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    return p


# ---------------------------------------------------------------------------
# Cutoff window
# ---------------------------------------------------------------------------

class CutoffWindow:
    """
    Modal tkinter Toplevel containing an embedded matplotlib histogram.

    Displays all selected columns as overlaid histograms with shared linear bin
    edges. The user clicks once to set a lower cutoff (red dashed line) and again
    to set an upper cutoff (blue dashed line), with the accepted region shaded green.
    Reset clears both lines and restarts. Accept is only enabled once both cutoffs
    are set.

    Result stored in self.result as (lower, upper), or None if the window is
    closed without accepting.
    """

    def __init__(self, parent: tk.Tk, selected_cols: list[str],
                 data: dict[str, np.ndarray]):
        self.result: tuple[float, float] | None = None
        self._lower: float | None = None
        self._upper: float | None = None
        self._state: int = 0        # 0=awaiting lower, 1=awaiting upper, 2=both set
        self._vlines: list = []
        self._patch = None

        self._top = tk.Toplevel(parent)
        self._top.title("Set cutoffs")
        self._top.grab_set()

        # --- matplotlib figure ---
        self._fig, self._ax = plt.subplots(figsize=(14, 5))
        self._draw_histograms(selected_cols, data)

        canvas = FigureCanvasTkAgg(self._fig, master=self._top)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._canvas = canvas
        self._cid = canvas.mpl_connect('button_press_event', self._on_click)

        # --- status / label row ---
        info_frame = tk.Frame(self._top)
        info_frame.pack(fill=tk.X, padx=10, pady=(4, 0))
        self._status_var = tk.StringVar(value="Click to set lower cutoff.")
        tk.Label(info_frame, textvariable=self._status_var,
                 anchor='w').pack(side=tk.LEFT)

        # --- button row ---
        btn_frame = tk.Frame(self._top)
        btn_frame.pack(fill=tk.X, padx=10, pady=6)
        tk.Button(btn_frame, text="Reset",
                  command=self._reset).pack(side=tk.LEFT, padx=(0, 8))
        self._accept_btn = tk.Button(btn_frame, text="Accept",
                                     state=tk.DISABLED, command=self._accept)
        self._accept_btn.pack(side=tk.RIGHT)

        parent.wait_window(self._top)   # block until window closed

    # ------------------------------------------------------------------
    def _draw_histograms(self, selected_cols: list[str],
                         data: dict[str, np.ndarray]):
        """
        Renders overlaid histograms for all selected columns onto self._ax.
        Bin edges are shared across all columns (computed from the combined
        range) so bars align for easy visual comparison.

        Args:
            selected_cols: column names to plot
            data: mapping of column name to value array
        """
        ax = self._ax
        ax.clear()

        arrays = [data[col][~np.isnan(data[col])] for col in selected_cols]
        arrays = [a for a in arrays if len(a) > 0]
        if not arrays:
            return

        # Shared linear bin edges so bars align across all columns
        all_vals = np.concatenate(arrays)
        shared_bins = np.linspace(all_vals.min(), all_vals.max(), 201)  # 200 bins

        for col, vals in zip(selected_cols, arrays):
            ax.hist(vals, bins=shared_bins, alpha=0.5, edgecolor='black',
                    linewidth=0.3, label=col)
        ax.set_xlabel('mass (pg)')
        ax.set_ylabel('count')
        ax.legend(fontsize=7, loc='upper right')
        self._fig.tight_layout()

    def _on_click(self, event: object):
        """
        Handles matplotlib mouse click events.
        State 0 → first click sets lower cutoff (red dashed line).
        State 1 → second click sets upper cutoff (blue dashed line) and shades
                  the accepted region. Accept button is then enabled.
        Clicks outside the axes or with upper ≤ lower are ignored/warned.
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
    Primary application window. Shows all column names from the input CSV in a
    scrollable multi-select listbox. The user repeatedly selects groups of columns,
    sets cutoffs via CutoffWindow, and repeats until all columns are assigned.
    "Done" is only enabled once every column has been processed, at which point
    output files are written and the application exits.
    """

    def __init__(self, root: tk.Tk, csv_path: Path,
                 columns: list[str], data: dict[str, np.ndarray]):
        self._root = root
        self._csv_path = csv_path
        self._columns = columns          # all column names
        self._data = data
        self._cutoffs: dict[str, tuple[float, float]] = {}
        self._groups: list[tuple[float, float, list[str]]] = []
        self._remaining: list[str] = list(columns)

        root.title("apply_bm_cutoffs")

        self._header_var = tk.StringVar()
        tk.Label(root, textvariable=self._header_var,
                 font=('TkDefaultFont', 11, 'bold'),
                 anchor='w').pack(fill=tk.X, padx=10, pady=(10, 4))

        # listbox + scrollbar
        list_frame = tk.Frame(root)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10)
        scrollbar = tk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self._listbox = tk.Listbox(
            list_frame, selectmode=tk.MULTIPLE, exportselection=False,
            yscrollcommand=scrollbar.set, height=24, width=120)
        scrollbar.config(command=self._listbox.yview)
        self._listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._listbox.bind('<<ListboxSelect>>', self._on_select)

        # buttons
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
        """Rebuilds the listbox from remaining unprocessed columns and updates the header."""
        self._listbox.delete(0, tk.END)
        for col in self._remaining:
            self._listbox.insert(tk.END, col)
        n_done = len(self._cutoffs)
        n_total = len(self._columns)
        self._header_var.set(
            f"Columns remaining: {n_total - n_done} / {n_total}")
        self._done_btn.config(
            state=tk.NORMAL if n_done == n_total else tk.DISABLED)
        self._set_btn.config(state=tk.DISABLED)

    def _on_select(self, _event):
        """Enables/disables "Set cutoffs" button based on whether anything is selected."""
        has_selection = bool(self._listbox.curselection())
        self._set_btn.config(state=tk.NORMAL if has_selection else tk.DISABLED)

    def _open_cutoff_window(self):
        """
        Opens a CutoffWindow for the currently selected columns. On acceptance,
        records the cutoffs, appends to _groups, removes processed columns from
        the listbox, and refreshes the header counter.
        """
        indices = self._listbox.curselection()
        if not indices:
            return
        selected = [self._remaining[i] for i in indices]
        win = CutoffWindow(self._root, selected, self._data)
        if win.result is None:
            return
        lower, upper = win.result
        for col in selected:
            self._cutoffs[col] = (lower, upper)
            self._remaining.remove(col)
        self._groups.append((lower, upper, selected))
        self._refresh_list()

    def _finish(self):
        """Writes all output files and closes the application."""
        out_dir = _write_output(self._csv_path, self._columns, self._data,
                                self._cutoffs, self._groups)
        messagebox.showinfo("Done", f"Output written to:\n{out_dir}")
        self._root.destroy()


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _write_output(csv_path: Path, columns: list[str],
                  data: dict[str, np.ndarray],
                  cutoffs: dict[str, tuple[float, float]],
                  groups: list[tuple[float, float, list[str]]]) -> Path:
    """
    Creates a timestamped output directory and writes all output files.

    Output structure:
        <YYYYMMDD-HHMMSS>_gated_bm_data/
          <stem>_cutoff.csv       gated data; values outside cutoffs removed,
                                  columns NaN-padded to equal length
          <stem>_cutoff_log.txt   per-column removal statistics
          histograms/
            group_01.png          overlaid histogram for each cutoff group
            group_02.png
            ...

    Args:
        csv_path: path to the original input CSV
        columns:  all column names in original order
        data:     mapping of column name to full (ungated) value array
        cutoffs:  mapping of column name to (lower, upper) cutoff pair
        groups:   ordered list of (lower, upper, [col_names]) as set by the user

    Returns:
        Path: the created output directory
    """
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    out_dir = csv_path.parent / f'{timestamp}_gated_bm_data'
    out_dir.mkdir()

    # Gated CSV
    series_list = []
    for col in columns:
        lo, hi = cutoffs[col]
        vals = data[col]
        vals = vals[(vals >= lo) & (vals <= hi)]
        series_list.append(pd.Series(vals, name=col))
    combined = pd.concat(series_list, axis=1)
    csv_out = out_dir / f'{csv_path.stem}_cutoff.csv'
    combined.to_csv(csv_out, index=False)
    print(f"Written: {csv_out}")

    # Histograms
    hist_dir = out_dir / 'histograms'
    hist_dir.mkdir()
    _save_group_histograms(hist_dir, data, groups)

    # Log
    _write_log(out_dir, csv_path, csv_out, data, groups, timestamp)

    return out_dir


def _save_group_histograms(hist_dir: Path, data: dict[str, np.ndarray],
                           groups: list[tuple[float, float, list[str]]]):
    """
    Saves one histogram PNG per cutoff group into hist_dir.

    Each plot mirrors what was shown in the GUI: overlaid histograms with shared
    bin edges, plus the lower/upper cutoff lines and shaded accepted region.
    Files are named group_01.png, group_02.png, etc.

    Args:
        hist_dir: directory to write PNG files into
        data:     mapping of column name to full (ungated) value array
        groups:   ordered list of (lower, upper, [col_names])
    """
    for i, (lo, hi, cols) in enumerate(groups, 1):
        arrays = [data[col][~np.isnan(data[col])] for col in cols]
        arrays = [a for a in arrays if len(a) > 0]
        if not arrays:
            continue

        all_vals = np.concatenate(arrays)
        shared_bins = np.linspace(all_vals.min(), all_vals.max(), 201)

        fig, ax = plt.subplots(figsize=(14, 5))
        for col, vals in zip(cols, arrays):
            ax.hist(vals, bins=shared_bins, alpha=0.5, edgecolor='black',
                    linewidth=0.3, label=col)
        ax.axvline(lo, color='red', linestyle='--', linewidth=1.2,
                   label=f'lower = {lo:.4g}')
        ax.axvline(hi, color='steelblue', linestyle='--', linewidth=1.2,
                   label=f'upper = {hi:.4g}')
        ax.axvspan(lo, hi, alpha=0.08, color='green')
        ax.set_xlabel('mass (pg)')
        ax.set_ylabel('count')
        ax.set_title(f'Group {i}   lower = {lo:.4g} pg   upper = {hi:.4g} pg',
                     fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
        fig.tight_layout()
        fig.savefig(hist_dir / f'group_{i:02d}.png', dpi=150)
        plt.close(fig)
        print(f"Written: {hist_dir / f'group_{i:02d}.png'}")


def _write_log(out_dir: Path, csv_path: Path, csv_out: Path,
               data: dict[str, np.ndarray],
               groups: list[tuple[float, float, list[str]]],
               timestamp: str):
    """
    Writes a plain-text log file documenting the cutoffs applied and the number
    of values removed per column.

    Log contents:
        - Input/output file paths and run timestamp
        - Per-group cutoff bounds
        - Per-column: original N, post-cutoff N, count and % removed
        - Total summary across all columns

    Written as UTF-8 to support the → character used in per-column lines.

    Args:
        out_dir:   output directory to write the log into
        csv_path:  path to the original input CSV
        csv_out:   path to the written gated CSV
        data:      mapping of column name to full (ungated) value array
        groups:    ordered list of (lower, upper, [col_names])
        timestamp: YYYYMMDD-HHMMSS string used to format the run time
    """
    log_path = out_dir / f'{csv_path.stem}_cutoff_log.txt'

    total_before = total_after = 0
    lines = []

    lines.append("apply_bm_cutoffs — Cutoff Log")
    lines.append("=" * 60)
    lines.append(f"Input:   {csv_path}")
    lines.append(f"Output:  {csv_out.name}")
    lines.append(f"Run:     {timestamp[:8][:4]}-{timestamp[4:6]}-{timestamp[6:8]}"
                 f" {timestamp[9:11]}:{timestamp[11:13]}:{timestamp[13:15]}")
    lines.append("")
    lines.append("Cutoff groups")
    lines.append("-" * 60)

    for i, (lo, hi, cols) in enumerate(groups, 1):
        lines.append(f"Group {i}   lower = {lo:.4g} pg   upper = {hi:.4g} pg")
        for col in cols:
            n_before = len(data[col])
            n_after = len(data[col][(data[col] >= lo) & (data[col] <= hi)])
            n_removed = n_before - n_after
            pct = 100 * n_removed / n_before if n_before else 0
            total_before += n_before
            total_after += n_after
            lines.append(
                f"  {col:<60s}  {n_before} → {n_after}"
                f"  ({n_removed} removed, {pct:.1f}%)"
            )
        lines.append("")

    total_removed = total_before - total_after
    total_pct = 100 * total_removed / total_before if total_before else 0
    n_cols = sum(len(g[2]) for g in groups)
    lines.append(
        f"Total: {total_before} → {total_after} values retained across "
        f"{n_cols} column(s)  ({total_removed} removed, {total_pct:.1f}%)"
    )

    log_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f"Written: {log_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    csv_path = parse_cli_args()
    df = pd.read_csv(csv_path)
    columns = list(df.columns)
    data = {col: df[col].dropna().values for col in columns}

    if not columns:
        print("No columns found in file.")
        sys.exit(1)

    root = tk.Tk()
    MainWindow(root, csv_path, columns, data)
    root.mainloop()


if __name__ == '__main__':
    main()
