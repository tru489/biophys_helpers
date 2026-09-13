"""
calculate_baseline_density.py

Computes the fluid baseline density for every sample in an experiment superdir
from its buoyant mass (SMR) data, and writes a single timestamped summary CSV.

For each sample subdir the newest ``*_mass_results`` folder is located (the same
discovery convention used by gate_experiments_inplace.py and
aggregate_bm_vol_files.py) and its buoyant mass CSV is read. The mean of the
per-cell ``avg_baseline`` column is taken and converted to a fluid baseline
density using a base-frequency / density calibration:

    baseline_density = (rfreq - mean_avg_baseline - intercept) / slope

where ``slope`` and ``intercept`` come from a calibration JSON (see --calib-json,
required) and ``rfreq`` is the experiment's resonant frequency (Hz), passed on
the command line. This mirrors the per-run calculation in the standalone
baseline_density_calc.py analysis script, generalised to sweep a whole superdir.

Output (written at the same level as the sample subdirs, i.e. inside <superdir>):
    <YYYYMMDD_HHMMSS>_baseline_density.csv
        one row per sample: sample, n_cells, mean_avg_baseline, baseline_density,
        rfreq, slope, intercept, source_csv

Usage:
    python calculate_baseline_density.py <superdir> --rfreq <hz> --calib-json <path>
    python calculate_baseline_density.py                      # opens a GUI form

    <superdir>      Experiment directory whose immediate subdirs are samples,
                    each containing a *_mass_results folder.
    --rfreq         Resonant frequency in Hz.
    --calib-json    Path to a calibration JSON with 'slope' and 'intercept' keys.

If <superdir>, --rfreq and --calib-json are all given on the command line, the
computation runs headlessly (as before) and prints its report to stdout.
Otherwise a small GUI form opens to collect whichever of the three inputs is
still missing.
"""
import argparse
import json
import re
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd

from fsutil import is_appledouble


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-sample fluid baseline density from buoyant "
                    "mass CSVs under an experiment superdir."
    )
    parser.add_argument('superdir', type=str, nargs='?', default=None,
                        help='Experiment directory whose subdirs are samples, '
                             'each with a *_mass_results folder. If omitted '
                             '(along with --rfreq), a GUI form opens instead.')
    parser.add_argument('--rfreq', type=float, default=None,
                        help='Resonant frequency in Hz')
    parser.add_argument('--calib-json', type=str, default=None,
                        help="Path to a calibration JSON with 'slope' and "
                             "'intercept' keys. Required for headless runs; if "
                             "omitted (along with superdir/--rfreq), a GUI form "
                             "opens instead.")
    args = parser.parse_args()

    if args.calib_json is not None and not Path(args.calib_json).is_file():
        raise FileNotFoundError(f"Calibration JSON not found: {args.calib_json}")
    return args


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def load_calibration(path: Path) -> dict:
    """Loads a calibration JSON providing 'slope' and 'intercept'."""
    with open(path) as f:
        cal = json.load(f)
    if 'slope' not in cal or 'intercept' not in cal:
        raise ValueError(f"Calibration JSON {path} must contain 'slope' and "
                         f"'intercept' keys.")
    print(f"Calibration ({cal.get('date', 'no date')}): "
          f"slope={cal['slope']:.6g}  intercept={cal['intercept']:.6g}")
    return cal


def apply_calculation(mean_baseline: float, cal: dict, rfreq: float) -> float:
    """Converts a sample's mean avg_baseline to fluid baseline density."""
    return (rfreq - mean_baseline - cal['intercept']) / cal['slope']


# ---------------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------------

def per_sample_avg_baseline(superdir: Path) -> list:
    """
    Finds the newest *_mass_results buoyant mass CSV for each sample subdir and
    returns [(sample_name, n_cells, mean_avg_baseline, source_csv)] sorted by
    sample name.

    Searches two levels deep (superdir -> sample_subdir -> *_mass_results),
    matching the discovery convention in gate_experiments_inplace.py. If a sample
    has multiple mass_results dirs, the lexicographically last one (newest
    timestamp) is used. Files named curation_index*.csv and AppleDouble sidecars
    are skipped.
    """
    run_dir_pattern = re.compile(r'.+_mass_results$')
    rows = []

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
            if not (f.is_file() and f.suffix == '.csv'
                    and not is_appledouble(f)
                    and not f.name.startswith('curation_index')):
                continue
            df = pd.read_csv(f)
            if 'mass_pg' not in df.columns or 'avg_baseline' not in df.columns:
                continue
            ab = df['avg_baseline'].to_numpy()
            ab = ab[np.isfinite(ab)]
            if ab.size == 0:
                continue
            rows.append((sample_dir.name, int(ab.size), float(ab.mean()),
                         str(f.relative_to(superdir))))
            break

    rows.sort(key=lambda r: r[0])
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_output(superdir: Path, rows: list, cal: dict,
                 rfreq: float) -> Path:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = superdir / f'{timestamp}_baseline_density.csv'

    slope = cal['slope']
    intercept = cal['intercept']

    out_rows = []
    for name, n, mean_b, source in rows:
        out_rows.append({
            'sample':            name,
            'n_cells':           n,
            'mean_avg_baseline': mean_b,
            'baseline_density':  apply_calculation(mean_b, cal, rfreq),
            'rfreq':             rfreq,
            'slope':             slope,
            'intercept':         intercept,
            'source_csv':        source,
        })

    pd.DataFrame(out_rows).to_csv(out_path, index=False)
    print(f"Written: {out_path}  ({len(out_rows)} sample(s))")
    return out_path


def format_report(superdir_name: str, rfreq: float, rows: list,
                  cal: dict) -> str:
    """The sample/n_cells/mean_baseline/baseline_density table as printable text."""
    lines = [f"=== baseline density ({superdir_name}, rfreq={rfreq:g}) ==="]
    lines.append(f"{'sample':<28}{'n_cells':>9}{'mean_baseline':>16}{'baseline_density':>18}")
    lines.append("-" * 71)
    for name, n, mean_b, _ in rows:
        density = apply_calculation(mean_b, cal, rfreq)
        lines.append(f"{name:<28}{n:>9d}{mean_b:>16.6g}{density:>18.6g}")
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Computation (shared by the headless and GUI entry points)
# ---------------------------------------------------------------------------

def run_computation(superdir: Path, rfreq: float,
                    calib_json: Path) -> tuple[list, dict, Path | None]:
    """
    Runs the full compute-and-write pipeline. Returns (rows, cal, out_path);
    out_path (and rows) is empty/None when no buoyant mass CSVs were found
    under superdir.
    """
    cal = load_calibration(calib_json)
    rows = per_sample_avg_baseline(superdir)
    if not rows:
        return [], cal, None
    out_path = write_output(superdir, rows, cal, rfreq)
    return rows, cal, out_path


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

_CHECKED = '☑'    # ☑
_UNCHECKED = '☐'  # ☐


class BaselineDensityGUI(ttk.Frame):
    """
    Simple form: a superdir picker, a calibration-JSON picker, and a
    resonant-frequency entry, with a Run button that computes and writes the
    same CSV as the headless path. Results are shown in a fixed-height,
    scrollable table with a checkbox per sample; Select All / Select None /
    Invert Selection buttons and a live average-density readout let the user
    pick a subset of samples to average.

    A ttk.Frame so it can be packed into any parent widget — a throwaway
    standalone root (see run_gui()) or a page of a larger embedding GUI.
    """

    _COLUMNS = ('check', 'sample', 'n_cells', 'mean_baseline', 'density')

    def __init__(self, parent: tk.Widget, *, superdir: str | None,
                 calib_json: str | None, rfreq: float | None,
                 superdir_var: tk.StringVar | None = None):
        super().__init__(parent)
        root = self

        self._superdir = superdir_var if superdir_var is not None else tk.StringVar(value=superdir or '')
        self._calib_json = tk.StringVar(value=calib_json or '')
        self._rfreq = tk.StringVar(value='' if rfreq is None else f'{rfreq:g}')
        self._checked: dict[str, bool] = {}
        self._densities: dict[str, float] = {}

        pad = {'padx': 8, 'pady': 6}
        form = ttk.Frame(root)
        form.pack(fill=tk.X, **pad)
        form.columnconfigure(1, weight=1)

        ttk.Label(form, text='Superdir:').grid(row=0, column=0, sticky='w')
        ttk.Entry(form, textvariable=self._superdir).grid(row=0, column=1, sticky='ew', padx=(6, 6))
        ttk.Button(form, text='Browse…', command=self._pick_superdir).grid(row=0, column=2)

        ttk.Label(form, text='Calibration JSON:').grid(row=1, column=0, sticky='w', pady=(6, 0))
        ttk.Entry(form, textvariable=self._calib_json).grid(row=1, column=1, sticky='ew', padx=(6, 6), pady=(6, 0))
        ttk.Button(form, text='Browse…', command=self._pick_calib_json).grid(row=1, column=2, pady=(6, 0))

        ttk.Label(form, text='Reference frequency (Hz):').grid(row=2, column=0, sticky='w', pady=(6, 0))
        ttk.Entry(form, textvariable=self._rfreq, width=16).grid(row=2, column=1, sticky='w', padx=(6, 6), pady=(6, 0))

        self._run_button = ttk.Button(form, text='Run', command=self._run)
        self._run_button.grid(row=3, column=1, sticky='w', pady=(10, 0))

        self._message = ttk.Label(root, text='', foreground='#a00')
        self._message.pack(fill=tk.X, padx=8)

        button_bar = ttk.Frame(root)
        button_bar.pack(fill=tk.X, padx=8, pady=(4, 0))
        ttk.Button(button_bar, text='Select All', command=self._select_all).pack(side=tk.LEFT)
        ttk.Button(button_bar, text='Select None', command=self._select_none).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(button_bar, text='Invert Selection', command=self._invert_selection).pack(side=tk.LEFT, padx=(6, 0))

        table_frame = ttk.Frame(root)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(6, 0))

        self._tree = ttk.Treeview(table_frame, columns=self._COLUMNS, show='headings',
                                  height=14, selectmode='none')
        headings = {'check': '', 'sample': 'Sample', 'n_cells': 'n cells',
                    'mean_baseline': 'Mean baseline', 'density': 'Baseline density'}
        widths = {'check': 30, 'sample': 220, 'n_cells': 70,
                 'mean_baseline': 130, 'density': 140}
        for col in self._COLUMNS:
            anchor = 'center' if col == 'check' else 'w'
            self._tree.heading(col, text=headings[col])
            self._tree.column(col, width=widths[col], anchor=anchor,
                              stretch=(col == 'sample'))

        vsb = ttk.Scrollbar(table_frame, orient='vertical', command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        self._tree.bind('<Button-1>', self._on_tree_click)

        self._current_average: float | None = None

        average_bar = ttk.Frame(root)
        average_bar.pack(fill=tk.X, padx=8, pady=8)
        self._average_label = ttk.Label(average_bar, text='Average density of selected samples: n/a',
                                        font=('Segoe UI', 10, 'bold'))
        self._average_label.pack(side=tk.LEFT)
        self._copy_button = ttk.Button(average_bar, text='Copy value', command=self._copy_average,
                                       state=tk.DISABLED)
        self._copy_button.pack(side=tk.LEFT, padx=(10, 0))

    def _pick_superdir(self):
        chosen = filedialog.askdirectory(title='Select experiment superdir',
                                         initialdir=self._superdir.get() or None)
        if chosen:
            self._superdir.set(chosen)

    def _pick_calib_json(self):
        chosen = filedialog.askopenfilename(
            title='Select calibration JSON',
            initialdir=str(Path(self._calib_json.get()).parent) if self._calib_json.get() else None,
            filetypes=[('JSON files', '*.json'), ('All files', '*.*')])
        if chosen:
            self._calib_json.set(chosen)

    # -- table population ---------------------------------------------------

    def _populate_table(self, rows: list, cal: dict, rfreq: float):
        self._tree.delete(*self._tree.get_children())
        self._checked.clear()
        self._densities.clear()

        for i, (name, n, mean_b, _source) in enumerate(rows):
            iid = str(i)
            density = apply_calculation(mean_b, cal, rfreq)
            self._checked[iid] = True
            self._densities[iid] = density
            self._tree.insert('', tk.END, iid=iid, values=(
                _CHECKED, name, n, f'{mean_b:.6g}', f'{density:.6g}',
            ))

        self._update_average()

    def _on_tree_click(self, event):
        if self._tree.identify_region(event.x, event.y) != 'cell':
            return
        if self._tree.identify_column(event.x) != '#1':
            return
        iid = self._tree.identify_row(event.y)
        if not iid:
            return
        self._set_checked(iid, not self._checked[iid])
        self._update_average()

    def _set_checked(self, iid: str, value: bool):
        self._checked[iid] = value
        self._tree.set(iid, 'check', _CHECKED if value else _UNCHECKED)

    def _select_all(self):
        for iid in self._checked:
            self._set_checked(iid, True)
        self._update_average()

    def _select_none(self):
        for iid in self._checked:
            self._set_checked(iid, False)
        self._update_average()

    def _invert_selection(self):
        for iid, value in self._checked.items():
            self._set_checked(iid, not value)
        self._update_average()

    def _update_average(self):
        selected = [self._densities[iid] for iid, checked in self._checked.items() if checked]
        if not selected:
            self._current_average = None
            self._average_label.configure(text='Average density of selected samples: n/a')
            self._copy_button.configure(state=tk.DISABLED)
            return
        avg = sum(selected) / len(selected)
        self._current_average = avg
        self._average_label.configure(
            text=f'Average density of selected samples: {avg:.4f}  (n={len(selected)})')
        self._copy_button.configure(state=tk.NORMAL)

    def _copy_average(self):
        if self._current_average is None:
            return
        self.clipboard_clear()
        self.clipboard_append(f'{self._current_average:.4f}')

    # -- run ------------------------------------------------------------

    def _run(self):
        superdir_text = self._superdir.get().strip()
        if not superdir_text:
            messagebox.showerror('Baseline Density Calculator', 'Choose a superdir first.')
            return
        superdir = Path(superdir_text)
        if not superdir.is_dir():
            messagebox.showerror('Baseline Density Calculator', f'Directory not found:\n{superdir}')
            return

        rfreq_text = self._rfreq.get().strip()
        try:
            rfreq = float(rfreq_text)
        except ValueError:
            messagebox.showerror('Baseline Density Calculator',
                                 f'Reference frequency must be a number, got: {rfreq_text!r}')
            return

        calib_text = self._calib_json.get().strip()
        if not calib_text:
            messagebox.showerror('Baseline Density Calculator', 'Choose a calibration JSON first.')
            return
        calib_json = Path(calib_text)
        if not calib_json.is_file():
            messagebox.showerror('Baseline Density Calculator', f'Calibration JSON not found:\n{calib_json}')
            return

        self._run_button.configure(state=tk.DISABLED)
        try:
            rows, cal, out_path = run_computation(superdir, rfreq, calib_json)
        except Exception as exc:
            messagebox.showerror('Baseline Density Calculator', str(exc))
            return
        finally:
            self._run_button.configure(state=tk.NORMAL)

        if not rows:
            self._message.configure(text=f'No buoyant mass CSVs found under {superdir}.')
            self._tree.delete(*self._tree.get_children())
            self._checked.clear()
            self._densities.clear()
            self._update_average()
            return

        self._message.configure(text='')
        self._populate_table(rows, cal, rfreq)
        messagebox.showinfo('Baseline Density Calculator', f'Written:\n{out_path}')


def run_gui(*, superdir: str | None, calib_json: str | None, rfreq: float | None):
    root = tk.Tk()
    root.title('Baseline Density Calculator')
    root.minsize(640, 480)
    gui = BaselineDensityGUI(root, superdir=superdir, calib_json=calib_json, rfreq=rfreq)
    gui.pack(fill=tk.BOTH, expand=True)
    root.mainloop()


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def build_embedded_page(parent: tk.Widget, *, superdir: str | None = None,
                        calib_json: str | None = None,
                        rfreq: float | None = None,
                        superdir_var: tk.StringVar | None = None) -> ttk.Frame:
    """
    Build this tool's GUI as a Frame suitable for embedding in a larger
    application (e.g. a wizard page), rather than owning its own Tk root.

    Pass an existing `superdir_var` (rather than `superdir`) to bind this
    page's directory field to a StringVar shared with another page — e.g. a
    wizard sharing one directory across several steps.
    """
    return BaselineDensityGUI(parent, superdir=superdir, calib_json=calib_json,
                              rfreq=rfreq, superdir_var=superdir_var)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_headless(args: argparse.Namespace):
    superdir = Path(args.superdir)
    if not superdir.is_dir():
        raise FileNotFoundError(f"Directory not found: {superdir}")

    rows, cal, _ = run_computation(superdir, args.rfreq, Path(args.calib_json))
    if not rows:
        print(f"No buoyant mass CSVs found under {superdir}.")
        return
    print(format_report(superdir.name, args.rfreq, rows, cal))


def main():
    args = parse_cli_args()
    if args.superdir is not None and args.rfreq is not None and args.calib_json is not None:
        run_headless(args)
    else:
        run_gui(superdir=args.superdir, calib_json=args.calib_json, rfreq=args.rfreq)


if __name__ == '__main__':
    main()
