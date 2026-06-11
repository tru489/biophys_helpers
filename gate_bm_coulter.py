"""
apply_bm_cutoffs.py

Interactive GUI for applying upper/lower cutoffs to columns of a mass_pg or
single-cell volumes CSV file. Supports both buoyant mass (linear scale) and
Coulter counter volume (log scale) data. The user selects a data type on
launch, groups columns, sets shared cutoffs visually via overlaid histograms,
and repeats until all columns are processed. Outputs are written to a
timestamped directory.

Workflow:
    1. A data-type selection dialog asks whether you are gating Buoyant Mass
       or Coulter Counter Volume data.
    2. A scrollable list of all column names is shown. The user multi-selects
       a group and clicks "Set cutoffs for selection".
    3. A histogram window opens showing all selected columns overlaid with
       shared bin edges. The user clicks once to set a lower cutoff (red dashed
       line) and again to set an upper cutoff (blue dashed line). "Accept"
       records the cutoffs.
    4. Steps 2-3 repeat until all columns are assigned. "Done" then becomes
       available.
    5. On "Done", a timestamped output directory is created containing:
         - <stem>_cutoff.csv        gated data (values outside cutoffs removed)
         - <stem>_cutoff_log.txt    per-column removal statistics
         - <stem>_cutoff_stats.csv  per-column descriptive statistics on gated data
         - histograms/group_NN.png  saved histogram for each group
    6. A "← Back" button on the main gating screen lets you re-pick a different
       CSV file without restarting the script.

Usage:
    python apply_bm_cutoffs.py <csv_file>

    <csv_file>   Path to a CSV file where each column is a dataset (no row
                 index). Typically mass_pg.csv from aggregate_bm_vol_files.py
                 or a sc_volumes CSV from extract_coulter_data.py.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog

from gating.common import (MainWindow, ask_data_type_dialog,
                           save_group_histograms, write_stats_csv, write_log)


# ---------------------------------------------------------------------------
# Mode configuration
# ---------------------------------------------------------------------------

_MODE = {
    'bm': {
        'label':      'Buoyant Mass',
        'unit':       'pg',
        'xlabel':     'Buoyant Mass (pg)',
        'scale':      'linear',
        'bins':       lambda vals: np.linspace(vals.min(), vals.max(), 201),
        'xlim':       None,
        'dir_suffix': '_gated_bm_data',
    },
    'cc': {
        'label':      'Coulter Counter Volume',
        'unit':       'fL',
        'xlabel':     'Total Volume (fL)',
        'scale':      'log',
        'bins':       lambda _: np.logspace(np.log10(20), np.log10(100_000), 201),
        'xlim':       (20, 100_000),
        'dir_suffix': '_gated_cc_data',
    },
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_cli_args() -> Path:
    parser = argparse.ArgumentParser(
        description="Interactively apply cutoffs to columns of a mass_pg or "
                    "single-cell volumes CSV file."
    )
    parser.add_argument('csv_file', type=str, help='Path to the CSV file')
    args = parser.parse_args()
    p = Path(args.csv_file)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    return p


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _write_output(csv_path: Path, columns: list, data: dict,
                  cutoffs: dict, groups: list, mode_cfg: dict) -> Path:
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    suffix = mode_cfg['dir_suffix']
    out_dir = csv_path.parent / f'{timestamp}{suffix}'
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

    hist_dir = out_dir / 'histograms'
    hist_dir.mkdir()
    save_group_histograms(hist_dir, data, groups, mode_cfg)

    ts = timestamp  # YYYYMMDD-HHMMSS
    run_str = (f"{ts[:4]}-{ts[4:6]}-{ts[6:8]} "
               f"{ts[9:11]}:{ts[11:13]}:{ts[13:15]}")
    header_lines = [
        f"apply_bm_cutoffs — {mode_cfg['label']} Cutoff Log",
        "=" * 60,
        f"Input:   {csv_path}",
        f"Output:  {csv_out.name}",
        f"Run:     {run_str}",
    ]
    log_path = out_dir / f'{csv_path.stem}_cutoff_log.txt'
    write_log(log_path, header_lines, data, groups, mode_cfg)

    stats_path = out_dir / f'{csv_path.stem}_cutoff_stats.csv'
    write_stats_csv(stats_path, data, cutoffs, groups, mode_cfg)

    return out_dir


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    initial_path = parse_cli_args()
    state = {'path': initial_path}

    while True:
        restart = [False]
        path = state['path']

        df = pd.read_csv(path)
        columns = list(df.columns)
        data = {col: df[col].dropna().values for col in columns}

        if not columns:
            print("No columns found in file.")
            sys.exit(1)

        root = tk.Tk()
        root.withdraw()
        mode_key = ask_data_type_dialog(root, [
            ('Buoyant Mass', 'bm'),
            ('Coulter Counter Volume', 'cc'),
        ])
        mode_cfg = _MODE[mode_key]

        def on_back(root=root, restart=restart):
            new = filedialog.askopenfilename(
                title="Select CSV file",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            )
            if new:
                state['path'] = Path(new)
                restart[0] = True
                root.destroy()

        def on_finish(cutoffs, groups, path=path, columns=columns,
                      data=data, mode_cfg=mode_cfg):
            return _write_output(path, columns, data, cutoffs, groups, mode_cfg)

        root.deiconify()
        MainWindow(root, columns, data, mode_cfg, on_finish, on_back,
                   context_label=path.name)
        root.mainloop()

        if not restart[0]:
            break


if __name__ == '__main__':
    main()
