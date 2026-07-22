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

where ``slope`` and ``intercept`` come from a calibration JSON (see --calib-json)
and ``rfreq`` is the experiment's resonant frequency (Hz), passed on the command
line. This mirrors the per-run calculation in the standalone
baseline_density_calc.py analysis script, generalised to sweep a whole superdir.

Without a calibration JSON the mean baseline is still reported but
baseline_density is written as NaN.

Output (written at the same level as the sample subdirs, i.e. inside <superdir>):
    <YYYYMMDD_HHMMSS>_baseline_density.csv
        one row per sample: sample, n_cells, mean_avg_baseline, baseline_density,
        rfreq, slope, intercept, source_csv

Usage:
    python calculate_baseline_density.py <superdir> --rfreq <hz> [--calib-json <path>]

    <superdir>      Experiment directory whose immediate subdirs are samples,
                    each containing a *_mass_results folder.
    --rfreq         Resonant frequency in Hz (required).
    --calib-json    Path to a calibration JSON with 'slope' and 'intercept'
                    keys (optional; baseline_density is NaN if omitted).
"""
import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path

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
    parser.add_argument('superdir', type=str,
                        help='Experiment directory whose subdirs are samples, '
                             'each with a *_mass_results folder')
    parser.add_argument('--rfreq', type=float, required=True,
                        help='Resonant frequency in Hz')
    parser.add_argument('--calib-json', type=str, default=None,
                        help="Path to a calibration JSON with 'slope' and "
                             "'intercept' keys. If omitted, baseline_density is NaN.")
    args = parser.parse_args()

    superdir = Path(args.superdir)
    if not superdir.is_dir():
        raise FileNotFoundError(f"Directory not found: {superdir}")
    if args.calib_json is not None and not Path(args.calib_json).is_file():
        raise FileNotFoundError(f"Calibration JSON not found: {args.calib_json}")
    return args


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def load_calibration(path: Path | None) -> dict | None:
    """
    Loads a calibration JSON providing 'slope' and 'intercept'. Returns None if
    no path is given, so callers can fall back to NaN baseline densities.
    """
    if path is None:
        return None
    with open(path) as f:
        cal = json.load(f)
    if 'slope' not in cal or 'intercept' not in cal:
        raise ValueError(f"Calibration JSON {path} must contain 'slope' and "
                         f"'intercept' keys.")
    print(f"Calibration ({cal.get('date', 'no date')}): "
          f"slope={cal['slope']:.6g}  intercept={cal['intercept']:.6g}")
    return cal


def apply_calculation(mean_baseline: float, cal: dict | None, rfreq: float) -> float:
    """
    Converts a sample's mean avg_baseline to fluid baseline density using the
    calibration slope/intercept and the experiment resonant frequency. Returns
    NaN when no calibration is available.
    """
    if cal is None:
        return float('nan')
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

def write_output(superdir: Path, rows: list, cal: dict | None,
                 rfreq: float) -> Path:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = superdir / f'{timestamp}_baseline_density.csv'

    slope = cal['slope'] if cal else float('nan')
    intercept = cal['intercept'] if cal else float('nan')

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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_cli_args()
    superdir = Path(args.superdir)
    cal = load_calibration(Path(args.calib_json) if args.calib_json else None)

    if cal is None:
        print("!! No calibration JSON given -- baseline_density will be NaN.")

    rows = per_sample_avg_baseline(superdir)
    if not rows:
        print(f"No buoyant mass CSVs found under {superdir}.")
        return

    print(f"\n=== baseline density ({superdir.name}, rfreq={args.rfreq:g}) ===")
    print(f"{'sample':<28}{'n_cells':>9}{'mean_baseline':>16}{'baseline_density':>18}")
    print("-" * 71)
    for name, n, mean_b, _ in rows:
        density = apply_calculation(mean_b, cal, args.rfreq)
        density_str = 'nan' if math.isnan(density) else f'{density:.6g}'
        print(f"{name:<28}{n:>9d}{mean_b:>16.6g}{density_str:>18}")

    write_output(superdir, rows, cal, args.rfreq)


if __name__ == '__main__':
    main()
