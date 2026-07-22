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
             <sample_subdir_name>_<mode>_gate.yaml
         - A summary folder is written into the superdir:
             <YYMMDD.HHMMSS>_<mode>_gating_summary/
               cutoff_log.txt
               cutoff_stats.csv
               histograms/group_NN.png
    7. A "← Back" button undoes the last group of cutoffs, restoring those
       samples to the remaining list. Can be pressed repeatedly.

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

import numpy as np
import pandas as pd
import tkinter as tk
import yaml

from gating.common import (MainWindow, ask_data_type_dialog,
                           save_group_histograms, write_stats_csv, write_log)

from fsutil import is_appledouble


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
                    and not is_appledouble(f)
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
        run_dir = run_dirs[-1]
        stage2 = run_dir / 'stage2_analysis'
        if not stage2.is_dir():
            print(f"  [skip] {sample_dir.name}: no stage2_analysis in {run_dir.name}")
            continue
        csv_found = False
        for f in stage2.iterdir():
            if (f.is_file() and not is_appledouble(f)
                    and f.name.endswith('_ProcessedVolumes.csv')):
                csv_found = True
                df = pd.read_csv(f)
                if 'volume' not in df.columns:
                    print(f"  [skip] {sample_dir.name}: no 'volume' column in {f.name}")
                    break
                vals = df['volume'].dropna().values
                if len(vals) == 0:
                    print(f"  [skip] {sample_dir.name}: 'volume' column is empty/all-NaN in {f.name}")
                    break
                data[sample_dir.name] = vals
                break
        if not csv_found:
            print(f"  [skip] {sample_dir.name}: no _ProcessedVolumes.csv in {run_dir.name}/stage2_analysis/")

    return data


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _write_yaml_files(superdir: Path, sample_dirs: dict,
                      cutoffs: dict, mode_cfg: dict, timestamp: str):
    """
    Creates a gating subdir inside each sample subfolder and writes a YAML
    gate file recording the bounds applied to that sample.

    Subdir:    <sample_subdir>/<YYYYMMDD_HHMMSS>_<mode_tag>/
    File:      <sample_subdir_name>_<mode>_gate.yaml
    Content:
        experiment: <superdir_name>
        data_type:  bm | ifxm_volume
        lower:      <float>
        upper:      <float>
    """
    subdir_name = f"{timestamp}_{mode_cfg['yaml_dir_tag']}"

    for sample, (lo, hi) in cutoffs.items():
        sample_dir = sample_dirs[sample]
        gate_dir = sample_dir / subdir_name
        gate_dir.mkdir(exist_ok=True)
        fname = f"{sample_dir.name}_{mode_cfg['yaml_suffix'].lstrip('_')}"
        out_path = gate_dir / fname
        payload = {
            'experiment': superdir.name,
            'data_type':  mode_cfg['data_type'],
            'lower':      float(lo),
            'upper':      float(hi),
        }
        with open(out_path, 'w') as fh:
            yaml.dump(payload, fh, default_flow_style=False, sort_keys=False)
        print(f"Written: {out_path}")

    print(f"[yaml] {len(cutoffs)} gate file(s) written.")


def _write_output(superdir: Path, sample_dirs: dict, columns: list,
                  data: dict, cutoffs: dict,
                  groups: list, mode_cfg: dict) -> Path:
    timestamp = datetime.now().strftime('%y%m%d.%H%M%S')
    suffix = mode_cfg['dir_suffix']
    summary_dir = superdir / f'{timestamp}{suffix}'
    summary_dir.mkdir()

    hist_dir = summary_dir / 'histograms'
    hist_dir.mkdir()

    ts_yaml = datetime.now().strftime('%Y%m%d_%H%M%S')
    _write_yaml_files(superdir, sample_dirs, cutoffs, mode_cfg, ts_yaml)
    save_group_histograms(hist_dir, data, groups, mode_cfg)

    ts = timestamp  # YYMMDD.HHMMSS
    run_str = f"20{ts[:2]}-{ts[2:4]}-{ts[4:6]} {ts[7:9]}:{ts[9:11]}:{ts[11:13]}"
    header_lines = [
        f"gate_experiment_subfolder — {mode_cfg['label']} Cutoff Log",
        "=" * 60,
        f"Experiment:  {superdir.name}",
        f"Superdir:    {superdir}",
        f"Run:         {run_str}",
    ]
    log_path = summary_dir / 'cutoff_log.txt'
    write_log(log_path, header_lines, data, groups, mode_cfg)

    stats_path = summary_dir / 'cutoff_stats.csv'
    write_stats_csv(stats_path, data, cutoffs, groups, mode_cfg)

    return summary_dir


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    superdir = parse_cli_args()

    root = tk.Tk()
    root.withdraw()
    mode_key = ask_data_type_dialog(root, [
        ('Buoyant Mass', 'bm'),
        ('iFXM Volume', 'ifxm'),
    ])
    mode_cfg = _MODE[mode_key]

    print(f"Discovering {mode_cfg['label']} data in {superdir.name}...")
    if mode_key == 'bm':
        data = _discover_bm(superdir)
    else:
        data = _discover_ifxm(superdir)

    if not data:
        print(f"No {mode_cfg['label']} data found in {superdir}")
        sys.exit(1)

    sample_dirs = {name: superdir / name for name in data}
    columns = list(data.keys())
    print(f"Found {len(columns)} sample(s): {', '.join(columns)}")

    def on_finish(cutoffs, groups):
        return _write_output(superdir, sample_dirs, columns,
                             data, cutoffs, groups, mode_cfg)

    root.deiconify()
    MainWindow(root, columns, data, mode_cfg, on_finish,
               context_label=superdir.name, listbox_width=80)
    root.mainloop()


if __name__ == '__main__':
    main()
