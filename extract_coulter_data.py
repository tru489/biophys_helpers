"""
extract_coulter_data.py

Parses a directory of Coulter counter (.#m4) files and exports CSV files
containing single-cell volume measurements and/or summary statistics
pre-selected in the Multisizer software.

Usage:
    python extract_coulter_data.py <directory> [-stats] [-single] [-r]

    <directory>   Path to folder containing .#m4 files
    -stats        Write only <dirname>_volume_stats.csv
    -single       Write only <dirname>_single_cell_volumes.csv
    (no flags)    Write both output files
    -r            Recursively include .#m4 files from subdirectories
"""
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
from CoulterFile import CoulterFile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    """
    Write 2 CSVs from a directory containing coulter counter files, 1 with
    summary stats based on preselected data from multisizer software, and 1
    with full single-cell volume data
    """
    # Get directory path
    dir_path, run_stats, run_sc, recursive = parse_cli_args()
    dp_obj = Path(dir_path)
    dirname = dp_obj.name

    full_fpaths, display_names = _collect_files(dp_obj, recursive)

    all_stems, vol_list, stats_stems, gated_vol_list, stats_list, skipped = \
        _parse_coulter_files(full_fpaths, display_names)

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    out_dir = dp_obj / f'{timestamp}_coulter-processed'
    out_dir.mkdir()

    if run_sc:
        _build_sc_df(all_stems, vol_list).to_csv(
            out_dir / f'{dirname}_sc_volumes_ungated.csv', index=False)
        if stats_stems:
            gated_csv = out_dir / f'{dirname}_sc_volumes_gated.csv'
            _build_sc_df(stats_stems, gated_vol_list).to_csv(gated_csv, index=False)
            _write_gating_log(out_dir, dp_obj, dirname, stats_stems, gated_vol_list,
                              stats_list, all_stems, vol_list, skipped, timestamp)

        _plot_sc_histograms(stats_stems, stats_list, all_stems, vol_list,
                            skipped, out_dir)

    if run_stats and stats_stems:
        _build_stats_df(stats_stems, stats_list).to_csv(
            out_dir / f'{dirname}_volume_stats.csv')

    if skipped:
        print(f"\nWarning: {len(skipped)} file(s) missing [SizeStats] — "
              f"excluded from gated and stats outputs:")
        for name in skipped:
            print(f"  {name}")

def parse_cli_args():
    """
    Parse CLI arguments. Takes path to coulter counter directory as CLI argument
    Arguments:
        None: saves stats and single cell data files
        -stats: only saves stats file
        -single: only saves single cell data

    Raises:
        FileNotFoundError: directory does not exist

    Returns:
        tuple(str, bool, bool): directory, whether just stats are requested, 
            whether just single cell volumes are requested
    """
    parser = argparse.ArgumentParser(description="Process a directory path.")
    parser.add_argument('directory', type=str, help='Path to the directory')
    parser.add_argument('-stats', action='store_true', help='Include stats')
    parser.add_argument('-single', action='store_true', help='Single mode')
    parser.add_argument('-r', action='store_true', help='Recursively include subdirectories')

    args = parser.parse_args()

    if not Path(args.directory).is_dir:
        raise FileNotFoundError(f"The directory '{args.directory}' does not exist.")

    if not args.stats and not args.single:
        args.stats = True
        args.single = True

    return args.directory, args.stats, args.single, args.r

def _collect_files(dp_obj: Path, recursive: bool) -> tuple:
    """
    Collect .#m4 files from dp_obj. If recursive, also descends into subdirectories,
    prefixing each file stem with underscore-joined subdir path components.

    Args:
        dp_obj (Path): root directory
        recursive (bool): whether to recurse into subdirectories

    Returns:
        tuple(list(Path), list(str)): full file paths and display names
    """
    is_m4 = lambda p: not p.name.startswith('.') and '.#m4' in p.name

    full_fpaths = []
    display_names = []

    # Files directly in the root dir (no prefix)
    root_files = sorted([e for e in dp_obj.iterdir() if e.is_file() and is_m4(e)],
                        key=lambda p: p.name)
    for f in root_files:
        full_fpaths.append(f)
        display_names.append(f.stem)

    if recursive:
        subdirs = sorted([e for e in dp_obj.iterdir() if e.is_dir() and not e.name.startswith('.')])
        for subdir in subdirs:
            for f in sorted(subdir.rglob('*') if True else [], key=lambda p: p):
                if f.is_file() and is_m4(f):
                    rel = f.relative_to(dp_obj)
                    prefix = '_'.join(rel.parts[:-1])
                    display_names.append(f'{prefix}_{f.stem}')
                    full_fpaths.append(f)

    return full_fpaths, display_names


def _parse_coulter_files(full_fpaths, display_names=None) -> tuple:
    """
    Opens each .#m4 file once and extracts both single-cell volumes and stats.

    Args:
        full_fpaths (list(Path)): list of file paths to be parsed

    Returns:
        tuple(list(str), list(np.array), list(dict)): file stems, volumes per
            file, stats per file
    """
    if display_names is None:
        display_names = [Path(fn).stem for fn in full_fpaths]
    all_stems, vol_list = [], []
    stats_stems, gated_vol_list, stats_list = [], [], []
    skipped = []
    n = len(full_fpaths)
    for i, (fn, name) in enumerate(zip(full_fpaths, display_names), 1):
        print(f"Parsing file {i}/{n}: {fn.name}")
        coulter_file = CoulterFile(fn.resolve())
        all_stems.append(name)
        vol_list.append(coulter_file.get_volumes_ungated())
        if coulter_file.get_stats() is not None:
            stats_stems.append(name)
            gated_vol_list.append(coulter_file.get_volumes_gated())
            stats_list.append(coulter_file.get_stats())
        else:
            skipped.append(name)
    return all_stems, vol_list, stats_stems, gated_vol_list, stats_list, skipped

def _build_sc_df(file_stems, vol_list) -> pd.DataFrame:
    max_length = max(len(arr) for arr in vol_list)
    padded_arrays = []
    for arr in vol_list:
        padded_arr = np.full(max_length, np.nan)
        padded_arr[:len(arr)] = arr
        padded_arrays.append(padded_arr)
    return pd.DataFrame(np.column_stack(padded_arrays), columns=file_stems)

def _build_stats_df(file_stems, stats_list) -> pd.DataFrame:
    keys_ = stats_list[0].keys()
    full_dict = {fs: [d[k] for k in keys_] for fs, d in zip(file_stems, stats_list)}
    df = pd.DataFrame.from_dict(full_dict)
    df.index = keys_
    return df

def _plot_sc_histograms(stats_stems, stats_list, all_stems, vol_list,
                        skipped, out_dir: Path):
    """
    Saves histogram PNGs into out_dir/fig/.

    Files sharing the same (MinSize, MaxSize) gate are overlaid on one plot
    (group_01.png, group_02.png, …), with vertical cutoff lines and shaded
    accepted region. Files missing [SizeStats] are plotted individually.
    """
    from collections import defaultdict

    fig_dir = out_dir / 'fig'
    fig_dir.mkdir(exist_ok=True)

    ungated = {stem: vols for stem, vols in zip(all_stems, vol_list)}

    # Build groups: (lo, hi) -> [(stem, vols), ...]
    groups = defaultdict(list)
    for stem, stats in zip(stats_stems, stats_list):
        lo = float(stats['MinSize'])
        hi = float(stats['MaxSize'])
        groups[(lo, hi)].append((stem, ungated[stem]))

    log_bins = np.logspace(np.log10(20), np.log10(100_000), 201)

    n_written = 0
    for i, ((lo, hi), entries) in enumerate(groups.items(), 1):
        arrays = [vols[~np.isnan(vols)] for _, vols in entries]
        arrays = [a for a in arrays if len(a) > 0]
        if not arrays:
            continue

        fig, ax = plt.subplots(figsize=(14, 5))
        for (stem, _), vals in zip(entries, arrays):
            ax.hist(vals, bins=log_bins, alpha=0.5, edgecolor='black',
                    linewidth=0.3, label=stem)
        ax.axvline(lo, color='red', linestyle='--', linewidth=1.2,
                   label=f'lower = {lo:.4g}')
        ax.axvline(hi, color='steelblue', linestyle='--', linewidth=1.2,
                   label=f'upper = {hi:.4g}')
        ax.axvspan(lo, hi, alpha=0.08, color='green')
        ax.set_xscale('log')
        ax.set_xlim(20, 100_000)
        ax.set_xlabel('volume (fL)')
        ax.set_ylabel('count')
        ax.set_title(f'Group {i}   lower = {lo:.4g} fL   upper = {hi:.4g} fL',
                     fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
        fig.tight_layout()
        out_path = fig_dir / f'group_{i:02d}.png'
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Written: {out_path}")
        n_written += 1

    # Individual plots for files without gate stats
    for stem in skipped:
        vols = ungated.get(stem)
        if vols is None or len(vols) == 0:
            continue
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.hist(vols[~np.isnan(vols)], bins=log_bins, edgecolor='black',
                linewidth=0.3)
        ax.set_xscale('log')
        ax.set_xlim(20, 100_000)
        ax.set_title(stem, fontsize=8, wrap=True)
        ax.set_xlabel('volume (fL)')
        ax.set_ylabel('count')
        fig.tight_layout()
        safe_name = stem.replace('/', '_').replace(' ', '_')
        out_path = fig_dir / f'{safe_name}.png'
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Written: {out_path}")
        n_written += 1

    print(f"[histograms] {n_written} histogram(s) written to {fig_dir}")


def _write_gating_log(out_dir, dp_obj, dirname, stats_stems, gated_vol_list,
                      stats_list, all_stems, vol_list, skipped, timestamp):
    """Write a plain-text log of gating cutoffs and removal statistics."""
    from collections import defaultdict

    log_path = out_dir / f'{dirname}_sc_volumes_gated_log.txt'
    ungated_counts = {stem: len(vols) for stem, vols in zip(all_stems, vol_list)}

    # Group files by (MinSize, MaxSize)
    groups = defaultdict(list)
    for stem, gated, stats in zip(stats_stems, gated_vol_list, stats_list):
        lo = float(stats['MinSize'])
        hi = float(stats['MaxSize'])
        n_before = ungated_counts.get(stem, len(gated))
        n_after = len(gated)
        groups[(lo, hi)].append((stem, n_before, n_after))

    total_before = total_after = 0
    lines = []

    lines.append("extract_coulter_data — Gating Log")
    lines.append("=" * 60)
    lines.append(f"Input:   {dp_obj}")
    lines.append(f"Output:  {dirname}_sc_volumes_gated.csv")
    ts = timestamp
    lines.append(f"Run:     {ts[:4]}-{ts[4:6]}-{ts[6:8]} {ts[9:11]}:{ts[11:13]}:{ts[13:15]}")
    lines.append("")
    lines.append("Gate groups")
    lines.append("-" * 60)

    for i, ((lo, hi), entries) in enumerate(groups.items(), 1):
        lines.append(f"Group {i}   lower = {lo:.4g} fL   upper = {hi:.4g} fL")
        for stem, n_before, n_after in entries:
            n_removed = n_before - n_after
            pct = 100 * n_removed / n_before if n_before else 0
            total_before += n_before
            total_after += n_after
            lines.append(
                f"  {stem:<60s}  {n_before} → {n_after}"
                f"  ({n_removed} removed, {pct:.1f}%)"
            )
        lines.append("")

    if skipped:
        lines.append("Skipped (no [SizeStats] section)")
        lines.append("-" * 60)
        for name in skipped:
            n = ungated_counts.get(name, '?')
            lines.append(f"  {name:<60s}  {n} values (not gated)")
        lines.append("")

    n_samples = sum(len(v) for v in groups.values())
    total_removed = total_before - total_after
    total_pct = 100 * total_removed / total_before if total_before else 0
    lines.append(
        f"Total: {total_before} → {total_after} values retained across "
        f"{n_samples} sample(s)  ({total_removed} removed, {total_pct:.1f}%)"
    )

    log_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f"Written: {log_path}")


def get_sc_volume_fromdir(full_fpaths) -> pd.DataFrame:
    """
    Gets a dataframe containing single cell volume data from dir of .#m4 files

    Args:
        full_fpaths (list(Path)): list of file paths to be parsed

    Returns:
        DataFrame: single-cell volume data for each coulter counter file in dir
    """
    all_stems, vol_list, _, _, _, _ = _parse_coulter_files(full_fpaths)
    return _build_sc_df(all_stems, vol_list)

def get_volume_stats_fromdir(full_fpaths) -> pd.DataFrame:
    """
    Gets a dataframe containing volume statistic data from dir of .#m4 files

    Args:
        full_fpaths (list(Path)): list of file paths to be parsed

    Returns:
        DataFrame: volume stat data for each coulter counter file in dir
    """
    _, _, stats_stems, _, stats_list, _ = _parse_coulter_files(full_fpaths)
    return _build_stats_df(stats_stems, stats_list)

if __name__ == "__main__":
    main()