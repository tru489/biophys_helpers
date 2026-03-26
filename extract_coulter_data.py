"""
extract_coulter_data.py

Parses a directory of Coulter counter (.#m4) files and exports CSV files
containing single-cell volume measurements and/or summary statistics
pre-selected in the Multisizer software. Output is written to a timestamped
directory alongside the input directory.

Usage:
    python extract_coulter_data.py <directory> [-stats] [-single-stats] [-r]

    <directory>     Path to folder containing .#m4 files
    (no flags)      Write single-cell volumes CSV and histograms (default)
    -stats          Write only the volume stats CSV (no volumes, no histograms)
    -single-stats   Write both single-cell volumes CSV and stats CSV + histograms
    -r              Recursively include .#m4 files from subdirectories;
                    column names are prefixed with the relative subdir path
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
    Extract single-cell volume data and/or summary statistics from a directory
    of Coulter counter .#m4 files. Outputs are written to a timestamped
    subdirectory inside the input directory.
    """
    dir_path, run_stats, run_sc, recursive = parse_cli_args()
    dp_obj = Path(dir_path)
    dirname = dp_obj.name

    full_fpaths, display_names = _collect_files(dp_obj, recursive)
    all_stems, vol_list, stats_stems, stats_list = _parse_coulter_files(
        full_fpaths, display_names)

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    out_dir = dp_obj / f'{timestamp}_coulter-processed'
    out_dir.mkdir()

    if run_sc:
        _build_sc_df(all_stems, vol_list).to_csv(
            out_dir / f'{dirname}_sc_volumes.csv', index=False)
        _plot_sc_histograms(all_stems, vol_list, out_dir)

    if run_stats and stats_stems:
        _build_stats_df(stats_stems, stats_list).to_csv(
            out_dir / f'{dirname}_volume_stats.csv')
    elif run_stats and not stats_stems:
        print("Warning: no files with [SizeStats] data found — stats CSV not written.")


def parse_cli_args():
    """
    Parse CLI arguments.

    Behaviour:
        (no flags)      single-cell volumes CSV + histograms
        -stats          stats CSV only
        -single-stats   both volumes CSV and stats CSV + histograms
        -r              recurse into subdirectories

    Raises:
        FileNotFoundError: directory does not exist

    Returns:
        tuple(str, bool, bool, bool): directory, run_stats, run_sc, recursive
    """
    parser = argparse.ArgumentParser(
        description="Extract volume data from a directory of Coulter counter .#m4 files."
    )
    parser.add_argument('directory', type=str, help='Path to the directory')
    parser.add_argument('-stats', action='store_true',
                        help='Write only the volume stats CSV')
    parser.add_argument('-single-stats', action='store_true',
                        help='Write both single-cell volumes CSV and stats CSV')
    parser.add_argument('-r', action='store_true',
                        help='Recursively include .#m4 files from subdirectories')

    args = parser.parse_args()

    if not Path(args.directory).is_dir():
        raise FileNotFoundError(f"The directory '{args.directory}' does not exist.")

    single_stats = getattr(args, 'single_stats', False)

    if args.stats:
        run_stats, run_sc = True, False
    elif single_stats:
        run_stats, run_sc = True, True
    else:
        run_stats, run_sc = False, True   # default: volumes only

    return args.directory, run_stats, run_sc, args.r


def _collect_files(dp_obj: Path, recursive: bool) -> tuple:
    """
    Collect .#m4 files from dp_obj. If recursive, also descends into
    subdirectories, prefixing each display name with underscore-joined
    subdir path components.

    Args:
        dp_obj (Path): root directory
        recursive (bool): whether to recurse into subdirectories

    Returns:
        tuple(list(Path), list(str)): full file paths and display names
    """
    is_m4 = lambda p: not p.name.startswith('.') and '.#m4' in p.name

    full_fpaths = []
    display_names = []

    root_files = sorted([e for e in dp_obj.iterdir() if e.is_file() and is_m4(e)],
                        key=lambda p: p.name)
    for f in root_files:
        full_fpaths.append(f)
        display_names.append(f.stem)

    if recursive:
        subdirs = sorted([e for e in dp_obj.iterdir()
                          if e.is_dir() and not e.name.startswith('.')])
        for subdir in subdirs:
            for f in sorted(subdir.rglob('*'), key=lambda p: p):
                if f.is_file() and is_m4(f):
                    rel = f.relative_to(dp_obj)
                    prefix = '_'.join(rel.parts[:-1])
                    display_names.append(f'{prefix}_{f.stem}')
                    full_fpaths.append(f)

    return full_fpaths, display_names


def _parse_coulter_files(full_fpaths, display_names=None) -> tuple:
    """
    Opens each .#m4 file once and extracts single-cell volumes and stats.
    Files without a [SizeStats] section are included in volumes but excluded
    from stats output (a warning is printed for each).

    Args:
        full_fpaths (list(Path)): file paths to parse
        display_names (list(str) | None): column names; defaults to file stems

    Returns:
        tuple(list(str), list(np.ndarray), list(str), list(dict)):
            all_stems, vol_list, stats_stems, stats_list
    """
    if display_names is None:
        display_names = [Path(fn).stem for fn in full_fpaths]

    all_stems, vol_list = [], []
    stats_stems, stats_list = [], []
    n = len(full_fpaths)

    for i, (fn, name) in enumerate(zip(full_fpaths, display_names), 1):
        print(f"Parsing file {i}/{n}: {fn.name}")
        coulter_file = CoulterFile(fn.resolve())
        all_stems.append(name)
        vol_list.append(coulter_file.get_volumes_ungated())
        stats = coulter_file.get_stats()
        if stats is not None:
            stats_stems.append(name)
            stats_list.append(stats)
        else:
            print(f"  Warning: no [SizeStats] in {fn.name} — excluded from stats output")

    return all_stems, vol_list, stats_stems, stats_list


def _build_sc_df(file_stems, vol_list) -> pd.DataFrame:
    """
    Build a DataFrame of single-cell volumes, NaN-padded to equal column length.

    Args:
        file_stems (list(str)): column names
        vol_list (list(np.ndarray)): volume arrays per file

    Returns:
        pd.DataFrame: shape (max_length, n_files)
    """
    max_length = max(len(arr) for arr in vol_list)
    padded_arrays = []
    for arr in vol_list:
        padded_arr = np.full(max_length, np.nan)
        padded_arr[:len(arr)] = arr
        padded_arrays.append(padded_arr)
    return pd.DataFrame(np.column_stack(padded_arrays), columns=file_stems)


def _build_stats_df(file_stems, stats_list) -> pd.DataFrame:
    """
    Build a DataFrame of Multisizer summary statistics.

    Args:
        file_stems (list(str)): column names
        stats_list (list(dict)): one dict of stats per file

    Returns:
        pd.DataFrame: shape (n_stats, n_files), indexed by stat name
    """
    keys_ = stats_list[0].keys()
    full_dict = {fs: [d[k] for k in keys_] for fs, d in zip(file_stems, stats_list)}
    df = pd.DataFrame.from_dict(full_dict)
    df.index = keys_
    return df


def _plot_sc_histograms(all_stems: list, vol_list: list, out_dir: Path):
    """
    Saves one histogram PNG per .#m4 file into out_dir/fig/.
    Each plot is ungated and uses a log x-axis with fixed 20–100,000 fL range.

    Args:
        all_stems (list(str)): file display names
        vol_list (list(np.ndarray)): ungated volume arrays
        out_dir (Path): output directory; fig/ subfolder is created here
    """
    fig_dir = out_dir / 'fig'
    fig_dir.mkdir(exist_ok=True)

    log_bins = np.logspace(np.log10(20), np.log10(100_000), 201)

    for stem, vols in zip(all_stems, vol_list):
        data = vols[~np.isnan(vols)]
        if len(data) == 0:
            continue
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.hist(data, bins=log_bins, alpha=0.5, edgecolor='black', linewidth=0.3)
        ax.set_xscale('log')
        ax.set_xlim(20, 100_000)
        ax.set_xlabel('Total Volume (fL)')
        ax.set_ylabel('count')
        ax.set_title(stem, fontsize=8, wrap=True)
        fig.tight_layout()
        safe_name = stem.replace('/', '_').replace(' ', '_')
        out_path = fig_dir / f'{safe_name}.png'
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Written: {out_path}")

    print(f"[histograms] {len(all_stems)} histogram(s) written to {fig_dir}")


def get_sc_volume_fromdir(full_fpaths) -> pd.DataFrame:
    """
    Public API: gets a DataFrame of single-cell volumes from a list of .#m4 paths.

    Args:
        full_fpaths (list(Path)): file paths to parse

    Returns:
        pd.DataFrame: single-cell volume data, one column per file
    """
    all_stems, vol_list, _, _ = _parse_coulter_files(full_fpaths)
    return _build_sc_df(all_stems, vol_list)


def get_volume_stats_fromdir(full_fpaths) -> pd.DataFrame:
    """
    Public API: gets a DataFrame of Multisizer summary statistics from a list
    of .#m4 paths.

    Args:
        full_fpaths (list(Path)): file paths to parse

    Returns:
        pd.DataFrame: volume stats, one column per file
    """
    _, _, stats_stems, stats_list = _parse_coulter_files(full_fpaths)
    return _build_stats_df(stats_stems, stats_list)


if __name__ == "__main__":
    main()
