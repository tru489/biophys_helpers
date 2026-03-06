import pandas as pd
import argparse
from pathlib import Path
from CoulterFile import CoulterFile
import numpy as np

def main():
    """
    Write 2 CSVs from a directory containing coulter counter files, 1 with 
    summary stats based on preselected data from multisizer software, and 1 
    with full single-cell volume data
    """
    # Get directory path
    dir_path, run_stats, run_sc = parse_cli_args()
    dp_obj = Path(dir_path)

    # Filter files
    file_criteria = lambda entry: entry.is_file() \
        and not entry.name.startswith('.') \
        and '.#m4' in entry.name
    filenames = [entry.name for entry in dp_obj.iterdir() if file_criteria(entry)]
    filenames.sort()
    full_fpaths = [dp_obj / Path(filename) for filename in filenames]

    file_stems, vol_list, stats_list = _parse_coulter_files(full_fpaths)

    if run_sc:
        df_sc = _build_sc_df(file_stems, vol_list)
        df_sc.to_csv(dp_obj / Path('single_cell_volumes.csv'), index=False)

    if run_stats:
        df_stats = _build_stats_df(file_stems, stats_list)
        df_stats.to_csv(dp_obj / Path('stats.csv'))

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

    args = parser.parse_args()

    if not Path(args.directory).is_dir:
        raise FileNotFoundError(f"The directory '{args.directory}' does not exist.")

    if not args.stats and not args.single:
        args.stats = True
        args.single = True

    return args.directory, args.stats, args.single

def _parse_coulter_files(full_fpaths) -> tuple:
    """
    Opens each .#m4 file once and extracts both single-cell volumes and stats.

    Args:
        full_fpaths (list(Path)): list of file paths to be parsed

    Returns:
        tuple(list(str), list(np.array), list(dict)): file stems, volumes per
            file, stats per file
    """
    file_stems, vol_list, stats_list = [], [], []
    n = len(full_fpaths)
    for i, fn in enumerate(full_fpaths, 1):
        print(f"Parsing file {i}/{n}: {fn.name}")
        coulter_file = CoulterFile(fn.resolve())
        file_stems.append(Path(fn).stem)
        vol_list.append(coulter_file.get_volumes())
        stats_list.append(coulter_file.get_stats())
    return file_stems, vol_list, stats_list

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

def get_sc_volume_fromdir(full_fpaths) -> pd.DataFrame:
    """
    Gets a dataframe containing single cell volume data from dir of .#m4 files

    Args:
        full_fpaths (list(Path)): list of file paths to be parsed

    Returns:
        DataFrame: single-cell volume data for each coulter counter file in dir
    """
    file_stems, vol_list, _ = _parse_coulter_files(full_fpaths)
    return _build_sc_df(file_stems, vol_list)

def get_volume_stats_fromdir(full_fpaths) -> pd.DataFrame:
    """
    Gets a dataframe containing volume statistic data from dir of .#m4 files

    Args:
        full_fpaths (list(Path)): list of file paths to be parsed

    Returns:
        DataFrame: volume stat data for each coulter counter file in dir
    """
    file_stems, _, stats_list = _parse_coulter_files(full_fpaths)
    return _build_stats_df(file_stems, stats_list)

if __name__ == "__main__":
    main()