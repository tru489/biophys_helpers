import pandas as pd
import argparse
from pathlib import Path
from CoulterFile import CoulterFile
import numpy as np

def main():
    """
    Write 2 CSVs from coulter counter files, 1 with summary stats based on pre-
    selected data from multisizer software, and 1 with full single-cell volume 
    data
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

    if run_sc:
        df_sc = get_sc_volume_fromdir(full_fpaths)
        df_sc.to_csv(dp_obj / Path('single_cell_volumes.csv'), index=False)

    if run_stats:
        df_stats = get_volume_stats_fromdir(full_fpaths)
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

def get_sc_volume_fromdir(full_fpaths) -> pd.DataFrame:
    """
    Gets a dataframe containing single cell volume data from dir of .#m4 files

    Args:
        full_fpaths (list(str)): list of file paths to be parsed

    Returns:
        DataFrame: single-cell volume data for each coulter counter file in dir
    """
    vol_list = []
    file_stems = []
    for fn in full_fpaths:
        coulter_file = CoulterFile(fn.resolve())
        vol_list.append(coulter_file.get_volumes())
        file_stems.append(Path(fn).stem)
    
    max_length = max(len(arr) for arr in vol_list)

    # Pad the arrays with NaN
    padded_arrays = []
    for arr in vol_list:
        # Create a new array filled with NaN, and then fill it with the values of the original array
        padded_arr = np.full(max_length, np.nan)  # or you can use None
        padded_arr[:len(arr)] = arr
        padded_arrays.append(padded_arr)

    # Convert to a 2D numpy array
    data_to_write = np.column_stack(padded_arrays)

    # Write to CSV using pandas for easier handling
    return pd.DataFrame(data_to_write, columns=file_stems)

def get_volume_stats_fromdir(full_fpaths) -> pd.DataFrame:
    """
    Gets a dataframe containing volume statistic data from dir of .#m4 files

    Args:
        full_fpaths (list(str)): list of file paths to be parsed

    Returns:
        DataFrame: volume stat data for each coulter counter file in dir
    """
    all_dicts = []
    file_stems = []
    for fn in full_fpaths:
        coulter_file = CoulterFile(fn.resolve())
        dict_single = coulter_file.get_stats()
        all_dicts.append(dict_single)
        file_stems.append(Path(fn).stem)
    
    keys_ = all_dicts[0].keys()
    full_dict = {fs: [] for fs in file_stems}
    for i, fs in enumerate(file_stems):
        for k in keys_:
            full_dict[fs].append(all_dicts[i][k])
    df = pd.DataFrame.from_dict(full_dict)
    df.index = keys_
    return df

if __name__ == "__main__":
    main()