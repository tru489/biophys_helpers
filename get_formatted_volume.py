import argparse
from pathlib import Path
from CoulterFile import CoulterFile
import pandas as pd
import numpy as np

def main():
    """
    Write CSV from coulter counter files based on pre-selected stats (i.e. 
    population volume gating) from those files
    """
    # Get directory path
    dir_path = parse_cli_args()
    dp_obj = Path(dir_path)
    
    # Filter files
    file_criteria = lambda entry: entry.is_file() \
        and not entry.name.startswith('.') \
        and '.#m4' in entry.name
    filenames = [entry.name for entry in dp_obj.iterdir() if file_criteria(entry)]
    full_fpaths = [dp_obj / Path(filename) for filename in filenames]

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
    df = pd.DataFrame(data_to_write, columns=file_stems)
    df.to_csv(dp_obj / Path('single_cell_volumes.csv'), index=False)

def parse_cli_args():
    """
    Parse CLI arguments. Takes path to coulter counter directory as CLI argument
    """
    parser = argparse.ArgumentParser(description="Process a directory path.")
    parser.add_argument('directory', type=str, help='Path to the directory')

    args = parser.parse_args()

    if not Path(args.directory).is_dir:
        raise FileNotFoundError(f"The directory '{args.directory}' does not exist.")

    return args.directory

if __name__ == "__main__":
    main()