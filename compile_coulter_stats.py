import pandas as pd
import re
import argparse
from pathlib import Path

def main():
    """
    Write CSV from coulter counter files based on pre-selected stats (i.e. 
    population volume gating) from those files
    """
    # Desired rows for final output CSV (matches with row names in #m4 file)
    desired_rows = ['Mean', 'Mode', 'Median', 'SD', 'CV', 'MinSize', 'MaxSize', 'SampleSize']

    # Get directory path
    dir_path = parse_cli_args()
    dp_obj = Path(dir_path)
    
    # Filter files. Default is 
    file_criteria = lambda entry: entry.is_file() \
        and not entry.name.startswith('.') \
        and not 'coulter_compiled' in entry.name
    filenames = [entry.name for entry in dp_obj.iterdir() if file_criteria(entry)]
    full_fpaths = [dp_obj / Path(filename) for filename in filenames]

    full_dict = {}
    for fn in full_fpaths:
        dict_single = extract_numbers(fn.resolve(), desired_rows, Path(fn).stem)
        full_dict.update(dict_single)
    
    df = pd.DataFrame.from_dict(full_dict)
    df.index = desired_rows
    df.to_csv(dp_obj / Path('coulter_compiled.csv'))

def parse_cli_args():
    """
    Parse CLI arguments. Takes path to coulter counter directory as CLI argument
    """
    parser = argparse.ArgumentParser(description="Process a directory path.")
    parser.add_argument('directory', type=str, help='Path to the directory')

    args = parser.parse_args()

    
    if Path(args.directory).is_dir:
        print(f"The directory '{args.directory}' exists.")
    else:
        print(f"The directory '{args.directory}' does not exist.")

    return args.directory

def extract_numbers(file_path, desired_rows, pd_col_name):
    """
    Extracts pre-selected/gated statistic values from a single coulter counter
    file

    Args:
        file_path (str): pathlib Path object for file
        desired_rows (list(str)): statistics to be selected in the file
        pd_col_name (str): name of dictionary key for values

    Raises:
        ValueError: Raises error if statistics are not in the coulter counter
            file

    Returns:
        dict: dictionary where the file name (- extension) is the key and a 
            list of statistics (consistent with desired_rows) are the values
    """
    with file_path.open('r') as file:
        lines = file.readlines()
    
    start_index = None
    end_index = None
    
    # Find the indices for [SizeStats] and [SizePctX]
    for i, line in enumerate(lines):
        if '[SizeStats]' in line:
            start_index = i
        if '[SizePctX]' in line:
            end_index = i
            break
    
    if start_index is None or end_index is None:
        raise ValueError("The specified sections were not found in the file.")
    
    # Extract lines between [SizeStats] and [SizePctX]
    relevant_lines = lines[start_index + 1:end_index]
    
    # Extract numbers after the equals sign
    numbers = []
    for line in relevant_lines:
        statname, val = re.match(r'([\w\(\)\,]+)=\s*([\d\.]+)', line).groups()
        if statname in desired_rows:
            numbers.append(val)
    
    return {pd_col_name: numbers}

if __name__ == '__main__':
    main()