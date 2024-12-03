import argparse
from pathlib import Path
import re
import shutil

def main():
    """
    From a directory of analyzed SMR data with buoyant mass data in subfolders,
    aggregate all BM CSVs into a folder at the level of the data-containing
    directories
    """
    dirpath = parse_cli_args()
    aggr_csvs(dirpath, 'bm_csv_aggr')


def parse_cli_args():
    """
    Parses CLI args, taking a directory to be parsed as an input

    Raises:
        FileNotFoundError: directory does not exist

    Returns:
        pathlib.Path: Path of chosen directory
    """
    parser = argparse.ArgumentParser(description="Process a directory path.")
    parser.add_argument('directory', type=str, help='Path to the directory')

    args = parser.parse_args()

    if not Path(args.directory).is_dir:
        msg = f"The directory '{args.directory}' does not exist."
        raise FileNotFoundError(msg)

    return Path(args.directory)


def aggr_csvs(dirpath: Path, aggr_dir_name: str):
    # Define the regex patterns
    subdir_pattern = re.compile(r"\d{8}\.\d{6}[a-zA-Z\d_-]*")
    csv_pattern = re.compile(r"\d{4}-\d{2}-\d{2}[a-zA-Z\d_-]*\.csv")

    # Iterate through each subdirectory in the base directory
    for subdir in dirpath.iterdir():
        if subdir.is_dir():
            subdir_found = False

            # Check for subdirectories matching the pattern
            for inner_subdir in subdir.iterdir():
                if inner_subdir.is_dir() and subdir_pattern.match(inner_subdir.name):
                    if subdir_found:
                        msg = "Multiple buoyant mass CSVs found in " + \
                            f"directory {subdir.name}"
                        raise RuntimeError(msg)

                    # Look for .csv files matching the pattern
                    for file in inner_subdir.iterdir():
                        if file.is_file() and csv_pattern.match(file.name):
                            # Define the new directory path
                            new_dir = dirpath / aggr_dir_name
                            new_dir.mkdir(exist_ok=True)
                            # Copy the matching .csv file
                            shutil.copy(file, new_dir / file.name)
                            subdir_found = True
                            break


if __name__ == '__main__':
    main()