import pandas as pd
import argparse
from pathlib import Path
from CoulterFile import CoulterFile

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
        and not 'coulter_compiled' in entry.name
    filenames = [entry.name for entry in dp_obj.iterdir() if file_criteria(entry)]
    full_fpaths = [dp_obj / Path(filename) for filename in filenames]

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
    df.to_csv(dp_obj / Path('coulter_compiled.csv'))

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

if __name__ == '__main__':
    main()