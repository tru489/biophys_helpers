"""
aggregate_bm_vol_files.py

Aggregates buoyant mass (BM) CSVs from SMR runs and FXM volume CSVs
(_ProcessedVolumes.csv) from iFXM runs across one or more experiment
superdirectories. Files are copied into an aggregated/ folder organised
by data type and superdir name.

Usage:
    python aggregate_bm_vol_files.py <superdir1> [superdir2 ...]
    python aggregate_bm_vol_files.py --from-file <paths.txt> [--output <dir>]

    directories     One or more superdir paths (positional)
    --from-file     Text file listing superdir paths, one per line
    --output        Parent directory for aggregated/ (default: parent of first superdir)

Both BM and FXM results are expected to be nested under a sample subdir:
    <superdir>/<sample_subdir>/<run_dir>/<results>
"""
import argparse
from pathlib import Path
import re
import shutil


def main():
    superdirs, output_dir = parse_cli_args()
    aggregate_all(superdirs, output_dir)


def parse_cli_args():
    """
    Parses CLI args. Accepts one or more superdir paths as positional arguments,
    or a text file listing superdir paths (one per line) via --from-file.
    An optional --output flag sets the parent directory for the aggregated folder;
    defaults to the parent of the first superdir.

    Raises:
        FileNotFoundError: a provided directory or path file does not exist

    Returns:
        tuple(list(Path), Path): list of superdirs, output parent directory
    """
    parser = argparse.ArgumentParser(
        description="Aggregate BM and FXM volume CSVs from one or more superdirs."
    )
    parser.add_argument(
        'directories', type=str, nargs='*',
        help='One or more superdir paths'
    )
    parser.add_argument(
        '--from-file', type=str, metavar='FILE',
        help='Text file listing superdir paths, one per line'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Parent directory for the aggregated folder '
             '(default: parent of the first superdir)'
    )

    args = parser.parse_args()

    if args.from_file and args.directories:
        parser.error("Provide either positional directories or --from-file, not both.")

    if args.from_file:
        fpath = Path(args.from_file)
        if not fpath.is_file():
            raise FileNotFoundError(f"Path list file '{args.from_file}' does not exist.")
        lines = fpath.read_text().splitlines()
        superdirs = [Path(ln.strip()) for ln in lines if ln.strip()]
    elif args.directories:
        superdirs = [Path(d) for d in args.directories]
    else:
        parser.error("Provide at least one directory or use --from-file.")

    for sd in superdirs:
        if not sd.is_dir():
            raise FileNotFoundError(f"Directory '{sd}' does not exist.")

    output_dir = Path(args.output) if args.output else superdirs[0].parent

    return superdirs, output_dir


def _find_bm_csvs(superdir: Path) -> list:
    """
    Finds buoyant mass CSV files within a superdir.

    Each direct subdir of superdir may contain one run subdir matching
    \d{8}.\d{6}... from which a dated CSV is collected.

    Args:
        superdir (Path): superdir to search

    Raises:
        RuntimeError: more than one BM CSV found within a single sample subdir

    Returns:
        list(Path): matched CSV files
    """
    subdir_pattern = re.compile(r"\d{8}\.\d{6}[a-zA-Z\d\s_+-]*")
    csv_pattern = re.compile(r"\d{4}-\d{2}-\d{2}[a-zA-Z\d\s_+-]*\.csv")
    found = []

    for subdir in sorted(superdir.iterdir()):
        if not subdir.is_dir():
            continue
        subdir_found = False
        for inner_subdir in subdir.iterdir():
            if not (inner_subdir.is_dir() and subdir_pattern.match(inner_subdir.name)):
                continue
            if subdir_found:
                raise RuntimeError(
                    f"Multiple BM CSVs found in directory {subdir.name}"
                )
            for file in inner_subdir.iterdir():
                if file.is_file() and csv_pattern.match(file.name):
                    found.append(file)
                    subdir_found = True
                    break

    return found


def _find_fxm_csvs(superdir: Path) -> list:
    """
    Finds FXM ProcessedVolumes CSV files within a superdir.

    Looks for run dirs matching \d{8}_\d{6}_imaging_fxm_results, then descends
    into stage2_analysis/ for *_ProcessedVolumes.csv files.

    Args:
        superdir (Path): superdir to search

    Returns:
        list(Path): matched CSV files
    """
    run_dir_pattern = re.compile(r"\d{8}_\d{6}_imaging_fxm_results")
    found = []

    for subdir in sorted(superdir.iterdir()):
        if not subdir.is_dir():
            continue
        for run_dir in subdir.iterdir():
            if not (run_dir.is_dir() and run_dir_pattern.match(run_dir.name)):
                continue
            stage2_dir = run_dir / 'stage2_analysis'
            if not stage2_dir.is_dir():
                continue
            for file in stage2_dir.iterdir():
                if file.is_file() and file.name.endswith('_ProcessedVolumes.csv'):
                    found.append(file)
                    break

    return found


def aggregate_all(superdirs: list, output_dir: Path):
    """
    Aggregates BM and FXM CSVs from all superdirs into:
        <output_dir>/aggregated/buoyant_mass/<superdir_name>/<csv files>
        <output_dir>/aggregated/volumes/<superdir_name>/<csv files>

    Args:
        superdirs (list(Path)): list of superdirs to process
        output_dir (Path): parent directory for the aggregated folder
    """
    aggr_dir = output_dir / 'aggregated'

    for superdir in superdirs:
        print(f"\nProcessing: {superdir.name}")

        bm_files = _find_bm_csvs(superdir)
        for f in bm_files:
            dest = aggr_dir / 'smr_data' / superdir.name
            dest.mkdir(parents=True, exist_ok=True)
            print(f"  [BM]  {f.name}")
            shutil.copy(f, dest / f.name)

        fxm_files = _find_fxm_csvs(superdir)
        for f in fxm_files:
            dest = aggr_dir / 'imaging_fxm' / superdir.name
            dest.mkdir(parents=True, exist_ok=True)
            print(f"  [FXM] {f.name}")
            shutil.copy(f, dest / f.name)

        if not bm_files and not fxm_files:
            print("  No matching files found.")

    print(f"\nDone. Output written to: {aggr_dir}")


if __name__ == '__main__':
    main()
