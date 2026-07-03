"""
prune_timestamped_subdirs.py

For each sample subdir inside an experiment superdir, finds all directories
whose names begin with a timestamp prefix (YYYYMMDD.HHMMSS or YYYYMMDD_HHMMSS),
groups them by their suffix (everything after the timestamp), and deletes all
but the most recent in each group.

Usage:
    python prune_timestamped_subdirs.py <superdir> [--dry-run]

    <superdir>   Path to the experiment superdir containing sample subdirs
    --dry-run    Print what would be kept/deleted without deleting anything
"""
import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path

_TS_PATTERN = re.compile(r'^(\d{8}[._]\d{6})_(.+)$')


def _ignore_already_gone(func, path, exc_info):
    """
    onexc handler for shutil.rmtree.

    On exFAT/FAT volumes (e.g. external drives on macOS), the OS stores
    resource forks / xattrs in AppleDouble sidecar files ("._<name>").
    Deleting a real entry auto-removes its sidecar, so rmtree can hit a
    FileNotFoundError trying to unlink a sidecar that already vanished.
    That's benign — swallow it; re-raise anything else.
    """
    exc = exc_info[1] if isinstance(exc_info, tuple) else exc_info
    if isinstance(exc, FileNotFoundError):
        return
    raise exc


def main():
    superdir, dry_run = _parse_cli_args()
    _process(superdir, dry_run)


def _parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Delete older timestamped subdirs, keeping only the latest per suffix."
    )
    parser.add_argument('superdir', type=str,
                        help='Path to the experiment superdir')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be kept/deleted without deleting')
    args = parser.parse_args()
    p = Path(args.superdir)
    if not p.is_dir():
        raise FileNotFoundError(f"Directory not found: {p}")
    return p, args.dry_run


def _scan_sample_dir(sample_dir: Path) -> dict:
    """
    Returns a dict mapping suffix → sorted list of (timestamp_str, Path) for
    all timestamped subdirs found directly inside sample_dir.
    Non-matching dirs are ignored.
    """
    groups = defaultdict(list)
    for d in sorted(sample_dir.iterdir()):
        if not d.is_dir():
            continue
        m = _TS_PATTERN.match(d.name)
        if not m:
            continue
        ts, suffix = m.group(1), m.group(2)
        groups[suffix].append((ts, d))
    for suffix in groups:
        groups[suffix].sort(key=lambda x: x[0])
    return dict(groups)


def _process(superdir: Path, dry_run: bool):
    total_kept = 0
    total_deleted = 0
    subdirs_with_matches = 0

    for sample_dir in sorted(superdir.iterdir()):
        if not sample_dir.is_dir():
            continue

        groups = _scan_sample_dir(sample_dir)
        if not groups:
            continue

        subdirs_with_matches += 1
        if dry_run:
            print(f"{sample_dir.name}/")

        for suffix in sorted(groups):
            entries = groups[suffix]   # sorted by timestamp asc
            keep_ts, keep_path = entries[-1]
            to_delete = entries[:-1]

            total_kept += 1
            total_deleted += len(to_delete)

            if dry_run:
                note = f"  ({len(to_delete)} older would be deleted)" if to_delete else ""
                print(f"\tKEEP\t{keep_path.name}{note}")
            else:
                for _, del_path in to_delete:
                    print(f"Deleting: {del_path}")
                    shutil.rmtree(del_path, onexc=_ignore_already_gone)

    if dry_run:
        print(f"\n[dry-run] Would keep {total_kept} dirs, delete {total_deleted} dirs "
              f"across {subdirs_with_matches} sample subdir(s).")
    else:
        print(f"\nDone. Kept {total_kept} dirs, deleted {total_deleted} dirs "
              f"across {subdirs_with_matches} sample subdir(s).")


if __name__ == '__main__':
    main()
