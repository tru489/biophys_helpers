"""
Small filesystem helpers shared across the biophys_helpers scripts.

Kept dependency-free (stdlib only) so any script can import it without
pulling in heavy GUI/plotting stacks.
"""

from pathlib import Path


def is_appledouble(p: Path) -> bool:
    """
    True for macOS AppleDouble sidecar files ("._<name>").

    On exFAT/FAT volumes (e.g. external drives), macOS stores resource forks /
    xattrs in these binary sidecars alongside every real file. They share the
    real file's suffix (e.g. "._foo.yaml"), so they slip past a plain suffix
    filter. Worse, "._foo.yaml" sorts *before* "foo.yaml", so a
    sorted(glob(...))[0] would pick the binary sidecar and fail to decode it.
    Skip them everywhere we scan the filesystem.
    """
    return p.name.startswith('._')
