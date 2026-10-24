"""
gating/gates.py

The per-sample gate file format, and the gate math shared by its writer
(gate_experiments_inplace.py) and its readers (compile_experiment.py,
pair_bm_runs.py). GUI-free, so readers don't pull in Tk/matplotlib.

Gate file:
    <sample_subdir>/<YYYYMMDD_HHMMSS>_gating/<sample_subdir_name>_gate.yaml

        experiment: <superdir name>
        sample:     <sample subdir name>
        gates:
          mass_pg:       {lower: <float>, upper: <float>}
          baseline_norm: {lower: <float>, upper: <float>}
          bl_slope:      {lower: <float>, upper: <float>}
          node_dev_mean: {lower: <float>, upper: <float>}
          volume_fl:     {lower: <float>, upper: <float>}

    Every key under `gates:` is optional; a cell passes if it falls inside
    every range present (logical AND). An empty `gates: {}` means the sample
    was deliberately left ungated. The newest *_gating folder wins.

The first four keys are per-cell columns of the SMR mass CSV (baseline_norm
derived, see baseline_norm()); volume_fl is the CELLGROUPED hdf5's
calibrated volume — a separate cell population, linked to the mass CSV only
through analysis/density/cells' matched_peak_index.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fsutil import is_appledouble


MASS_GATE_KEYS = ('mass_pg', 'baseline_norm', 'bl_slope', 'node_dev_mean')
VOLUME_GATE_KEY = 'volume_fl'
GATE_KEYS = MASS_GATE_KEYS + (VOLUME_GATE_KEY,)

# Mass CSV columns MASS_GATE_KEYS need (baseline_norm uses the last two).
MASS_GATE_COLUMNS = ('mass_pg', 'bl_slope', 'node_dev_mean', 'avg_baseline',
                     'peak_time_m')

GATE_DIR_TAG = 'gating'
_GATE_DIR_RE = re.compile(r'^\d{8}_\d{6}_gating$')
# Pre-consolidation layout (separate *_bm_gating / *_ifxm-vol_gating folders,
# top-level lower/upper) — no longer read; only detected so callers can warn.
_LEGACY_GATE_DIR_RE = re.compile(r'_(bm|ifxm-vol)_gating$')


def baseline_norm(df: pd.DataFrame) -> np.ndarray:
    """
    avg_baseline normalized to the mean baseline over the first 10% of the
    run (by peak_time_m) — mirrors gate_mass_results.m's values_for_spec for
    the 'baseline_norm' derived quantity. Falls back to the raw avg_baseline
    if there's no usable reference window.

    Always computed over a sample's full, ungated mass table, so the value a
    cell gets is the same at gating time and at compile time.
    """
    bl = df['avg_baseline'].to_numpy(dtype=float)
    tm = df['peak_time_m'].to_numpy(dtype=float)
    good = np.isfinite(bl) & np.isfinite(tm)
    if not good.any():
        return bl
    tmin = tm[good].min()
    tmax = tm[good].max()
    thr = tmin + 0.10 * (tmax - tmin)
    ref_mask = good & (tm <= thr)
    if not ref_mask.any():
        ref_mask = good
    ref = bl[ref_mask].mean()
    if not np.isfinite(ref) or ref == 0:
        return bl
    return bl / ref


def mass_attr_values(df: pd.DataFrame, key: str) -> np.ndarray:
    """One mass-table gate attribute's per-cell values for a mass CSV table."""
    if key == 'baseline_norm':
        return baseline_norm(df)
    return df[key].to_numpy(dtype=float)


def mass_gate_mask(df: pd.DataFrame, gates: dict) -> np.ndarray:
    """
    Per-row boolean mask of a mass CSV table: the AND of every mass-table
    range in `gates` ({key: (lower, upper)}). Keys outside MASS_GATE_KEYS
    (i.e. volume_fl) are ignored. NaN values fail any gate they're tested by.
    """
    mask = np.ones(len(df), dtype=bool)
    for key, (lo, hi) in gates.items():
        if key not in MASS_GATE_KEYS:
            continue
        v = mass_attr_values(df, key)
        mask &= (v >= lo) & (v <= hi)
    return mask


def range_mask(values, gate: tuple | None) -> np.ndarray:
    """values within gate=(lower, upper), or all-True if gate is None."""
    v = np.asarray(values, dtype=float)
    if gate is None:
        return np.ones(v.shape, dtype=bool)
    lo, hi = gate
    return (v >= lo) & (v <= hi)


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def gate_file_path(sample_dir: Path, timestamp: str) -> Path:
    """Where a gate file written at `timestamp` (YYYYMMDD_HHMMSS) goes."""
    return (sample_dir / f'{timestamp}_{GATE_DIR_TAG}'
            / f'{sample_dir.name}_gate.yaml')


def write_gate_file(sample_dir: Path, experiment: str, gates: dict,
                    timestamp: str) -> Path:
    """Write one sample's gates ({key: (lower, upper)}) — see module docstring."""
    out_path = gate_file_path(sample_dir, timestamp)
    out_path.parent.mkdir(exist_ok=True)
    payload = {
        'experiment': experiment,
        'sample':     sample_dir.name,
        'gates': {
            key: {'lower': float(gates[key][0]), 'upper': float(gates[key][1])}
            for key in GATE_KEYS if key in gates
        },
    }
    with open(out_path, 'w', encoding='utf-8') as fh:
        yaml.dump(payload, fh, default_flow_style=False, sort_keys=False)
    return out_path


def find_gate_file(sample_dir: Path) -> Path | None:
    """The newest <sample_dir>/<timestamp>_gating/*.yaml, or None."""
    gate_dirs = sorted(
        d for d in sample_dir.iterdir()
        if d.is_dir() and not is_appledouble(d) and _GATE_DIR_RE.match(d.name)
    )
    if not gate_dirs:
        return None
    yaml_files = sorted(f for f in gate_dirs[-1].glob('*.yaml')
                        if not is_appledouble(f))
    return yaml_files[0] if yaml_files else None


def has_legacy_gate_dirs(sample_dir: Path) -> bool:
    """True if sample_dir holds pre-consolidation *_bm_gating /
    *_ifxm-vol_gating folders (which are no longer read)."""
    return any(d.is_dir() and _LEGACY_GATE_DIR_RE.search(d.name)
               for d in sample_dir.iterdir())


def load_gate_file(path: Path) -> dict:
    """
    {key: (lower, upper)} for every gate in a gate file — {} if the sample
    was deliberately left ungated. Raises on an unreadable/malformed file or
    an unknown gate key, so a typo can't silently drop a gate.
    """
    payload = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    raw = payload.get('gates') or {}
    unknown = set(raw) - set(GATE_KEYS)
    if unknown:
        raise ValueError(f"unknown gate key(s): {', '.join(sorted(unknown))}")
    return {key: (float(raw[key]['lower']), float(raw[key]['upper']))
            for key in GATE_KEYS if key in raw}
