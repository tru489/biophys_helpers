"""
compile_experiment.py

Automatically discovers and compiles all per-sample data for an experiment
into a pair of HDF5 files. Given a superdir, it walks each sample subdir,
finds every known data type (BM mass, iFXM volume, pairing, gating, images),
loads them, and writes a structured output.

Recognised sub-subdir types (all optional; most-recent used if multiple exist):
    *_mass_results          — mass CSV with mass_pg column
    *_imaging_fxm_results   — stage2_analysis/*_ProcessedVolumes.csv;
                               stage1_image_processing/*_CELLGROUPED.hdf5;
                               stage2_analysis/*_Hdf5PathIndex.csv
    *_pairing_results       — *_PairedSMRVolumes.csv
    *_bm_gating             — YAML with lower/upper thresholds
    *_ifxm-vol_gating       — YAML with lower/upper thresholds

Pairing resolution (priority):
    1. *_pairing_results/*_PairedSMRVolumes.csv (if dir present)
    2. ProcessedVolumes rows where matched_mass is not NaN
    3. None — no pairing key written

Output:
    <superdir>/YYYYMMDD_HHMMSS_compiled/
        experiment_data.h5   — DataFrames (pandas HDFStore)
        images.h5            — per-transit BF/FL image stacks (h5py)

experiment_data.h5 key layout:
    /metadata                          — one row per sample (summary + gate values)
    /samples/{safe_name}/mass          — full mass CSV DataFrame
    /samples/{safe_name}/volume        — full ProcessedVolumes DataFrame
    /samples/{safe_name}/pairing       — paired rows (if available)

images.h5 key layout:
    /{safe_name}/{transit_idx:05d}/bf  — (n_frames, H, W) uint8
    /{safe_name}/{transit_idx:05d}/fl  — (n_frames, H, W) uint16

{safe_name} replaces - and . with _ so HDF5 key rules are satisfied.
The original sample directory name is preserved in /metadata.sample_name.

Usage:
    python compile_experiment.py <superdir>
"""
import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import warnings
import h5py
import numpy as np
import pandas as pd
import yaml

# Sample dirs that start with digits (e.g. "0h_pt1") trigger a benign
# NaturalNameWarning from PyTables — suppress it; we always use
# store[key] notation, not attribute-style natural naming.
warnings.filterwarnings('ignore', message='object name is not a valid Python identifier')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_cli_args() -> Path:
    parser = argparse.ArgumentParser(
        description="Compile per-sample experiment data into a single HDF5 file."
    )
    parser.add_argument('superdir', type=str,
                        help='Path to the experiment superdir')
    args = parser.parse_args()
    p = Path(args.superdir)
    if not p.is_dir():
        raise FileNotFoundError(f"Directory not found: {p}")
    return p


# ---------------------------------------------------------------------------
# Key sanitisation
# ---------------------------------------------------------------------------

def _safe_key(name: str) -> str:
    """Replace characters invalid in HDF5 key names (- and .) with _."""
    return re.sub(r'[-.]', '_', name)


# ---------------------------------------------------------------------------
# Per-sample discovery
# ---------------------------------------------------------------------------

def _last_matching_dir(parent: Path, pattern: re.Pattern) -> Path | None:
    """Return the lexicographically last subdir whose name matches pattern."""
    matches = sorted(
        d for d in parent.iterdir()
        if d.is_dir() and pattern.search(d.name)
    )
    return matches[-1] if matches else None


def _discover_sample(sample_dir: Path) -> dict:
    """
    Locate the relevant file path for each known data type inside sample_dir.

    Returns a dict with keys:
        mass_path, volume_path, pairing_path, bm_gate_path, ifxm_gate_path
    Any key whose source is absent is set to None.
    """
    paths = {
        'mass_path':       None,
        'volume_path':     None,
        'pairing_path':    None,
        'bm_gate_path':    None,
        'ifxm_gate_path':  None,
        'hdf5_src_path':   None,
        'hdf5_index_path': None,
    }

    # --- mass_results ---
    mass_dir = _last_matching_dir(sample_dir, re.compile(r'_mass_results$'))
    if mass_dir is not None:
        for f in sorted(mass_dir.iterdir()):
            if (f.is_file() and f.suffix == '.csv'
                    and not f.name.startswith('curation_index')):
                try:
                    hdr = pd.read_csv(f, nrows=0)
                except Exception:
                    continue
                if 'mass_pg' in hdr.columns:
                    paths['mass_path'] = f
                    break

    # --- imaging_fxm_results ---
    fxm_dir = _last_matching_dir(sample_dir, re.compile(r'_imaging_fxm_results$'))
    if fxm_dir is not None:
        stage2 = fxm_dir / 'stage2_analysis'
        if stage2.is_dir():
            for f in stage2.iterdir():
                if f.is_file() and f.name.endswith('_ProcessedVolumes.csv'):
                    paths['volume_path'] = f
                    break
            idx_files = list(stage2.glob('*_Hdf5PathIndex.csv'))
            if idx_files:
                paths['hdf5_index_path'] = idx_files[0]

        stage1 = fxm_dir / 'stage1_image_processing'
        if stage1.is_dir():
            hdf5_files = list(stage1.glob('*.hdf5'))
            if hdf5_files:
                paths['hdf5_src_path'] = hdf5_files[0]

    # --- pairing_results ---
    pair_dir = _last_matching_dir(sample_dir, re.compile(r'_pairing_results$'))
    if pair_dir is not None:
        for f in pair_dir.iterdir():
            if f.is_file() and f.name.endswith('_PairedSMRVolumes.csv'):
                paths['pairing_path'] = f
                break

    # --- bm_gating ---
    bm_gate_dir = _last_matching_dir(sample_dir, re.compile(r'_bm_gating$'))
    if bm_gate_dir is not None:
        yaml_files = sorted(bm_gate_dir.glob('*.yaml'))
        if yaml_files:
            paths['bm_gate_path'] = yaml_files[0]

    # --- ifxm-vol_gating ---
    ifxm_gate_dir = _last_matching_dir(sample_dir, re.compile(r'_ifxm-vol_gating$'))
    if ifxm_gate_dir is not None:
        yaml_files = sorted(ifxm_gate_dir.glob('*.yaml'))
        if yaml_files:
            paths['ifxm_gate_path'] = yaml_files[0]

    return paths


# ---------------------------------------------------------------------------
# Image stack helpers
# ---------------------------------------------------------------------------

def _pad_stack(frames: list) -> np.ndarray:
    """Stack a list of 2D arrays; zero-pad to max shape if sizes differ."""
    if len(set(f.shape for f in frames)) == 1:
        return np.stack(frames)
    max_h = max(f.shape[0] for f in frames)
    max_w = max(f.shape[1] for f in frames)
    out = np.zeros((len(frames), max_h, max_w), dtype=frames[0].dtype)
    for i, f in enumerate(frames):
        out[i, :f.shape[0], :f.shape[1]] = f
    return out


def _save_images_for_sample(hdf5_src: Path, index_csv: Path,
                             out_grp, sample_name: str) -> int:
    """
    Read per-transit BF frames from the CELLGROUPED source and write stacks
    into out_grp (an open h5py group for this sample).

    Key layout inside out_grp:
        {transit_idx:05d}/bf  — (n_frames, H, W) uint8
    """
    idx_df = pd.read_csv(index_csv)
    n_transits = idx_df['TransitIndex'].nunique()

    with h5py.File(hdf5_src, 'r') as src:
        for transit_id, rows in idx_df.groupby('TransitIndex'):
            bf_frames = [src[p][()] for p in rows['Hdf5PathsBF']]
            bf_stack  = _pad_stack(bf_frames)
            key = f'{int(transit_id):05d}'
            out_grp.create_dataset(f'{key}/bf', data=bf_stack,
                                   compression='gzip', compression_opts=4)

    return n_transits


# ---------------------------------------------------------------------------
# Per-type loaders
# ---------------------------------------------------------------------------

def _load_gate(path: Path) -> tuple[float, float] | None:
    """Read a gating YAML and return (lower, upper), or None on failure."""
    try:
        data = yaml.safe_load(path.read_text(encoding='utf-8'))
        return (float(data['lower']), float(data['upper']))
    except Exception as exc:
        print(f"  [warn] could not read gate YAML {path.name}: {exc}")
        return None


def _resolve_pairing(volume_df: pd.DataFrame | None,
                     pairing_path: Path | None) -> tuple[pd.DataFrame | None, str]:
    """
    Determine the pairing DataFrame and source label.

    Priority:
        1. pairing_path (PairedSMRVolumes.csv from pairing_results dir)
        2. Non-NaN matched_mass rows in volume_df
        3. None
    Returns (df_or_None, source_label).
    """
    if pairing_path is not None:
        try:
            df = pd.read_csv(pairing_path)
            return df, 'pairing_results'
        except Exception as exc:
            print(f"  [warn] could not read {pairing_path.name}: {exc}")

    if volume_df is not None and 'matched_mass' in volume_df.columns:
        paired = volume_df[volume_df['matched_mass'].notna()].copy()
        if not paired.empty:
            return paired, 'volume_cols'

    return None, 'none'


# ---------------------------------------------------------------------------
# Main compilation
# ---------------------------------------------------------------------------

def compile_experiment(superdir: Path) -> tuple[list[dict], list[dict]]:
    """
    Walk every sample subdir, discover data, and return:
        (sample_records, meta_rows)
    where each sample_record has the loaded DataFrames ready for writing.
    """
    sample_records = []

    for sample_dir in sorted(superdir.iterdir()):
        if not sample_dir.is_dir():
            continue
        paths = _discover_sample(sample_dir)

        # Skip dirs that look like output dirs (no recognised data)
        if all(v is None for v in paths.values()):
            continue

        name = sample_dir.name

        # Load each data type
        mass_df = None
        if paths['mass_path'] is not None:
            try:
                mass_df = pd.read_csv(paths['mass_path'])
            except Exception as exc:
                print(f"  [warn] {name}: could not read mass CSV: {exc}")

        volume_df = None
        if paths['volume_path'] is not None:
            try:
                volume_df = pd.read_csv(paths['volume_path'])
            except Exception as exc:
                print(f"  [warn] {name}: could not read volume CSV: {exc}")

        pairing_df, pairing_src = _resolve_pairing(
            volume_df, paths['pairing_path'])

        bm_gate = (_load_gate(paths['bm_gate_path'])
                   if paths['bm_gate_path'] else None)
        ifxm_gate = (_load_gate(paths['ifxm_gate_path'])
                     if paths['ifxm_gate_path'] else None)

        has_images = (paths['hdf5_src_path'] is not None
                      and paths['hdf5_index_path'] is not None)

        # Console summary line
        def _tick(val, label=''):
            return f'ok({label})' if (val is not None and label) else ('ok' if val is not None else '--')

        print(
            f"[{name}]"
            f"  mass={_tick(mass_df)}"
            f"  volume={_tick(volume_df)}"
            f"  pairing={_tick(pairing_df, pairing_src)}"
            f"  bm_gate={_tick(bm_gate)}"
            f"  ifxm_gate={_tick(ifxm_gate)}"
            f"  images={'ok' if has_images else '--'}"
        )

        sample_records.append({
            'name':            name,
            'mass_df':         mass_df,
            'volume_df':       volume_df,
            'pairing_df':      pairing_df,
            'bm_gate':         bm_gate,
            'ifxm_gate':       ifxm_gate,
            'hdf5_src_path':   paths['hdf5_src_path'],
            'hdf5_index_path': paths['hdf5_index_path'],
        })

    return sample_records


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _write_output(superdir: Path, sample_records: list) -> Path:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = superdir / f'{timestamp}_compiled'
    out_dir.mkdir()
    h5_path = out_dir / 'experiment_data.h5'
    images_h5_path = out_dir / 'images.h5'

    meta_rows = []
    for rec in sample_records:
        name = rec['name']
        bm = rec['bm_gate']
        ifxm = rec['ifxm_gate']
        meta_rows.append({
            'sample_name':    name,
            'hdf5_key':       _safe_key(name),
            'has_mass':       rec['mass_df'] is not None,
            'has_volume':     rec['volume_df'] is not None,
            'has_pairing':    rec['pairing_df'] is not None,
            'has_bm_gate':    bm is not None,
            'has_ifxm_gate':  ifxm is not None,
            'has_images':     rec['hdf5_src_path'] is not None,
            'bm_gate_lower':  bm[0] if bm else float('nan'),
            'bm_gate_upper':  bm[1] if bm else float('nan'),
            'ifxm_gate_lower': ifxm[0] if ifxm else float('nan'),
            'ifxm_gate_upper': ifxm[1] if ifxm else float('nan'),
        })

    meta_df = pd.DataFrame(meta_rows)

    with pd.HDFStore(str(h5_path), mode='w') as store:
        store.put('/metadata', meta_df, format='table', data_columns=True)

        for rec in sample_records:
            name = rec['name']
            key_base = f'/samples/{_safe_key(name)}'

            if rec['mass_df'] is not None:
                store.put(f'{key_base}/mass', rec['mass_df'],
                          format='table', data_columns=True)

            if rec['volume_df'] is not None:
                store.put(f'{key_base}/volume', rec['volume_df'],
                          format='table', data_columns=True)

            if rec['pairing_df'] is not None:
                store.put(f'{key_base}/pairing', rec['pairing_df'],
                          format='table', data_columns=True)

    print(f"\nCompiled {len(sample_records)} sample(s) -> {h5_path}")

    image_candidates = [r for r in sample_records
                        if r['hdf5_src_path'] and r['hdf5_index_path']]
    if image_candidates:
        print(f"Writing image stacks for {len(image_candidates)} sample(s) -> {images_h5_path}")
        with h5py.File(str(images_h5_path), 'w') as img_store:
            for rec in image_candidates:
                grp = img_store.require_group(_safe_key(rec['name']))
                n = _save_images_for_sample(
                    rec['hdf5_src_path'], rec['hdf5_index_path'],
                    grp, rec['name'])
                print(f"  [{rec['name']}] {n} transits")

    return out_dir


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    superdir = parse_cli_args()
    print(f"Compiling {superdir.name}...")
    records = compile_experiment(superdir)
    if not records:
        print("No sample data found.")
        sys.exit(1)
    _write_output(superdir, records)


if __name__ == '__main__':
    main()
