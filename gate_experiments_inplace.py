"""
gate_experiment_subfolder.py

Interactive GUI for gating buoyant mass (BM) or iFXM volume data across all
sample subfolders in an experiment superdir. One column per sample is shown in
the GUI. After gating, a YAML file recording the gate bounds is written into
each sample subfolder, and a timestamped summary folder is written into the
superdir.

Workflow:
    1. A data-type selection dialog asks whether you are gating Buoyant Mass
       or iFXM Volume data.
    2. The script discovers the relevant data file for each sample subdir.
       iFXM gates a single histogram (volume_fl); BM gates FOUR independent
       histograms at once — mass_pg, normalized baseline (avg_baseline
       divided by its mean over the first 10% of the run), bl_slope, and
       node_dev_mean — porting the four-attribute gate from this repo's
       MATLAB gate_mass_results.m (see MultiAttributeGatingPanel).
    3. A single window shows the sample list on the left and the
       histogram(s) on the right. Selecting/deselecting samples in the list
       immediately previews their overlaid histogram(s) (shared bin edges).
    4. The user clicks on a histogram to set a lower then upper cutoff;
       changing the left-list selection while doing this restarts the cutoff
       for the new selection. For BM, every one of the four histograms is
       independently clickable at all times (no per-plot "arm" button) and
       gating each is optional — the eventual per-sample gate is the
       logical AND of whichever attribute(s) were actually gated, and each
       gated histogram shows a live "% retained" readout (BM also shows the
       pooled AND retained %). "Apply cutoffs" commits the group (BM:
       confirming first if zero attributes are gated) and returns the
       panel to an idle preview.
    5. Step 3-4 repeats until all samples are assigned. "Finalize gating"
       (highlighted) then becomes available.
    6. On "Finalize gating":
         - A YAML gate file is written into each sample subfolder:
             <sample_subdir_name>_<mode>_gate.yaml
           For BM this includes a `gates:` mapping of every attribute
           actually gated; top-level lower/upper mirror the mass_pg gate
           specifically, for backward compatibility with
           compile_experiment.py's gate reader.
         - A summary folder is written into the superdir:
             <YYMMDD.HHMMSS>_<mode>_gating_summary/
               cutoff_log.txt
               cutoff_stats.csv
               histograms/group_NN.png    (2x2 per attribute, for BM)
    7. A "← Back" button undoes the last group of cutoffs, restoring those
       samples to the remaining list. Can be pressed repeatedly.

On the iFXM page, if a sample already has a BM gate on disk (see
_read_bm_gate_bounds) and matched mass+volume cells (analysis/density/cells
in its CELLGROUPED hdf5), an extra readout shows the % of paired cells
retained by that BM mass_pg bound together with the volume gate being set.

Expected directory structure:

    BM:
        <superdir>/<sample_subdir>/<name>_mass_results/<date>_<name>.csv
        Required columns: mass_pg, avg_baseline, bl_slope, node_dev_mean,
        peak_time_m

    iFXM:
        <superdir>/<sample_subdir>/<YYYYMMDD.HHMMSS>_imaging_fxm_results/
            <sample>_CELLGROUPED.hdf5, table analysis/volume/cells
            (falls back to <sample>_ProcessedVolumes.csv, or
            stage2_analysis/<sample>_ProcessedVolumes.csv, for runs from
            before SMRFXMAnalysis wrote calibrated volumes into the hdf5)
        Target column: volume_fl (hdf5) or volume (legacy csv)

Usage:
    python gate_experiment_subfolder.py <superdir>
"""
import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import yaml

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from gating.common import (GatingPanel, ask_data_type_dialog,
                           save_group_histograms, style_finalize_button,
                           write_stats_csv, write_log)

from fsutil import is_appledouble


# ---------------------------------------------------------------------------
# Mode configuration
# ---------------------------------------------------------------------------

_MODE = {
    'bm': {
        'label':       'Buoyant Mass',
        'unit':        'pg',
        'xlabel':      'Buoyant Mass (pg)',
        'scale':       'linear',
        'bins':        lambda vals: np.linspace(vals.min(), vals.max(), 201),
        'xlim':        None,
        'data_type':    'bm',
        'dir_suffix':   '_bm_gating_summary',
        'yaml_suffix':  '_bm_gate.yaml',
        'yaml_dir_tag': 'bm_gating',
    },
    'ifxm': {
        'label':        'iFXM Volume',
        'unit':         'fL',
        'xlabel':       'Volume (fL)',
        'scale':        'linear',
        'bins':         lambda vals: np.linspace(vals.min(), vals.max(), 201),
        'xlim':         None,
        'data_type':    'ifxm_volume',
        'dir_suffix':   '_ifxm_volume_gating_summary',
        'yaml_suffix':  '_ifxm_volume_gate.yaml',
        'yaml_dir_tag': 'ifxm-vol_gating',
    },
}


# ---------------------------------------------------------------------------
# BM multi-attribute gating (mass, normalized baseline, baseline slope,
# average node deviation) — ports the four-histogram gate from MATLAB's
# gate_mass_results.m (smr_data_analysis) into this script's BM gating page.
# ---------------------------------------------------------------------------

_BM_REQUIRED_COLUMNS = ('mass_pg', 'avg_baseline', 'bl_slope', 'node_dev_mean',
                        'peak_time_m')

_BM_ATTRS = [
    {'key': 'mass_pg', 'label': 'Buoyant mass',
     'xlabel': 'Buoyant mass (pg)'},
    {'key': 'baseline_norm', 'label': 'Normalized baseline',
     'xlabel': 'Normalized baseline (frac. of first-10% mean)', 'derived': True},
    {'key': 'bl_slope', 'label': 'Baseline slope',
     'xlabel': 'Baseline slope'},
    {'key': 'node_dev_mean', 'label': 'Average node deviation',
     'xlabel': 'Avg node deviation'},
]
_ATTR_SPEC = {a['key']: a for a in _BM_ATTRS}
_ATTR_KEYS = [a['key'] for a in _BM_ATTRS]
_ATTR_LABELS = {a['key']: a['label'] for a in _BM_ATTRS}


def _baseline_norm(df: pd.DataFrame) -> np.ndarray:
    """
    avg_baseline normalized to the mean baseline over the first 10% of the
    run (by peak_time_m) — mirrors gate_mass_results.m's values_for_spec for
    the 'baseline_norm' derived quantity. Falls back to the raw avg_baseline
    if there's no usable reference window.
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


def _attr_values(df: pd.DataFrame, attr: dict) -> np.ndarray:
    """The vector of values one BM attribute spec operates on for a sample table."""
    if attr.get('derived'):
        return _baseline_norm(df)
    return df[attr['key']].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_cli_args() -> Path:
    parser = argparse.ArgumentParser(
        description="Interactively gate BM or iFXM data across sample "
                    "subfolders of an experiment superdir."
    )
    parser.add_argument('superdir', type=str,
                        help='Path to the experiment superdir')
    args = parser.parse_args()
    p = Path(args.superdir)
    if not p.is_dir():
        raise FileNotFoundError(f"Directory not found: {p}")
    return p


# ---------------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------------

def _find_latest_mass_csv(sample_dir: Path) -> Path | None:
    """
    The newest *_mass_results dir's mass_pg CSV for one sample subdir (the
    discovery convention shared across this repo's BM tooling), or None.
    """
    run_dir_pattern = re.compile(r'.+_mass_results$')
    run_dirs = sorted(
        d for d in sample_dir.iterdir()
        if d.is_dir() and run_dir_pattern.match(d.name)
    )
    if not run_dirs:
        return None
    run_dir = run_dirs[-1]      # most recent if multiple

    for f in sorted(run_dir.iterdir()):
        if (f.is_file() and f.suffix == '.csv'
                and not is_appledouble(f)
                and not f.name.startswith('curation_index')):
            return f
    return None


def _discover_bm_tables(superdir: Path) -> dict:
    """
    Loads the full per-cell mass CSV (not just mass_pg) for each sample
    subdir, for the multi-attribute BM gating page (mass, normalized
    baseline, baseline slope, average node deviation).

    A sample is included only if its mass CSV has every column the four
    attributes need — mirrors gate_mass_results.m's req_cols check.

    Returns:
        dict: {sample_name (str): pd.DataFrame} (index reset to a plain
              RangeIndex so gate masks line up positionally)
    """
    data = {}
    for sample_dir in sorted(superdir.iterdir()):
        if not sample_dir.is_dir():
            continue
        csv_path = _find_latest_mass_csv(sample_dir)
        if csv_path is None:
            continue
        df = pd.read_csv(csv_path)
        missing = [c for c in _BM_REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            print(f"  [skip] {sample_dir.name}: mass CSV missing column(s): "
                  f"{', '.join(missing)}")
            continue
        if df.empty:
            continue
        data[sample_dir.name] = df.reset_index(drop=True)
    return data


def _read_hdf5_volume_fl(hdf5_path: Path) -> np.ndarray | None:
    """
    Reads the volume_fl column out of a *_CELLGROUPED.hdf5's
    analysis/volume/cells table, keeping only cells whose status_code is
    'ok' (every other status, e.g. no_valid_roi_frames, means volume was not
    successfully computed for that cell and volume_fl is NaN).

    Returns None if the file/table could not be read at all.
    """
    try:
        with h5py.File(hdf5_path, 'r') as f:
            cells = f['analysis/volume/cells'][:]
    except Exception as exc:
        print(f"  [warn] could not read {hdf5_path.name}: {exc}")
        return None
    ok = cells[cells['status_code'] == b'ok']
    vals = ok['volume_fl'].astype(float)
    return vals[~np.isnan(vals)]


def _discover_ifxm(superdir: Path) -> dict:
    """
    Finds iFXM volume data for each sample subdir in superdir.

    Searches two levels deep (superdir → sample_subdir →
    *_imaging_fxm_results/) for a *_CELLGROUPED.hdf5 and reads its
    analysis/volume/cells table's volume_fl column (status_code == 'ok'
    cells only) — the current SMRFXMAnalysis output.

    Runs from before SMRFXMAnalysis wrote calibrated volumes into the hdf5
    instead have a *_ProcessedVolumes.csv (nested under a stage2_analysis/
    subdirectory for runs from before the stage1/stage2 split was dropped);
    this legacy source is used as a fallback when no hdf5 is present.

    Args:
        superdir (Path): experiment superdir

    Returns:
        dict: {sample_name (str): np.ndarray of volume values (fL)}
    """
    run_dir_pattern = re.compile(r'\d{8}\.\d{6}_imaging_fxm_results$')
    data = {}

    for sample_dir in sorted(superdir.iterdir()):
        if not sample_dir.is_dir():
            continue
        run_dirs = sorted(
            d for d in sample_dir.iterdir()
            if d.is_dir() and run_dir_pattern.match(d.name)
        )
        if not run_dirs:
            continue
        run_dir = run_dirs[-1]

        hdf5_files = sorted(
            f for f in run_dir.iterdir()
            if f.is_file() and not is_appledouble(f)
            and f.name.endswith('_CELLGROUPED.hdf5')
        )
        if hdf5_files:
            f = hdf5_files[-1]
            vals = _read_hdf5_volume_fl(f)
            if vals is None:
                continue
            if len(vals) == 0:
                print(f"  [skip] {sample_dir.name}: no 'ok' cells with a volume_fl in {f.name}")
                continue
            data[sample_dir.name] = vals
            continue

        # Legacy fallback: pre-hdf5 runs wrote a ProcessedVolumes.csv instead.
        stage2 = run_dir / 'stage2_analysis'
        if not stage2.is_dir():
            stage2 = run_dir
        csv_found = False
        for f in stage2.iterdir():
            if (f.is_file() and not is_appledouble(f)
                    and f.name.endswith('_ProcessedVolumes.csv')):
                csv_found = True
                df = pd.read_csv(f)
                if 'volume' not in df.columns:
                    print(f"  [skip] {sample_dir.name}: no 'volume' column in {f.name}")
                    break
                vals = df['volume'].dropna().values
                if len(vals) == 0:
                    print(f"  [skip] {sample_dir.name}: 'volume' column is empty/all-NaN in {f.name}")
                    break
                data[sample_dir.name] = vals
                break
        if not csv_found:
            print(f"  [skip] {sample_dir.name}: no CELLGROUPED.hdf5 or _ProcessedVolumes.csv in {run_dir.name}/")

    return data


# ---------------------------------------------------------------------------
# Paired mass+volume retained-% indicator (iFXM gating page)
#
# When a sample has both a BM gate already recorded (by the BM gating page,
# read back off the *_bm_gating/*_bm_gate.yaml written to disk — not
# in-memory state, so this works regardless of which session wrote it) and
# matched mass+volume cells (the CELLGROUPED hdf5's analysis/density/cells
# table, the same source compile_experiment.py pairs from), this powers an
# extra "% of paired data retained" readout on the iFXM gating page: the
# fraction of matched cells whose mass_pg falls in the BM gate AND whose
# volume_fl falls in the volume gate being set right now.
# ---------------------------------------------------------------------------

def _find_cellgrouped_hdf5(sample_dir: Path) -> Path | None:
    """The newest *_imaging_fxm_results run's *_CELLGROUPED.hdf5, or None."""
    run_dir_pattern = re.compile(r'\d{8}\.\d{6}_imaging_fxm_results$')
    run_dirs = sorted(
        d for d in sample_dir.iterdir()
        if d.is_dir() and run_dir_pattern.match(d.name)
    )
    if not run_dirs:
        return None
    run_dir = run_dirs[-1]
    hdf5_files = sorted(
        f for f in run_dir.iterdir()
        if f.is_file() and not is_appledouble(f)
        and f.name.endswith('_CELLGROUPED.hdf5')
    )
    return hdf5_files[-1] if hdf5_files else None


def _load_paired_mass_volume(hdf5_path: Path) -> pd.DataFrame | None:
    """
    mass_pg + volume_fl for every matched cell (status_code == 'ok') in a
    *_CELLGROUPED.hdf5's analysis/density/cells table — the same pairing
    source compile_experiment.py's _load_density_pairing reads. None if the
    file/table can't be read, or this sample never paired (FXM-only run).
    """
    try:
        with h5py.File(hdf5_path, 'r') as f:
            if 'analysis/density/cells' not in f:
                return None
            cells = f['analysis/density/cells'][:]
    except Exception as exc:
        print(f"  [warn] could not read {hdf5_path.name}: {exc}")
        return None
    ok = cells[cells['status_code'] == b'ok']
    if ok.size == 0:
        return None
    return pd.DataFrame({
        'mass_pg':   ok['matched_peak_mass_pg'].astype(float),
        'volume_fl': ok['volume_fl'].astype(float),
    })


def _read_bm_gate_bounds(sample_dir: Path) -> tuple | None:
    """
    The mass_pg (lower, upper) bounds from the newest
    <sample_dir>/*_bm_gating/*_bm_gate.yaml, or None if there isn't one (or
    it has no mass_pg gate — e.g. only the other 3 BM attributes were gated).
    """
    gate_dirs = sorted(
        d for d in sample_dir.iterdir()
        if d.is_dir() and not is_appledouble(d) and d.name.endswith('_bm_gating')
    )
    if not gate_dirs:
        return None
    yaml_files = sorted(f for f in gate_dirs[-1].glob('*.yaml') if not is_appledouble(f))
    if not yaml_files:
        return None
    try:
        payload = yaml.safe_load(yaml_files[0].read_text(encoding='utf-8'))
        return float(payload['lower']), float(payload['upper'])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _write_yaml_files(superdir: Path, sample_dirs: dict,
                      cutoffs: dict, mode_cfg: dict, timestamp: str):
    """
    Creates a gating subdir inside each sample subfolder and writes a YAML
    gate file recording the bounds applied to that sample.

    Subdir:    <sample_subdir>/<YYYYMMDD_HHMMSS>_<mode_tag>/
    File:      <sample_subdir_name>_<mode>_gate.yaml
    Content:
        experiment: <superdir_name>
        data_type:  bm | ifxm_volume
        lower:      <float>
        upper:      <float>
    """
    subdir_name = f"{timestamp}_{mode_cfg['yaml_dir_tag']}"

    for sample, (lo, hi) in cutoffs.items():
        sample_dir = sample_dirs[sample]
        gate_dir = sample_dir / subdir_name
        gate_dir.mkdir(exist_ok=True)
        fname = f"{sample_dir.name}_{mode_cfg['yaml_suffix'].lstrip('_')}"
        out_path = gate_dir / fname
        payload = {
            'experiment': superdir.name,
            'data_type':  mode_cfg['data_type'],
            'lower':      float(lo),
            'upper':      float(hi),
        }
        with open(out_path, 'w') as fh:
            yaml.dump(payload, fh, default_flow_style=False, sort_keys=False)
        print(f"Written: {out_path}")

    print(f"[yaml] {len(cutoffs)} gate file(s) written.")


def _write_output(superdir: Path, sample_dirs: dict, columns: list,
                  data: dict, cutoffs: dict,
                  groups: list, mode_cfg: dict) -> Path:
    timestamp = datetime.now().strftime('%y%m%d.%H%M%S')
    suffix = mode_cfg['dir_suffix']
    summary_dir = superdir / f'{timestamp}{suffix}'
    summary_dir.mkdir()

    hist_dir = summary_dir / 'histograms'
    hist_dir.mkdir()

    ts_yaml = datetime.now().strftime('%Y%m%d_%H%M%S')
    _write_yaml_files(superdir, sample_dirs, cutoffs, mode_cfg, ts_yaml)
    save_group_histograms(hist_dir, data, groups, mode_cfg)

    ts = timestamp  # YYMMDD.HHMMSS
    run_str = f"20{ts[:2]}-{ts[2:4]}-{ts[4:6]} {ts[7:9]}:{ts[9:11]}:{ts[11:13]}"
    header_lines = [
        f"gate_experiment_subfolder — {mode_cfg['label']} Cutoff Log",
        "=" * 60,
        f"Experiment:  {superdir.name}",
        f"Superdir:    {superdir}",
        f"Run:         {run_str}",
    ]
    log_path = summary_dir / 'cutoff_log.txt'
    write_log(log_path, header_lines, data, groups, mode_cfg)

    stats_path = summary_dir / 'cutoff_stats.csv'
    write_stats_csv(stats_path, data, cutoffs, groups, mode_cfg)

    return summary_dir


# ---------------------------------------------------------------------------
# Output — BM multi-attribute (mass / normalized baseline / baseline slope /
# average node deviation)
# ---------------------------------------------------------------------------

def _bm_multi_mask(df: pd.DataFrame, ranges: dict) -> np.ndarray:
    """Boolean AND of every attribute range set in `ranges` ({key: (lo, hi)})."""
    mask = np.ones(len(df), dtype=bool)
    for key, (lo, hi) in ranges.items():
        v = _attr_values(df, _ATTR_SPEC[key])
        mask &= (v >= lo) & (v <= hi)
    return mask


def _write_bm_multi_yaml_files(superdir: Path, sample_dirs: dict,
                               cutoffs: dict, timestamp: str):
    """
    Like _write_yaml_files, but for the multi-attribute BM page: `cutoffs`
    maps sample -> {attr_key: (lower, upper)} (0-4 entries; an attribute not
    gated for a sample is simply absent). Written under the same
    <sample_subdir>/<timestamp>_bm_gating/<sample>_bm_gate.yaml path
    compile_experiment.py already looks for, so it keeps working unchanged.

    Content:
        experiment: <superdir_name>
        data_type:  bm
        lower/upper:  the mass_pg gate specifically, kept at the top level
                      for backward compatibility with compile_experiment.py's
                      _load_gate (which reads exactly these two keys) — only
                      present if mass_pg was one of the gated attributes.
        gates:        {attr_key: {lower, upper}, ...} for every attribute
                      actually gated for that sample (mass_pg included).
    """
    mode_cfg = _MODE['bm']
    subdir_name = f"{timestamp}_{mode_cfg['yaml_dir_tag']}"

    for sample, ranges in cutoffs.items():
        sample_dir = sample_dirs[sample]
        gate_dir = sample_dir / subdir_name
        gate_dir.mkdir(exist_ok=True)
        fname = f"{sample_dir.name}_{mode_cfg['yaml_suffix'].lstrip('_')}"
        out_path = gate_dir / fname
        payload = {
            'experiment': superdir.name,
            'data_type':  mode_cfg['data_type'],
        }
        if 'mass_pg' in ranges:
            lo, hi = ranges['mass_pg']
            payload['lower'] = float(lo)
            payload['upper'] = float(hi)
        if ranges:
            payload['gates'] = {
                key: {'lower': float(lo), 'upper': float(hi)}
                for key, (lo, hi) in ranges.items()
            }
        with open(out_path, 'w') as fh:
            yaml.dump(payload, fh, default_flow_style=False, sort_keys=False)
        print(f"Written: {out_path}")

    print(f"[yaml] {len(cutoffs)} gate file(s) written.")


def _save_bm_multi_group_histograms(hist_dir: Path, data: dict, groups: list):
    """One 2x2 PNG per cutoff group — all four attributes, mirroring the
    gate_mass_results.m layout — with cutoff lines drawn for whichever
    attributes were actually gated in that group."""
    for i, (ranges, names) in enumerate(groups, 1):
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        for ax, spec in zip(axs.flat, _BM_ATTRS):
            key = spec['key']
            arrays = []
            for name in names:
                v = _attr_values(data[name], spec)
                v = v[np.isfinite(v)]
                if v.size:
                    arrays.append((name, v))
            if not arrays:
                ax.axis('off')
                continue

            pooled = np.concatenate([v for _, v in arrays])
            lo, hi = float(pooled.min()), float(pooled.max())
            if key in ranges:
                _glo, _ghi, view = ranges[key]
                if view is not None:
                    lo, hi = view
            if hi <= lo:
                hi = lo + 1
            edges = np.linspace(lo, hi, 101)

            for name, v in arrays:
                ax.hist(v, bins=edges, alpha=0.5, edgecolor='black',
                        linewidth=0.3, label=name)
            if key in ranges:
                glo, ghi, _view = ranges[key]
                ax.axvline(glo, color='red', linestyle='--', linewidth=1.2,
                          label=f'lower = {glo:.4g}')
                ax.axvline(ghi, color='steelblue', linestyle='--', linewidth=1.2,
                          label=f'upper = {ghi:.4g}')
                ax.axvspan(glo, ghi, alpha=0.1, color='green')
            ax.set_xlim(lo, hi)
            ax.set_xlabel(spec['xlabel'], fontsize=9)
            ax.set_ylabel('count')
            ax.set_title(spec['label'], fontsize=10)
            ax.legend(fontsize=6, loc='upper right')

        fig.suptitle(f'Group {i}: {", ".join(names)}', fontsize=10)
        fig.tight_layout()
        out_path = hist_dir / f'group_{i:02d}.png'
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Written: {out_path}")


def _write_bm_multi_log(log_path: Path, superdir: Path, data: dict, groups: list):
    lines = [
        "gate_experiments_inplace — Buoyant Mass (multi-attribute) Cutoff Log",
        "=" * 70,
        f"Experiment:  {superdir.name}",
        f"Superdir:    {superdir}",
        f"Run:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Cutoff groups",
        "-" * 70,
    ]
    total_before = total_after = 0
    for i, (ranges, names) in enumerate(groups, 1):
        if ranges:
            range_desc = '; '.join(
                f"{_ATTR_LABELS[k]}: {lo:.4g} - {hi:.4g}"
                for k, (lo, hi, _v) in ranges.items())
        else:
            range_desc = '(no gates set — all points accepted)'
        lines.append(f"Group {i}   {range_desc}")

        flat_ranges = {k: (lo, hi) for k, (lo, hi, _v) in ranges.items()}
        for name in names:
            df = data[name]
            n_before = len(df)
            n_after = int(np.count_nonzero(_bm_multi_mask(df, flat_ranges)))
            n_removed = n_before - n_after
            pct = 100 * n_removed / n_before if n_before else 0.0
            total_before += n_before
            total_after += n_after
            lines.append(f"  {name:<40s}  {n_before} -> {n_after}"
                         f"  ({n_removed} removed, {pct:.1f}%)")
        lines.append("")

    total_removed = total_before - total_after
    total_pct = 100 * total_removed / total_before if total_before else 0.0
    n_items = sum(len(names) for _, names in groups)
    lines.append(
        f"Total: {total_before} -> {total_after} rows retained across "
        f"{n_items} item(s)  ({total_removed} removed, {total_pct:.1f}%)")

    log_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f"Written: {log_path}")


def _write_bm_multi_stats_csv(stats_path: Path, data: dict, cutoffs: dict, groups: list):
    rows = []
    for _ranges, names in groups:
        for name in names:
            df = data[name]
            flat_ranges = cutoffs.get(name, {})
            mask = _bm_multi_mask(df, flat_ranges)
            mass = df['mass_pg'].to_numpy(dtype=float)[mask]
            mass = mass[np.isfinite(mass)]

            row = {'sample': name, 'n_before': len(df),
                  'n_after': int(np.count_nonzero(mask))}
            for key in _ATTR_KEYS:
                lo, hi = flat_ranges.get(key, (np.nan, np.nan))
                row[f'{key}_lower'] = lo
                row[f'{key}_upper'] = hi
            if mass.size:
                row['mass_mean'] = float(mass.mean())
                row['mass_median'] = float(np.median(mass))
                row['mass_std'] = float(mass.std())
            else:
                row['mass_mean'] = row['mass_median'] = row['mass_std'] = np.nan
            rows.append(row)

    pd.DataFrame(rows).to_csv(stats_path, index=False)
    print(f"Written: {stats_path}")


def _write_bm_multi_output(superdir: Path, sample_dirs: dict, data: dict,
                          cutoffs: dict, groups: list) -> Path:
    mode_cfg = _MODE['bm']
    timestamp = datetime.now().strftime('%y%m%d.%H%M%S')
    summary_dir = superdir / f'{timestamp}{mode_cfg["dir_suffix"]}'
    summary_dir.mkdir()

    hist_dir = summary_dir / 'histograms'
    hist_dir.mkdir()

    ts_yaml = datetime.now().strftime('%Y%m%d_%H%M%S')
    _write_bm_multi_yaml_files(superdir, sample_dirs, cutoffs, ts_yaml)
    _save_bm_multi_group_histograms(hist_dir, data, groups)
    _write_bm_multi_log(summary_dir / 'cutoff_log.txt', superdir, data, groups)
    _write_bm_multi_stats_csv(summary_dir / 'cutoff_stats.csv', data, cutoffs, groups)

    return summary_dir


# ---------------------------------------------------------------------------
# MultiAttributeGatingPanel — BM gating (mass / normalized baseline /
# baseline slope / average node deviation)
# ---------------------------------------------------------------------------

class MultiAttributeGatingPanel(tk.Frame):
    """
    BM gating page: same left sample-list + Back/Finalize gating layout as
    GatingPanel, but the right side is a 2x2 grid of independent
    histogram/cutoff panels — one per attribute (mass, normalized baseline,
    baseline slope, average node deviation) — ported from MATLAB's
    gate_mass_results.m. Each attribute panel is always "live": clicking on
    it sets its own lower then upper cutoff without any per-plot activation
    button, and selecting/deselecting samples on the left immediately
    updates all four histograms (as in GatingPanel).

    Unlike the MATLAB version, gating an attribute is entirely optional —
    "Apply cutoffs" commits the group using whichever 0-4 attributes were
    actually gated (confirming first if none were); the final per-sample
    gate is the logical AND of whichever range(s) were set.

    Every attribute panel shows a live "Retained: X%" readout once its own
    cutoff is set (its data only), and the bottom bar shows the pooled
    percentage retained by the AND of every attribute currently gated.

    Args:
        parent:        parent tkinter widget
        columns:       ordered list of all sample names
        data:          mapping of sample name -> per-cell pd.DataFrame
                       (must carry every column in _BM_REQUIRED_COLUMNS)
        on_finish:     callable(cutoffs, groups) -> Path | str | None, called
                       on Finalize gating; cutoffs maps sample ->
                       {attr_key: (lower, upper)}, groups is an ordered list
                       of ({attr_key: (lower, upper, view)}, [sample, ...]).
        context_label: text appended to the header, e.g. the superdir name
        on_done:       optional callable() invoked after the Finalize dialog
                       is dismissed (e.g. root.destroy standalone; a wizard
                       "mark this step complete" callback when embedded).
    """

    def __init__(self, parent: tk.Widget, columns: list, data: dict,
                on_finish, context_label: str = '', on_done=None):
        super().__init__(parent)
        self._columns = columns
        self._data = data
        self._on_finish = on_finish
        self._on_done = on_done
        self._cutoffs: dict = {}
        self._groups: list = []
        self._remaining: list = list(columns)
        self._history: list = []
        self._active_selection: list | None = None

        ctx = f"  [{context_label}]" if context_label else ''
        self.title = f"Gating — Buoyant Mass (mass, baseline, slope, node dev){ctx}"

        left = tk.Frame(self)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        right = tk.Frame(self)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- left: sample list ---
        self._header_var = tk.StringVar()
        tk.Label(left, textvariable=self._header_var,
                font=('TkDefaultFont', 11, 'bold'),
                anchor='w').pack(fill=tk.X, pady=(10, 4))

        list_frame = tk.Frame(left)
        list_frame.pack(fill=tk.BOTH, expand=True)
        scrollbar = tk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self._listbox = tk.Listbox(
            list_frame, selectmode=tk.MULTIPLE, exportselection=False,
            yscrollcommand=scrollbar.set, height=24, width=32)
        scrollbar.config(command=self._listbox.yview)
        self._listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._listbox.bind('<<ListboxSelect>>', self._on_select)

        btn_frame = tk.Frame(left)
        btn_frame.pack(fill=tk.X, pady=8)
        self._back_btn = tk.Button(btn_frame, text="← Back",
                                   state=tk.DISABLED, command=self._do_back)
        self._back_btn.pack(side=tk.LEFT)
        self._done_btn = tk.Button(btn_frame, text="Finalize gating",
                                   command=self._finish)
        self._done_btn.pack(side=tk.LEFT, padx=(8, 0))
        style_finalize_button(self._done_btn, False)

        # --- right: 2x2 attribute grid ---
        grid = tk.Frame(right)
        grid.pack(fill=tk.BOTH, expand=True)
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        grid.rowconfigure(0, weight=1)
        grid.rowconfigure(1, weight=1)

        self._attr: dict = {}
        for idx, spec in enumerate(_BM_ATTRS):
            r, c = divmod(idx, 2)
            state = self._build_attr_cell(grid, spec)
            state['frame'].grid(row=r, column=c, sticky='nsew', padx=3, pady=3)
            self._attr[spec['key']] = state

        bottom = tk.Frame(right)
        bottom.pack(fill=tk.X, pady=(6, 0))
        self._overall_var = tk.StringVar()
        tk.Label(bottom, textvariable=self._overall_var, anchor='w',
                font=('TkDefaultFont', 10, 'bold')).pack(side=tk.LEFT)
        self._apply_btn = tk.Button(bottom, text='Apply cutoffs',
                                    state=tk.DISABLED, command=self._apply_cutoffs)
        self._apply_btn.pack(side=tk.RIGHT)

        for key in _ATTR_KEYS:
            self._draw_idle_attr(key)
        self._update_overall_retained()
        self._refresh_list()

    # -- attribute cell construction -------------------------------------

    def _build_attr_cell(self, parent: tk.Widget, spec: dict) -> dict:
        key = spec['key']
        cell = tk.Frame(parent, bd=1, relief=tk.GROOVE)
        tk.Label(cell, text=spec['label'],
                font=('TkDefaultFont', 9, 'bold')).pack(anchor='w', padx=4, pady=(2, 0))

        # Figure() directly, not plt.subplots(): the latter creates an
        # extra hidden Tk root per figure via pyplot's stateful backend
        # (unused — we embed via our own FigureCanvasTkAgg below) that's
        # never torn down. With 4 of these per BM panel, that's enough
        # orphaned roots to keep the whole process alive after the app's
        # real window is closed. Figure() never touches Tkinter.
        fig = Figure(figsize=(4.6, 2.8))
        ax = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=cell)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=2)
        canvas.mpl_connect('button_press_event',
                           lambda e, k=key: self._on_attr_click(k, e))

        view_row = tk.Frame(cell)
        view_row.pack(fill=tk.X, padx=4, pady=(2, 0))
        tk.Label(view_row, text='x:').pack(side=tk.LEFT)
        xmin_var = tk.StringVar()
        tk.Entry(view_row, textvariable=xmin_var, width=8).pack(side=tk.LEFT, padx=(2, 2))
        xmax_var = tk.StringVar()
        tk.Entry(view_row, textvariable=xmax_var, width=8).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(view_row, text='Apply',
                 command=lambda k=key: self._apply_attr_xlim(k)).pack(side=tk.LEFT)
        tk.Button(view_row, text='Reset view',
                 command=lambda k=key: self._reset_attr_xlim(k)).pack(side=tk.LEFT, padx=(4, 0))
        tk.Button(view_row, text='Clear gate',
                 command=lambda k=key: self._reset_attr_gate(k)).pack(side=tk.RIGHT)

        status_var = tk.StringVar()
        tk.Label(cell, textvariable=status_var, anchor='w',
                font=('TkDefaultFont', 8)).pack(fill=tk.X, padx=4)
        retained_var = tk.StringVar()
        tk.Label(cell, textvariable=retained_var, anchor='w',
                font=('TkDefaultFont', 8, 'bold'),
                foreground='#1a6b1a').pack(fill=tk.X, padx=4, pady=(0, 2))

        return {
            'frame': cell, 'fig': fig, 'ax': ax, 'canvas': canvas,
            'xmin_var': xmin_var, 'xmax_var': xmax_var,
            'status_var': status_var, 'retained_var': retained_var,
            'lower': None, 'upper': None, 'state': 0,
            'view': None, 'full_xlim': None,
        }

    # -- left: sample list -------------------------------------------------

    def _refresh_list(self):
        self._listbox.delete(0, tk.END)
        for col in self._remaining:
            self._listbox.insert(tk.END, col)
        n_done = len(self._cutoffs)
        n_total = len(self._columns)
        self._header_var.set(f"Remaining: {n_total - n_done} / {n_total}   "
                             f"[Buoyant Mass]")
        style_finalize_button(self._done_btn, n_done == n_total)
        self._back_btn.config(state=tk.NORMAL if self._history else tk.DISABLED)

    def _on_select(self, _event):
        indices = self._listbox.curselection()
        if not indices:
            self._clear_active_gating()
            return
        self._activate_gating([self._remaining[i] for i in indices])

    def _do_back(self):
        if not self._history:
            return
        self._cutoffs, self._groups, self._remaining = self._history.pop()
        self._clear_active_gating()
        self._refresh_list()

    def _finish(self):
        result = self._on_finish(self._cutoffs, self._groups)
        msg = f"Output written to:\n{result}" if result else "Done."
        messagebox.showinfo("Done", msg, parent=self.winfo_toplevel())
        if self._on_done is not None:
            self._on_done()

    # -- selection lifecycle ------------------------------------------------

    def _activate_gating(self, selection: list):
        self._active_selection = selection
        for key in _ATTR_KEYS:
            st = self._attr[key]
            st['lower'] = None
            st['upper'] = None
            st['state'] = 0
            st['view'] = None
            st['xmin_var'].set('')
            st['xmax_var'].set('')
            st['status_var'].set('Click to set lower cutoff.')
            self._draw_attr_histograms(key)
            self._update_attr_retained(key)
        self._apply_btn.config(state=tk.NORMAL)
        self._update_overall_retained()

    def _clear_active_gating(self):
        self._active_selection = None
        for key in _ATTR_KEYS:
            st = self._attr[key]
            st['lower'] = None
            st['upper'] = None
            st['state'] = 0
            st['view'] = None
            st['status_var'].set('')
            self._draw_idle_attr(key)
            self._update_attr_retained(key)
        self._apply_btn.config(state=tk.DISABLED)
        self._update_overall_retained()

    def _apply_cutoffs(self):
        if self._active_selection is None:
            return
        set_keys = [k for k in _ATTR_KEYS if self._attr[k]['state'] == 2]
        if not set_keys:
            if not messagebox.askyesno(
                    "No cutoffs set",
                    f"No cutoffs are set for any attribute. Mark the "
                    f"{len(self._active_selection)} selected sample(s) as "
                    f"done without any gate?",
                    parent=self.winfo_toplevel()):
                return

        self._history.append((
            self._cutoffs.copy(), list(self._groups), list(self._remaining)))

        ranges = {k: (self._attr[k]['lower'], self._attr[k]['upper'], self._attr[k]['view'])
                 for k in set_keys}
        flat_ranges = {k: (lo, hi) for k, (lo, hi, _v) in ranges.items()}
        for name in self._active_selection:
            self._cutoffs[name] = dict(flat_ranges)
            self._remaining.remove(name)
        self._groups.append((ranges, list(self._active_selection)))

        self._clear_active_gating()
        self._refresh_list()

    # -- per-attribute histogram / cutoff -----------------------------------

    def _draw_idle_attr(self, key: str):
        st = self._attr[key]
        ax = st['ax']
        ax.clear()
        ax.text(0.5, 0.5, 'Select samples\non the left', ha='center', va='center',
                transform=ax.transAxes, color='#888888', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        st['fig'].tight_layout()
        st['canvas'].draw()

    def _draw_attr_histograms(self, key: str, xlim: tuple = None):
        if self._active_selection is None:
            self._draw_idle_attr(key)
            return

        spec = _ATTR_SPEC[key]
        st = self._attr[key]
        ax = st['ax']
        ax.clear()

        arrays = []
        for name in self._active_selection:
            v = _attr_values(self._data[name], spec)
            v = v[np.isfinite(v)]
            if v.size:
                arrays.append((name, v))
        if not arrays:
            self._draw_idle_attr(key)
            return

        pooled = np.concatenate([v for _, v in arrays])
        st['full_xlim'] = (float(pooled.min()), float(pooled.max()))
        if not st['xmin_var'].get() and not st['xmax_var'].get():
            st['xmin_var'].set(f"{st['full_xlim'][0]:.4g}")
            st['xmax_var'].set(f"{st['full_xlim'][1]:.4g}")

        lo, hi = xlim if xlim is not None else st['full_xlim']
        if hi <= lo:
            hi = lo + 1
        edges = np.linspace(lo, hi, 101)

        for name, v in arrays:
            ax.hist(v, bins=edges, alpha=0.5, edgecolor='black',
                    linewidth=0.3, label=name)
        ax.set_xlim(lo, hi)
        ax.set_xlabel(spec['xlabel'], fontsize=8)
        ax.set_ylabel('count', fontsize=8)
        ax.tick_params(labelsize=7)
        if key == _ATTR_KEYS[0] and len(arrays) > 1:
            ax.legend(fontsize=6, loc='upper right')
        st['fig'].tight_layout()

        self._draw_attr_cutoffs(key)
        st['canvas'].draw()

    def _draw_attr_cutoffs(self, key: str):
        st = self._attr[key]
        ax = st['ax']
        y_top = ax.get_ylim()[1]
        if st['lower'] is not None:
            ax.axvline(st['lower'], color='red', linestyle='--', linewidth=1.0)
            ax.text(st['lower'], y_top, f"{st['lower']:.3g}", color='red',
                    fontsize=7, va='top', ha='right')
        if st['upper'] is not None:
            ax.axvline(st['upper'], color='steelblue', linestyle='--', linewidth=1.0)
            ax.text(st['upper'], y_top, f"{st['upper']:.3g}", color='steelblue',
                    fontsize=7, va='top', ha='left')
        if st['lower'] is not None and st['upper'] is not None:
            ax.axvspan(st['lower'], st['upper'], alpha=0.12, color='green')

    def _current_attr_xlim(self, key: str) -> tuple:
        st = self._attr[key]
        try:
            lo = float(st['xmin_var'].get())
            hi = float(st['xmax_var'].get())
        except ValueError:
            st['status_var'].set('x view: min/max must be numbers.')
            return None
        if hi <= lo:
            st['status_var'].set('x view: max must be > min.')
            return None
        return lo, hi

    def _apply_attr_xlim(self, key: str):
        if self._active_selection is None:
            return
        xlim = self._current_attr_xlim(key)
        if xlim is None:
            return
        self._attr[key]['view'] = xlim
        self._draw_attr_histograms(key, xlim=xlim)

    def _reset_attr_xlim(self, key: str):
        st = self._attr[key]
        if self._active_selection is None or st['full_xlim'] is None:
            return
        lo, hi = st['full_xlim']
        st['xmin_var'].set(f'{lo:.4g}')
        st['xmax_var'].set(f'{hi:.4g}')
        st['view'] = None
        self._draw_attr_histograms(key)

    def _reset_attr_gate(self, key: str):
        if self._active_selection is None:
            return
        st = self._attr[key]
        st['lower'] = None
        st['upper'] = None
        st['state'] = 0
        st['status_var'].set('Click to set lower cutoff.')
        self._draw_attr_histograms(key, xlim=st['view'])
        self._update_attr_retained(key)
        self._update_overall_retained()

    def _on_attr_click(self, key: str, event):
        if self._active_selection is None:
            return
        if event.inaxes is None or event.xdata is None:
            return
        st = self._attr[key]
        x = event.xdata

        if st['state'] == 0:
            st['lower'] = x
            st['state'] = 1
            st['status_var'].set(f"Lower: {x:.4g} — click upper cutoff")
        elif st['state'] == 1:
            if x <= st['lower']:
                st['status_var'].set("Upper must be > lower. Click again.")
                return
            st['upper'] = x
            st['state'] = 2
            st['status_var'].set(f"Lower: {st['lower']:.4g}  Upper: {x:.4g}")
        else:
            return

        self._draw_attr_cutoffs(key)
        st['canvas'].draw()
        self._update_attr_retained(key)
        self._update_overall_retained()

    def _update_attr_retained(self, key: str):
        st = self._attr[key]
        if (self._active_selection is None or st['lower'] is None
                or st['upper'] is None):
            st['retained_var'].set('')
            return
        lo, hi = st['lower'], st['upper']
        spec = _ATTR_SPEC[key]
        total = 0
        kept = 0
        for name in self._active_selection:
            v = _attr_values(self._data[name], spec)
            total += v.size
            kept += int(np.count_nonzero((v >= lo) & (v <= hi)))
        pct = 100 * kept / total if total else 0.0
        st['retained_var'].set(f'Retained: {pct:.1f}%  ({kept}/{total})')

    def _update_overall_retained(self):
        if self._active_selection is None:
            self._overall_var.set('')
            return
        set_keys = [k for k in _ATTR_KEYS if self._attr[k]['state'] == 2]
        total = 0
        kept = 0
        for name in self._active_selection:
            df = self._data[name]
            n = len(df)
            total += n
            if not set_keys:
                kept += n
                continue
            flat_ranges = {k: (self._attr[k]['lower'], self._attr[k]['upper'])
                          for k in set_keys}
            kept += int(np.count_nonzero(_bm_multi_mask(df, flat_ranges)))
        pct = 100 * kept / total if total else 0.0
        self._overall_var.set(
            f'Gates set: {len(set_keys)}/4    '
            f'Overall retained (AND): {pct:.1f}%  ({kept}/{total})')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    superdir = parse_cli_args()

    root = tk.Tk()
    root.withdraw()
    mode_key = ask_data_type_dialog(root, [
        ('Buoyant Mass', 'bm'),
        ('iFXM Volume', 'ifxm'),
    ])

    if mode_key == 'bm':
        print(f"Discovering Buoyant Mass data in {superdir.name}...")
        data = _discover_bm_tables(superdir)
        if not data:
            print(f"No Buoyant Mass data found in {superdir}")
            sys.exit(1)
        sample_dirs = {name: superdir / name for name in data}
        columns = list(data.keys())
        print(f"Found {len(columns)} sample(s): {', '.join(columns)}")

        def on_finish(cutoffs, groups):
            return _write_bm_multi_output(superdir, sample_dirs, data, cutoffs, groups)

        root.deiconify()
        panel = MultiAttributeGatingPanel(root, columns, data, on_finish,
                                          context_label=superdir.name,
                                          on_done=root.destroy)
    else:
        mode_cfg = _MODE[mode_key]
        print(f"Discovering {mode_cfg['label']} data in {superdir.name}...")
        data = _discover_ifxm(superdir)
        if not data:
            print(f"No {mode_cfg['label']} data found in {superdir}")
            sys.exit(1)
        sample_dirs = {name: superdir / name for name in data}
        columns = list(data.keys())
        print(f"Found {len(columns)} sample(s): {', '.join(columns)}")

        def on_finish(cutoffs, groups):
            return _write_output(superdir, sample_dirs, columns,
                                 data, cutoffs, groups, mode_cfg)

        root.deiconify()
        panel = GatingPanel(root, columns, data, mode_cfg, on_finish,
                            context_label=superdir.name, on_done=root.destroy)

    root.title(panel.title)
    panel.pack(fill=tk.BOTH, expand=True)
    root.mainloop()


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def build_embedded_page(parent: tk.Widget, mode_key: str, *,
                        initial_superdir: str | None = None,
                        superdir_var: tk.StringVar | None = None,
                        on_finalized=None, on_discover=None) -> ttk.Frame:
    """
    Build this tool's gating UI (for a single data type, 'bm' or 'ifxm') as a
    Frame suitable for embedding in a larger application, e.g. a wizard page.

    Unlike the standalone main(), the data-type is fixed by `mode_key` rather
    than asked via ask_data_type_dialog (a wizard page is already scoped to
    one mode), and there is a directory field + "Discover" button in place of
    the CLI's positional superdir argument. Discovery also runs automatically
    whenever the directory field resolves to a *new* valid directory —
    whether typed here, chosen via Browse, or (via a shared `superdir_var`)
    set from another page entirely — so a wizard step ahead of this one
    setting the directory is enough; the button remains only for an explicit
    manual re-discover (e.g. after files changed on disk without the path
    itself changing). Either way, (re)running discovery (re)builds the
    gating panel underneath from scratch, so changing directories after
    gating has started simply starts over.

    Pass an existing `superdir_var` (rather than `initial_superdir`) to bind
    this page's directory field to a StringVar shared with another page —
    e.g. the BM and iFXM gating pages of a wizard sharing one directory.

    `on_finalized`, if given, is called once each time the embedded panel's
    "Finalize gating" is clicked (i.e. its own on_done) — e.g. a wizard uses
    this to enable its Next button. `on_discover`, if given, is called at the
    start of every discovery run — whether from the button or automatic —
    before this page's own state is reset — e.g. the same wizard uses this
    to re-disable Next, since a fresh discovery means whatever was
    previously finalized here no longer applies.
    """
    if mode_key == 'bm':
        return _build_bm_embedded_page(parent, initial_superdir=initial_superdir,
                                       superdir_var=superdir_var,
                                       on_finalized=on_finalized,
                                       on_discover=on_discover)
    return _build_ifxm_embedded_page(parent, initial_superdir=initial_superdir,
                                     superdir_var=superdir_var,
                                     on_finalized=on_finalized,
                                     on_discover=on_discover)


def _build_bm_embedded_page(parent: tk.Widget, *, initial_superdir=None,
                            superdir_var=None, on_finalized=None,
                            on_discover=None) -> ttk.Frame:
    page = ttk.Frame(parent)

    top = ttk.Frame(page)
    top.pack(fill=tk.X, padx=8, pady=8)
    top.columnconfigure(1, weight=1)

    ttk.Label(top, text='Experiment superdir:').grid(row=0, column=0, sticky='w')
    if superdir_var is None:
        superdir_var = tk.StringVar(value=initial_superdir or '')
    ttk.Entry(top, textvariable=superdir_var).grid(
        row=0, column=1, sticky='ew', padx=(6, 6))

    def _browse():
        chosen = filedialog.askdirectory(
            title='Select experiment superdir',
            initialdir=superdir_var.get() or None)
        if chosen:
            superdir_var.set(chosen)

    ttk.Button(top, text='Browse…', command=_browse).grid(row=0, column=2)

    status_var = tk.StringVar(value='')
    ttk.Label(page, textvariable=status_var, foreground='#a00').pack(fill=tk.X, padx=8)

    panel_container = ttk.Frame(page)
    panel_container.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))

    def _discover():
        if on_discover is not None:
            on_discover()
        for child in panel_container.winfo_children():
            child.destroy()
        status_var.set('')

        superdir_text = superdir_var.get().strip()
        if not superdir_text:
            status_var.set('Choose an experiment superdir first.')
            return
        superdir = Path(superdir_text)
        if not superdir.is_dir():
            status_var.set(f'Directory not found: {superdir}')
            return

        data = _discover_bm_tables(superdir)
        if not data:
            status_var.set(f'No Buoyant Mass data found under {superdir}.')
            return

        sample_dirs = {name: superdir / name for name in data}
        columns = list(data.keys())

        def on_finish(cutoffs, groups):
            out_dir = _write_bm_multi_output(superdir, sample_dirs, data, cutoffs, groups)
            status_var.set(f'Gate files written to: {out_dir}')
            return out_dir

        gating_panel = MultiAttributeGatingPanel(
            panel_container, columns, data, on_finish,
            context_label=superdir.name, on_done=on_finalized)
        gating_panel.pack(fill=tk.BOTH, expand=True)

    ttk.Button(top, text='Discover Buoyant Mass data',
              command=_discover).grid(row=1, column=1, sticky='w', pady=(8, 0))

    # Auto-discover once a *new* valid directory is in place — whether typed
    # here, chosen via Browse, or set from another shared-var page (e.g. a
    # wizard's earlier step) — so this page never requires an extra click
    # just because the directory came from elsewhere. Gated on the resolved
    # directory actually changing (not every keystroke) so it doesn't
    # discard in-progress gating over an incidental edit that leaves the
    # effective directory the same; the button above remains for an
    # explicit manual re-discover (e.g. after files changed on disk without
    # the path itself changing).
    _last_auto_dir = {'value': None}

    def _auto_discover(*_a):
        text = superdir_var.get().strip()
        if text and Path(text).is_dir() and text != _last_auto_dir['value']:
            _last_auto_dir['value'] = text
            _discover()

    superdir_var.trace_add('write', _auto_discover)
    _auto_discover()

    return page


def _build_ifxm_embedded_page(parent: tk.Widget, *, initial_superdir=None,
                              superdir_var=None, on_finalized=None,
                              on_discover=None) -> ttk.Frame:
    mode_cfg = _MODE['ifxm']
    page = ttk.Frame(parent)

    top = ttk.Frame(page)
    top.pack(fill=tk.X, padx=8, pady=8)
    top.columnconfigure(1, weight=1)

    ttk.Label(top, text='Experiment superdir:').grid(row=0, column=0, sticky='w')
    if superdir_var is None:
        superdir_var = tk.StringVar(value=initial_superdir or '')
    ttk.Entry(top, textvariable=superdir_var).grid(
        row=0, column=1, sticky='ew', padx=(6, 6))

    def _browse():
        chosen = filedialog.askdirectory(
            title='Select experiment superdir',
            initialdir=superdir_var.get() or None)
        if chosen:
            superdir_var.set(chosen)

    ttk.Button(top, text='Browse…', command=_browse).grid(row=0, column=2)

    status_var = tk.StringVar(value='')
    ttk.Label(page, textvariable=status_var, foreground='#a00').pack(fill=tk.X, padx=8)

    paired_var = tk.StringVar(value='')
    tk.Label(page, textvariable=paired_var, anchor='w',
            font=('TkDefaultFont', 9, 'bold'), foreground='#1a6b1a').pack(
        fill=tk.X, padx=8)

    panel_container = ttk.Frame(page)
    panel_container.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))

    def _discover():
        if on_discover is not None:
            on_discover()
        for child in panel_container.winfo_children():
            child.destroy()
        status_var.set('')
        paired_var.set('')

        superdir_text = superdir_var.get().strip()
        if not superdir_text:
            status_var.set('Choose an experiment superdir first.')
            return
        superdir = Path(superdir_text)
        if not superdir.is_dir():
            status_var.set(f'Directory not found: {superdir}')
            return

        data = _discover_ifxm(superdir)
        if not data:
            status_var.set(f'No {mode_cfg["label"]} data found under {superdir}.')
            return

        sample_dirs = {name: superdir / name for name in data}
        columns = list(data.keys())

        # Pre-load, per sample, whatever's needed for the paired mass+volume
        # indicator: a BM gate already on disk (from the BM gating page —
        # possibly a previous session's, that's fine) and a matched
        # mass+volume table. Only samples with both contribute.
        paired_info = {}
        for name in columns:
            bounds = _read_bm_gate_bounds(sample_dirs[name])
            if bounds is None:
                continue
            hdf5_path = _find_cellgrouped_hdf5(sample_dirs[name])
            if hdf5_path is None:
                continue
            paired_df = _load_paired_mass_volume(hdf5_path)
            if paired_df is None or paired_df.empty:
                continue
            paired_info[name] = (bounds, paired_df)

        def on_finish(cutoffs, groups):
            out_dir = _write_output(superdir, sample_dirs, columns,
                                    data, cutoffs, groups, mode_cfg)
            status_var.set(f'Gate files written to: {out_dir}')
            return out_dir

        def _update_paired_indicator():
            if not paired_info:
                paired_var.set('')
                return
            selection, lower, upper = gating_panel.get_active_gate()
            if selection is None or lower is None or upper is None:
                paired_var.set('')
                return
            relevant = [s for s in selection if s in paired_info]
            if not relevant:
                paired_var.set('Paired mass+volume retained: n/a for this selection')
                return
            total = 0
            kept = 0
            for name in relevant:
                (mlo, mhi), pdf = paired_info[name]
                mass = pdf['mass_pg'].to_numpy(dtype=float)
                vol = pdf['volume_fl'].to_numpy(dtype=float)
                total += len(pdf)
                kept += int(np.count_nonzero(
                    (mass >= mlo) & (mass <= mhi) & (vol >= lower) & (vol <= upper)))
            pct = 100 * kept / total if total else 0.0
            paired_var.set(
                f'Paired mass+volume retained (BM gate x this volume gate): '
                f'{pct:.1f}%  ({kept}/{total})')

        gating_panel = GatingPanel(panel_container, columns, data, mode_cfg,
                                   on_finish, context_label=superdir.name,
                                   on_done=on_finalized,
                                   on_gate_change=_update_paired_indicator)
        gating_panel.pack(fill=tk.BOTH, expand=True)

    ttk.Button(top, text=f'Discover {mode_cfg["label"]} data',
              command=_discover).grid(row=1, column=1, sticky='w', pady=(8, 0))

    # Auto-discover once a *new* valid directory is in place — see the
    # matching comment in _build_bm_embedded_page.
    _last_auto_dir = {'value': None}

    def _auto_discover(*_a):
        text = superdir_var.get().strip()
        if text and Path(text).is_dir() and text != _last_auto_dir['value']:
            _last_auto_dir['value'] = text
            _discover()

    superdir_var.trace_add('write', _auto_discover)
    _auto_discover()

    return page


if __name__ == '__main__':
    main()
