"""
gate_experiments_inplace.py

Interactive GUI for gating buoyant mass (BM) and iFXM volume data across all
sample subfolders in an experiment superdir. After gating, one YAML file
recording every gate is written into each sample subfolder, and a timestamped
summary folder is written into the superdir.

Workflow:
    1. The script discovers each sample subdir's mass table and volume data.
       Up to FIVE independent histograms are gated at once — mass_pg,
       normalized baseline (avg_baseline divided by its mean over the first
       10% of the run), bl_slope, and node_dev_mean, porting the
       four-attribute gate from this repo's MATLAB gate_mass_results.m, plus
       volume_fl (see MultiAttributeGatingPanel). A sample with only mass or
       only volume data simply doesn't appear in the other kind's
       histogram(s); the volume histogram is omitted entirely if no sample has
       volume data.
    2. A single window shows the sample list on the left and the histograms
       on the right. Selecting/deselecting samples in the list immediately
       previews their overlaid histograms (shared bin edges).
    3. The user clicks on a histogram to set lower and upper cutoffs (either
       order); changing the left-list selection while doing this restarts the
       cutoffs for the new selection. Every histogram is independently
       clickable at all times (no per-plot "arm" button) and gating each is
       optional — the eventual per-sample gate is the logical AND of whichever
       attribute(s) were actually gated. Each gated histogram shows a live
       "% retained" readout, and the bottom bar shows the pooled mass AND
       retained %. "Apply cutoffs" commits the group (confirming first if zero
       attributes are gated) and returns the panel to an idle preview.
    4. Step 2-3 repeats until all samples are assigned. "Finalize gating"
       (highlighted) then becomes available.
    5. On "Finalize gating":
         - One gate file is written into each sample subfolder (format in
           gating/gates.py):
             <sample_subdir>/<YYYYMMDD_HHMMSS>_gating/<sample_subdir_name>_gate.yaml
           holding a `gates:` mapping of every attribute actually gated for
           that sample (only attributes the sample has data for). An empty
           `gates: {}` means the sample was deliberately left ungated.
           compile_experiment.py applies the AND of all of them.
         - A summary folder is written into the superdir:
             <YYMMDD.HHMMSS>_gating_summary/
               cutoff_log.txt
               cutoff_stats.csv
               histograms/group_NN.png    (one panel per attribute)
    6. A "← Back" button undoes the last group of cutoffs, restoring those
       samples to the remaining list. Can be pressed repeatedly.

For samples with matched mass+volume cells (analysis/density/cells in the
CELLGROUPED hdf5), a readout shows the % of paired cells inside both the live
mass_pg gate and the live volume gate.

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
    python gate_experiments_inplace.py <superdir>
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

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from gating.common import style_finalize_button
from gating.gates import (MASS_GATE_COLUMNS, mass_attr_values, mass_gate_mask,
                          write_gate_file)

from fsutil import is_appledouble


# ---------------------------------------------------------------------------
# Gate attributes: the four mass-table attributes (mass, normalized baseline,
# baseline slope, average node deviation) — ported from the four-histogram
# gate in MATLAB's gate_mass_results.m (smr_data_analysis) — plus iFXM volume.
# ---------------------------------------------------------------------------

_BM_REQUIRED_COLUMNS = MASS_GATE_COLUMNS

_BM_ATTRS = [
    {'key': 'mass_pg', 'label': 'Buoyant mass',
     'xlabel': 'Buoyant mass (pg)'},
    {'key': 'baseline_norm', 'label': 'Normalized baseline',
     'xlabel': 'Normalized baseline (frac. of first-10% mean)', 'derived': True,
     # Default view extends past the data by this fraction of its range on
     # each side, leaving room to place gates in the tails.
     'x_pad': 0.25},
    {'key': 'bl_slope', 'label': 'Baseline slope',
     'xlabel': 'Baseline slope'},
    {'key': 'node_dev_mean', 'label': 'Average node deviation',
     'xlabel': 'Avg node deviation'},
]
_ATTR_SPEC = {a['key']: a for a in _BM_ATTRS}
_ATTR_KEYS = [a['key'] for a in _BM_ATTRS]
_ATTR_LABELS = {a['key']: a['label'] for a in _BM_ATTRS}

# The optional 5th histogram: iFXM volume. Unlike the four BM attributes (all
# columns of one per-cell mass table), volume comes from separate per-sample
# data (_discover_ifxm) with its own cell population, so the live "mass
# retained" readout never ANDs it in. It's written to the same per-sample gate
# file, though, and compile_experiment.py ANDs it with the mass-table gates on
# matched cells (via analysis/density/cells' matched_peak_index).
_VOLUME_ATTR = {'key': 'volume_fl', 'label': 'iFXM volume',
                'xlabel': 'Volume (fL)', 'source': 'volume', 'scale': 'log',
                # On a log axis x_pad is a fraction of the range in decades.
                'x_pad': 0.25}
_ALL_ATTRS = _BM_ATTRS + [_VOLUME_ATTR]
_ATTR_LABELS[_VOLUME_ATTR['key']] = _VOLUME_ATTR['label']


def _attr_values(df: pd.DataFrame, attr: dict) -> np.ndarray:
    """The vector of values one BM attribute spec operates on for a sample table."""
    return mass_attr_values(df, attr['key'])


def _sample_values(name: str, spec: dict, mass_data: dict,
                   volume_data: dict) -> np.ndarray | None:
    """One sample's values for one attribute (mass table or volume), or None
    if that sample has no data of this attribute's kind."""
    if spec.get('source') == 'volume':
        return volume_data.get(name)
    df = mass_data.get(name)
    return None if df is None else _attr_values(df, spec)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_cli_args() -> Path:
    parser = argparse.ArgumentParser(
        description="Interactively gate BM and iFXM volume data across "
                    "sample subfolders of an experiment superdir."
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
# Paired mass+volume cells, for the panel's "% of paired data retained"
# readout: the fraction of matched cells (the CELLGROUPED hdf5's
# analysis/density/cells table, the same source compile_experiment.py pairs
# from) inside both the live mass_pg gate and the live volume gate.
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


def _discover_combined(superdir: Path) -> tuple[dict, dict, dict]:
    """
    Everything the combined mass + volume gating page needs:
    (mass_data, volume_data, paired_data) — per-sample mass tables
    (_discover_bm_tables), volume arrays (_discover_ifxm), and, for samples
    with both, matched mass+volume cells (_load_paired_mass_volume).
    """
    mass_data = _discover_bm_tables(superdir)
    volume_data = _discover_ifxm(superdir)
    paired_data = {}
    for name in mass_data.keys() & volume_data.keys():
        hdf5_path = _find_cellgrouped_hdf5(superdir / name)
        if hdf5_path is None:
            continue
        pdf = _load_paired_mass_volume(hdf5_path)
        if pdf is not None and not pdf.empty:
            paired_data[name] = pdf
    return mass_data, volume_data, paired_data


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _relevant_gates(name: str, ranges: dict, mass_data: dict,
                    volume_data: dict) -> dict:
    """The subset of a group's gates that apply to one sample: mass-table
    attributes only if it has mass data, volume only if it has volume data."""
    out = {}
    for key, r in ranges.items():
        has = (name in volume_data) if key == _VOLUME_ATTR['key'] else (name in mass_data)
        if has:
            out[key] = r
    return out


def _write_gate_files(superdir: Path, sample_dirs: dict, cutoffs: dict,
                      timestamp: str):
    """One <sample>/<timestamp>_gating/<sample>_gate.yaml per sample (format in
    gating/gates.py). Written even for a sample with no gates, so it
    supersedes any older gate file on disk."""
    for sample, gates in cutoffs.items():
        out_path = write_gate_file(sample_dirs[sample], superdir.name, gates,
                                   timestamp)
        print(f"Written: {out_path}")
    print(f"[yaml] {len(cutoffs)} gate file(s) written.")


def _save_group_histograms(hist_dir: Path, mass_data: dict, volume_data: dict,
                           groups: list):
    """One PNG per cutoff group — every attribute with data in that group
    (2x2 for the four mass-table attributes, 2x3 with volume) — with cutoff
    lines drawn for whichever attributes were actually gated."""
    for i, (ranges, names) in enumerate(groups, 1):
        specs = [s for s in _ALL_ATTRS
                 if any(_sample_values(n, s, mass_data, volume_data) is not None
                        for n in names)]
        if not specs:
            continue
        n_cols = 2 if len(specs) <= 4 else 3
        n_rows = (len(specs) + n_cols - 1) // n_cols
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows),
                                squeeze=False)
        for ax in axs.flat[len(specs):]:
            ax.axis('off')
        for ax, spec in zip(axs.flat, specs):
            key = spec['key']
            log = spec.get('scale') == 'log'
            arrays = []
            for name in names:
                v = _sample_values(name, spec, mass_data, volume_data)
                if v is None:
                    continue
                v = v[np.isfinite(v) & (v > 0)] if log else v[np.isfinite(v)]
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
                hi = lo * 2 if log else lo + 1
            edges = np.geomspace(lo, hi, 101) if log else np.linspace(lo, hi, 101)

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
            if log:
                ax.set_xscale('log')
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


def _retained_counts(name: str, gates: dict, mass_data: dict,
                     volume_data: dict) -> dict:
    """n_before/n_after for a sample's mass table (AND of the mass-table
    gates) and volume data (volume gate) — None where it has no such data."""
    out = {'mass': None, 'volume': None}
    df = mass_data.get(name)
    if df is not None:
        out['mass'] = (len(df), int(np.count_nonzero(mass_gate_mask(df, gates))))
    vol = volume_data.get(name)
    if vol is not None:
        g = gates.get(_VOLUME_ATTR['key'])
        n_after = (len(vol) if g is None
                   else int(np.count_nonzero((vol >= g[0]) & (vol <= g[1]))))
        out['volume'] = (len(vol), n_after)
    return out


def _write_cutoff_log(log_path: Path, superdir: Path, mass_data: dict,
                      volume_data: dict, cutoffs: dict, groups: list):
    lines = [
        "gate_experiments_inplace — Cutoff Log",
        "=" * 70,
        f"Experiment:  {superdir.name}",
        f"Superdir:    {superdir}",
        f"Run:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Cutoff groups (mass = AND of the mass-table gates; volume = volume gate)",
        "-" * 70,
    ]
    for i, (ranges, names) in enumerate(groups, 1):
        if ranges:
            range_desc = '; '.join(
                f"{_ATTR_LABELS[k]}: {lo:.4g} - {hi:.4g}"
                for k, (lo, hi, _v) in ranges.items())
        else:
            range_desc = '(no gates set — all points accepted)'
        lines.append(f"Group {i}   {range_desc}")

        for name in names:
            counts = _retained_counts(name, cutoffs.get(name, {}),
                                      mass_data, volume_data)
            parts = []
            for kind in ('mass', 'volume'):
                if counts[kind] is None:
                    continue
                n_before, n_after = counts[kind]
                pct = 100 * (n_before - n_after) / n_before if n_before else 0.0
                parts.append(f"{kind} {n_before} -> {n_after} ({pct:.1f}% removed)")
            lines.append(f"  {name:<40s}  {'   '.join(parts)}")
        lines.append("")

    log_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f"Written: {log_path}")


def _write_stats_csv(stats_path: Path, mass_data: dict, volume_data: dict,
                     cutoffs: dict, groups: list):
    rows = []
    for _ranges, names in groups:
        for name in names:
            gates = cutoffs.get(name, {})
            row = {'sample': name}
            for spec in _ALL_ATTRS:
                lo, hi = gates.get(spec['key'], (np.nan, np.nan))
                row[f"{spec['key']}_lower"] = lo
                row[f"{spec['key']}_upper"] = hi

            counts = _retained_counts(name, gates, mass_data, volume_data)
            for kind in ('mass', 'volume'):
                n_before, n_after = counts[kind] or (np.nan, np.nan)
                row[f'n_{kind}_before'] = n_before
                row[f'n_{kind}_after'] = n_after

            mass = np.array([])
            df = mass_data.get(name)
            if df is not None:
                mass = df['mass_pg'].to_numpy(dtype=float)[mass_gate_mask(df, gates)]
                mass = mass[np.isfinite(mass)]
            row['mass_mean'] = float(mass.mean()) if mass.size else np.nan
            row['mass_median'] = float(np.median(mass)) if mass.size else np.nan
            row['mass_std'] = float(mass.std()) if mass.size else np.nan

            vol = volume_data.get(name)
            if vol is not None:
                g = gates.get(_VOLUME_ATTR['key'])
                if g is not None:
                    vol = vol[(vol >= g[0]) & (vol <= g[1])]
            row['volume_median'] = (float(np.median(vol))
                                    if vol is not None and vol.size else np.nan)
            rows.append(row)

    pd.DataFrame(rows).to_csv(stats_path, index=False)
    print(f"Written: {stats_path}")


def _write_output(superdir: Path, sample_dirs: dict, mass_data: dict,
                  volume_data: dict, cutoffs: dict, groups: list) -> Path:
    """
    Writes every sample's gate file plus one summary folder.

    `cutoffs` maps sample -> {attr_key: (lower, upper)} and `groups` is an
    ordered list of ({attr_key: (lower, upper, view)}, [sample, ...]), with
    'volume_fl' as just another attribute key. Each sample's gate file only
    gets the attributes it has data for (see _relevant_gates).
    """
    cutoffs = {name: _relevant_gates(name, ranges, mass_data, volume_data)
               for name, ranges in cutoffs.items()}

    timestamp = datetime.now().strftime('%y%m%d.%H%M%S')
    summary_dir = superdir / f'{timestamp}_gating_summary'
    summary_dir.mkdir()
    hist_dir = summary_dir / 'histograms'
    hist_dir.mkdir()

    _write_gate_files(superdir, sample_dirs, cutoffs,
                      datetime.now().strftime('%Y%m%d_%H%M%S'))
    _save_group_histograms(hist_dir, mass_data, volume_data, groups)
    _write_cutoff_log(summary_dir / 'cutoff_log.txt', superdir, mass_data,
                      volume_data, cutoffs, groups)
    _write_stats_csv(summary_dir / 'cutoff_stats.csv', mass_data, volume_data,
                     cutoffs, groups)
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
    it sets its own lower and upper cutoff (either order) without any per-plot activation
    button, and selecting/deselecting samples on the left immediately
    updates all four histograms (as in GatingPanel).

    Unlike the MATLAB version, gating an attribute is entirely optional —
    "Apply cutoffs" commits the group using whichever 0-4 attributes were
    actually gated (confirming first if none were); the final per-sample
    gate is the logical AND of whichever range(s) were set.

    Every attribute panel shows a live "Retained: X%" readout once its own
    cutoff is set (its data only), and the bottom bar shows the pooled
    percentage retained by the AND of every mass-table attribute currently
    gated.

    If `volume_data` is given, a 5th iFXM volume histogram (log x axis) is
    added to the grid (2x3 instead of 2x2) and gated per-sample together with
    the other four. Its cells are a separate population from the mass table,
    so the live mass-table AND readout leaves it out; instead, for samples
    with matched mass+volume cells (`paired_data`), the bottom bar also shows
    the % of paired cells inside both the mass_pg and the volume gate.
    Samples may have mass data, volume data, or both; each histogram just
    skips samples lacking its data.

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
        volume_data:   optional mapping of sample name -> volume array (fL)
        paired_data:   optional mapping of sample name -> pd.DataFrame with
                       mass_pg + volume_fl per matched cell
    """

    def __init__(self, parent: tk.Widget, columns: list, data: dict,
                on_finish, context_label: str = '', on_done=None,
                volume_data: dict | None = None, paired_data: dict | None = None):
        super().__init__(parent)
        self._columns = columns
        self._data = data
        self._volume_data = volume_data or {}
        self._paired_data = paired_data or {}
        self._on_finish = on_finish
        self._on_done = on_done
        self._cutoffs: dict = {}
        self._groups: list = []
        self._remaining: list = list(columns)
        self._history: list = []
        self._active_selection: list | None = None

        self._specs = list(_BM_ATTRS) + ([_VOLUME_ATTR] if self._volume_data else [])
        self._spec = {s['key']: s for s in self._specs}
        self._keys = [s['key'] for s in self._specs]

        ctx = f"  [{context_label}]" if context_label else ''
        vol = ', volume' if self._volume_data else ''
        self.title = f"Gating — Buoyant Mass (mass, baseline, slope, node dev{vol}){ctx}"

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

        # --- right: attribute grid (2x2, or 2x3 with volume) ---
        n_cols = 2 if len(self._specs) <= 4 else 3
        grid = tk.Frame(right)
        grid.pack(fill=tk.BOTH, expand=True)
        for c in range(n_cols):
            grid.columnconfigure(c, weight=1, uniform='attr')
        grid.rowconfigure(0, weight=1, uniform='attr')
        grid.rowconfigure(1, weight=1, uniform='attr')

        figsize = (4.6, 2.8) if n_cols == 2 else (3.6, 2.6)
        self._attr: dict = {}
        for idx, spec in enumerate(self._specs):
            r, c = divmod(idx, n_cols)
            state = self._build_attr_cell(grid, spec, figsize)
            state['frame'].grid(row=r, column=c, sticky='nsew', padx=3, pady=3)
            self._attr[spec['key']] = state

        bottom = tk.Frame(right)
        bottom.pack(fill=tk.X, pady=(6, 0))
        self._apply_btn = tk.Button(bottom, text='Apply cutoffs',
                                    state=tk.DISABLED, command=self._apply_cutoffs)
        self._apply_btn.pack(side=tk.RIGHT, anchor='n')
        self._overall_var = tk.StringVar()
        tk.Label(bottom, textvariable=self._overall_var, anchor='w',
                font=('TkDefaultFont', 10, 'bold')).pack(fill=tk.X)
        self._paired_var = tk.StringVar()
        if self._paired_data:
            tk.Label(bottom, textvariable=self._paired_var, anchor='w',
                    font=('TkDefaultFont', 9, 'bold'),
                    foreground='#1a6b1a').pack(fill=tk.X)

        for key in self._keys:
            self._draw_idle_attr(key)
        self._update_overall_retained()
        self._refresh_list()

    # -- attribute cell construction -------------------------------------

    def _build_attr_cell(self, parent: tk.Widget, spec: dict, figsize: tuple) -> dict:
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
        fig = Figure(figsize=figsize)
        ax = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=cell)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=2)
        canvas.mpl_connect('button_press_event',
                           lambda e, k=key: self._on_attr_click(k, e))

        view_row = tk.Frame(cell)
        view_row.pack(fill=tk.X, padx=4, pady=(2, 0))
        tk.Label(view_row, text='x:').pack(side=tk.LEFT)
        xmin_var = tk.StringVar()
        xmin_entry = tk.Entry(view_row, textvariable=xmin_var, width=8)
        xmin_entry.pack(side=tk.LEFT, padx=(2, 2))
        xmax_var = tk.StringVar()
        xmax_entry = tk.Entry(view_row, textvariable=xmax_var, width=8)
        xmax_entry.pack(side=tk.LEFT, padx=(0, 4))
        for entry in (xmin_entry, xmax_entry):
            for seq in ('<Return>', '<KP_Enter>'):
                entry.bind(seq, lambda _e, k=key: self._apply_attr_xlim(k))
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
        kind = 'Buoyant Mass + Volume' if self._volume_data else 'Buoyant Mass'
        self._header_var.set(f"Remaining: {n_total - n_done} / {n_total}   "
                             f"[{kind}]")
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
        for key in self._keys:
            st = self._attr[key]
            st['lower'] = None
            st['upper'] = None
            st['state'] = 0
            st['view'] = None
            st['xmin_var'].set('')
            st['xmax_var'].set('')
            st['status_var'].set('Click to set a cutoff.')
            self._draw_attr_histograms(key)
            self._update_attr_retained(key)
        self._apply_btn.config(state=tk.NORMAL)
        self._update_overall_retained()

    def _clear_active_gating(self):
        self._active_selection = None
        for key in self._keys:
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
        set_keys = [k for k in self._keys if self._attr[k]['state'] == 2]
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

    def _values(self, name: str, spec: dict) -> np.ndarray | None:
        """One sample's values for one attribute, or None if that sample has
        no data of this attribute's kind (mass table vs. volume)."""
        return _sample_values(name, spec, self._data, self._volume_data)

    def _draw_idle_attr(self, key: str, text: str = 'Select samples\non the left'):
        st = self._attr[key]
        ax = st['ax']
        ax.clear()
        ax.text(0.5, 0.5, text, ha='center', va='center',
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

        spec = self._spec[key]
        st = self._attr[key]
        ax = st['ax']
        ax.clear()

        log = spec.get('scale') == 'log'
        arrays = []
        for name in self._active_selection:
            v = self._values(name, spec)
            if v is None:
                continue
            v = v[np.isfinite(v) & (v > 0)] if log else v[np.isfinite(v)]
            if v.size:
                arrays.append((name, v))
        if not arrays:
            self._draw_idle_attr(key, 'No data for\nthis selection')
            return

        pooled = np.concatenate([v for _, v in arrays])
        pmin, pmax = float(pooled.min()), float(pooled.max())
        x_pad = spec.get('x_pad', 0.0)
        if log:
            # Pad by a fraction of the range in decades, so the margin looks
            # the same on both sides of a log axis and never crosses zero.
            factor = (pmax / pmin) ** x_pad
            st['full_xlim'] = (pmin / factor, pmax * factor)
        else:
            pad = x_pad * (pmax - pmin)
            st['full_xlim'] = (pmin - pad, pmax + pad)
        if not st['xmin_var'].get() and not st['xmax_var'].get():
            st['xmin_var'].set(f"{st['full_xlim'][0]:.4g}")
            st['xmax_var'].set(f"{st['full_xlim'][1]:.4g}")

        lo, hi = xlim if xlim is not None else st['full_xlim']
        if hi <= lo:
            hi = lo * 2 if log else lo + 1
        # Log-spaced edges on a log axis so bars are equal width.
        edges = np.geomspace(lo, hi, 101) if log else np.linspace(lo, hi, 101)

        for name, v in arrays:
            ax.hist(v, bins=edges, alpha=0.5, edgecolor='black',
                    linewidth=0.3, label=name)
        if log:
            ax.set_xscale('log')
        ax.set_xlim(lo, hi)
        ax.set_xlabel(spec['xlabel'], fontsize=8)
        ax.set_ylabel('count', fontsize=8)
        ax.tick_params(labelsize=7)
        if key == self._keys[0] and len(arrays) > 1:
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
        if self._spec[key].get('scale') == 'log' and lo <= 0:
            st['status_var'].set('x view: min must be > 0 on a log axis.')
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
        st['status_var'].set('Click to set a cutoff.')
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
            st['status_var'].set(f"First cutoff: {x:.4g} — click second cutoff")
        elif st['state'] == 1:
            # Cutoffs may be clicked in either order; sort them here.
            if x == st['lower']:
                st['status_var'].set("Second cutoff must differ from the first. Click again.")
                return
            st['lower'], st['upper'] = sorted((st['lower'], x))
            st['state'] = 2
            st['status_var'].set(f"Lower: {st['lower']:.4g}  Upper: {st['upper']:.4g}")
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
        spec = self._spec[key]
        total = 0
        kept = 0
        for name in self._active_selection:
            v = self._values(name, spec)
            if v is None:
                continue
            total += v.size
            kept += int(np.count_nonzero((v >= lo) & (v <= hi)))
        pct = 100 * kept / total if total else 0.0
        st['retained_var'].set(f'Retained: {pct:.1f}%  ({kept}/{total})')

    def _update_overall_retained(self):
        self._update_paired_retained()
        if self._active_selection is None:
            self._overall_var.set('')
            return
        set_keys = [k for k in self._keys if self._attr[k]['state'] == 2]
        # The mass-table AND covers only the four BM attributes; volume is a
        # separate cell population (see _update_paired_retained).
        flat_ranges = {k: (self._attr[k]['lower'], self._attr[k]['upper'])
                      for k in set_keys if k in _ATTR_SPEC}
        total = 0
        kept = 0
        for name in self._active_selection:
            df = self._data.get(name)
            if df is None:
                continue
            total += len(df)
            kept += int(np.count_nonzero(mass_gate_mask(df, flat_ranges)))
        if total:
            pct = 100 * kept / total
            retained = f'Mass retained (AND of BM gates): {pct:.1f}%  ({kept}/{total})'
        else:
            retained = 'Mass retained: n/a (no mass data in selection)'
        self._overall_var.set(
            f'Gates set: {len(set_keys)}/{len(self._keys)}    {retained}')

    def _update_paired_retained(self):
        """% of matched mass+volume cells (CELLGROUPED analysis/density/cells)
        inside both the mass_pg gate and the volume gate — whichever of the
        two is set; an unset one passes everything."""
        if not self._paired_data:
            return
        if self._active_selection is None:
            self._paired_var.set('')
            return
        relevant = [n for n in self._active_selection if n in self._paired_data]
        if not relevant:
            self._paired_var.set('Paired mass+volume retained: n/a for this selection')
            return
        gates = {}
        for key in ('mass_pg', _VOLUME_ATTR['key']):
            st = self._attr.get(key)
            if st is not None and st['state'] == 2:
                gates[key] = (st['lower'], st['upper'])
        total = 0
        kept = 0
        for name in relevant:
            pdf = self._paired_data[name]
            mask = np.ones(len(pdf), dtype=bool)
            for key, (lo, hi) in gates.items():
                v = pdf[key].to_numpy(dtype=float)
                mask &= (v >= lo) & (v <= hi)
            total += len(pdf)
            kept += int(np.count_nonzero(mask))
        pct = 100 * kept / total if total else 0.0
        self._paired_var.set(
            f'Paired mass+volume retained (mass gate x volume gate): '
            f'{pct:.1f}%  ({kept}/{total})')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    superdir = parse_cli_args()

    print(f"Discovering Buoyant Mass + iFXM Volume data in {superdir.name}...")
    mass_data, volume_data, paired_data = _discover_combined(superdir)
    columns = sorted(mass_data.keys() | volume_data.keys())
    if not columns:
        print(f"No Buoyant Mass or iFXM Volume data found in {superdir}")
        sys.exit(1)
    sample_dirs = {name: superdir / name for name in columns}
    print(f"Found {len(columns)} sample(s): {', '.join(columns)}")

    def on_finish(cutoffs, groups):
        return _write_output(superdir, sample_dirs, mass_data, volume_data,
                             cutoffs, groups)

    root = tk.Tk()
    panel = MultiAttributeGatingPanel(root, columns, mass_data, on_finish,
                                      context_label=superdir.name,
                                      on_done=root.destroy,
                                      volume_data=volume_data,
                                      paired_data=paired_data)
    root.title(panel.title)
    panel.pack(fill=tk.BOTH, expand=True)
    root.mainloop()


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def build_embedded_page(parent: tk.Widget, *,
                        initial_superdir: str | None = None,
                        superdir_var: tk.StringVar | None = None,
                        on_finalized=None, on_discover=None) -> ttk.Frame:
    """
    Build this tool's gating UI as a Frame suitable for embedding in a larger
    application, e.g. a wizard page: the four mass-table histograms plus an
    iFXM volume histogram, gated together per-sample (the volume histogram is
    simply absent if the directory has no volume data).

    A directory field + "Discover" button stand in for the CLI's positional
    superdir argument. Discovery also runs automatically whenever the
    directory field resolves to a *new* valid directory — whether typed here,
    chosen via Browse, or (via a shared `superdir_var`) set from another page
    entirely — so a wizard step ahead of this one setting the directory is
    enough; the button remains only for an explicit manual re-discover (e.g.
    after files changed on disk without the path itself changing). Either
    way, (re)running discovery (re)builds the gating panel underneath from
    scratch, so changing directories after gating has started simply starts
    over.

    Pass an existing `superdir_var` (rather than `initial_superdir`) to bind
    this page's directory field to a StringVar shared with another page.

    `on_finalized`, if given, is called once each time the embedded panel's
    "Finalize gating" is clicked (i.e. its own on_done) — e.g. a wizard uses
    this to enable its Next button. `on_discover`, if given, is called at the
    start of every discovery run — whether from the button or automatic —
    before this page's own state is reset — e.g. the same wizard uses this
    to re-disable Next, since a fresh discovery means whatever was
    previously finalized here no longer applies.
    """
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

        mass_data, volume_data, paired_data = _discover_combined(superdir)
        columns = sorted(mass_data.keys() | volume_data.keys())
        if not columns:
            status_var.set(f'No Buoyant Mass or iFXM Volume data found under {superdir}.')
            return

        sample_dirs = {name: superdir / name for name in columns}

        def on_finish(cutoffs, groups):
            out = _write_output(superdir, sample_dirs, mass_data, volume_data,
                                cutoffs, groups)
            status_var.set(f'Gate files written; summary in: {out}')
            return out

        gating_panel = MultiAttributeGatingPanel(
            panel_container, columns, mass_data, on_finish,
            context_label=superdir.name, on_done=on_finalized,
            volume_data=volume_data, paired_data=paired_data)
        gating_panel.pack(fill=tk.BOTH, expand=True)

    ttk.Button(top, text='Discover Buoyant Mass + Volume data',
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


if __name__ == '__main__':
    main()
