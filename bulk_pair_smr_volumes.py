"""
bulk_pair_smr_volumes.py
Batch SMR mass / FXM volume pairing — discovers experiment folders and runs
pairing on each one.

Expected folder structure (default, depth=2)
---------------------------------------------
    <root_dir>/
      2026-06-03_collaborator_drugtreat/    <- depth-1 superfolder (date filter applied here)
        zota_24h_samp05/                    <- depth-2 experiment folder
        zota_24h_samp06/
      2026-06-05_cells/
        rep_1/

Each experiment folder must contain both a *_imaging_fxm_results and a
*_mass_results subdirectory.  The most recent of each (by timestamp prefix)
is used automatically.

Depth-1 superfolders must begin with YYYY-MM-DD for date filtering to apply.
Folders named *_imaging_fxm_results, *_mass_results, stage1_image_processing,
and stage2_analysis are always skipped during discovery.

In recursive mode, any directory at any depth that contains both analysis
subdirs qualifies.

Usage
-----
    # All experiments under root, last 3 superfolders
    python bulk_pair_smr_volumes.py E:\\experiments --last 3

    # Date range, skip already-paired folders
    python bulk_pair_smr_volumes.py E:\\experiments --from 2026-06-01 --to 2026-06-07 --skip-paired

    # From a selection file
    python bulk_pair_smr_volumes.py E:\\experiments --from-file batch.txt

    # Dry run — show what would be processed without running anything
    python bulk_pair_smr_volumes.py E:\\experiments --from 2026-06-03 --dry-run

    # Restrict to fixed depth=2 instead of recursive (recursive is default)
    python bulk_pair_smr_volumes.py E:\\experiments --no-recursive --skip-paired
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from datetime import date, datetime
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.stage2.pairing_utils import (
    gaussian_kernel as _gaussian_kernel,
    build_mass_signal,
    build_vol_signal,
    xcorr_lag,
    make_vol_idx_signal,
    find_match_pairs,
)


# =============================================================================
# Directory / file discovery (shared with pair_smr_volumes.py logic)
# =============================================================================

_TS_PATTERNS = [
    (re.compile(r"^(\d{8}_\d{6})_"), "%Y%m%d_%H%M%S"),   # 20260606_025232_…
    (re.compile(r"^(\d{8}\.\d{6})_"), "%Y%m%d.%H%M%S"),  # 20260607.150735_…
]

_DATE_RE   = re.compile(r"^(\d{4}-\d{2}-\d{2})")
_SKIP_NAMES = {"stage1_image_processing", "stage2_analysis"}


def _parse_timestamp(dirname: str) -> Optional[datetime]:
    for pat, fmt in _TS_PATTERNS:
        m = pat.match(os.path.basename(dirname))
        if m:
            try:
                return datetime.strptime(m.group(1), fmt)
            except ValueError:
                pass
    return None


def _parse_folder_date(name: str) -> Optional[date]:
    m = _DATE_RE.match(name)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d").date()
        except ValueError:
            pass
    return None


def find_most_recent_dir(parent: str, suffix: str) -> str:
    candidates = [
        d for d in os.listdir(parent)
        if d.endswith(suffix) and os.path.isdir(os.path.join(parent, d))
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No directory ending with '{suffix}' found in:\n  {parent}"
        )
    return max(candidates, key=lambda d: _parse_timestamp(d) or datetime.min)


def find_file_in_dir(directory: str, pattern: str) -> str:
    matches = glob.glob(os.path.join(directory, pattern))
    if not matches:
        raise FileNotFoundError(
            f"No file matching '{pattern}' found in:\n  {directory}"
        )
    return matches[0]


def find_mass_csv(mass_dir: str) -> str:
    csvs = [
        f for f in glob.glob(os.path.join(mass_dir, "*.csv"))
        if not os.path.basename(f).startswith("curation_index")
    ]
    if not csvs:
        raise FileNotFoundError(f"No mass CSV found in:\n  {mass_dir}")
    return max(csvs, key=os.path.getsize)


def _has_both_analyses(folder: str) -> bool:
    """True if folder has both a *_imaging_fxm_results and a *_mass_results subdir."""
    has_vol  = False
    has_mass = False
    try:
        for name in os.listdir(folder):
            if name.endswith("_imaging_fxm_results") and os.path.isdir(os.path.join(folder, name)):
                has_vol = True
            if name.endswith("_mass_results") and os.path.isdir(os.path.join(folder, name)):
                has_mass = True
    except OSError:
        pass
    return has_vol and has_mass


def _is_paired(folder: str) -> bool:
    """True if folder already contains a *_pairing_results subdirectory."""
    try:
        for name in os.listdir(folder):
            if name.endswith("_pairing_results") and os.path.isdir(os.path.join(folder, name)):
                return True
    except OSError:
        pass
    return False


def _filter_depth1(root_dir: str, date_from=None, date_to=None, last_n=None):
    try:
        entries = sorted(os.listdir(root_dir))
    except OSError:
        return []

    candidates = []
    for name in entries:
        path = os.path.join(root_dir, name)
        if not os.path.isdir(path):
            continue
        if (name.endswith("_imaging_fxm_results")
                or name.endswith("_mass_results")
                or name in _SKIP_NAMES):
            continue
        d = _parse_folder_date(name)
        if date_from is not None and (d is None or d < date_from):
            continue
        if date_to is not None and (d is None or d > date_to):
            continue
        candidates.append((d, name))

    candidates.sort(key=lambda x: (x[0] is None, x[0], x[1]))
    if last_n is not None:
        candidates = candidates[-last_n:]
    return [name for _, name in candidates]


def discover_experiments(
    root_dir: str,
    date_from=None,
    date_to=None,
    last_n=None,
    skip_paired: bool = False,
    recursive: bool = False,
):
    """
    Yield absolute paths of experiment folders that have both analysis results.

    Default (recursive=False): depth-1 = date-named superfolders, depth-2 = experiment folders.
    recursive=True: any subdirectory at any depth with both analysis subdirs qualifies.
    """
    root_dir = os.path.normpath(root_dir)
    passing_depth1 = _filter_depth1(root_dir, date_from=date_from, date_to=date_to, last_n=last_n)

    for super_name in passing_depth1:
        super_path = os.path.join(root_dir, super_name)

        if not recursive:
            try:
                sub_entries = sorted(os.listdir(super_path))
            except OSError:
                continue
            for sub_name in sub_entries:
                sub_path = os.path.join(super_path, sub_name)
                if not os.path.isdir(sub_path):
                    continue
                if (sub_name.endswith("_imaging_fxm_results")
                        or sub_name.endswith("_mass_results")
                        or sub_name.endswith("_pairing_results")
                        or sub_name in _SKIP_NAMES):
                    continue
                if not _has_both_analyses(sub_path):
                    continue
                if skip_paired and _is_paired(sub_path):
                    continue
                yield sub_path
        else:
            for dirpath, dirnames, _ in os.walk(super_path):
                dirnames[:] = sorted(
                    d for d in dirnames
                    if not d.endswith("_imaging_fxm_results")
                    and not d.endswith("_mass_results")
                    and not d.endswith("_pairing_results")
                    and d not in _SKIP_NAMES
                )
                if not _has_both_analyses(dirpath):
                    continue
                if skip_paired and _is_paired(dirpath):
                    continue
                yield dirpath


def _load_from_file(root_dir: str, filepath: str) -> list[str]:
    folders = []
    with open(filepath) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            path = line if os.path.isabs(line) else os.path.join(root_dir, line)
            path = os.path.normpath(path)
            if not os.path.isdir(path):
                print(f"[WARN] From-file entry not found, skipping: {path}")
                continue
            folders.append(path)
    return folders


# =============================================================================
# Pairing algorithm (uses shared primitives from pipeline.stage2.pairing_utils)
# =============================================================================

def pair_mass_and_volumes(
    vol_df: pd.DataFrame,
    transit_times: pd.Series,
    mass_df: pd.DataFrame,
    timebase: float = 1e-3,
    peak_tolerance: int = 11,
    gaussian_width: int = 15,
    utc_offset_hours: float = 0.0,
) -> list[dict]:
    float_mass_cols = [
        "matched_mass", "buoyant_density", "node_dev_mean",
        "avg_baseline", "volume_time", "mass_time", "run_number",
    ]
    for col in float_mass_cols:
        if col in vol_df.columns:
            vol_df[col] = pd.to_numeric(vol_df[col], errors="coerce")
    for col in ("mass_table_row", "mass_csv_row"):
        if col in vol_df.columns:
            vol_df[col] = pd.to_numeric(vol_df[col], errors="coerce")
    for col in ("sample", "sample_id", "condition"):
        if col in vol_df.columns:
            vol_df[col] = vol_df[col].astype(object)

    kern = _gaussian_kernel(gaussian_width)

    mass_df = mass_df.copy()
    mass_df["_orig_row"] = range(len(mass_df))
    mass_sorted = (
        mass_df.dropna(subset=["real_time_s"])
        .sort_values("real_time_s")
        .reset_index(drop=True)
    )
    n_dropped = len(mass_df) - len(mass_sorted)
    if n_dropped:
        print(f"[WARN]  Dropped {n_dropped} mass rows with NaN real_time_s.")
    if mass_sorted.empty:
        print("[WARN]  No valid mass timestamps — nothing to pair.")
        return []
    all_mass  = mass_sorted["mass_pg"].values
    all_times = mass_sorted["real_time_s"].values % 86400 + utc_offset_hours * 3600

    t_axis, mass_blurred, mass_idx_sig, _ = build_mass_signal(
        all_times, all_mass, timebase, peak_tolerance, kern
    )

    good_mask = (
        (vol_df["error_code"].fillna("") == "")
        & pd.to_numeric(vol_df["calibrated_weighted_volume"], errors="coerce").notna()
    )
    good_df = vol_df[good_mask].copy()
    good_df["_vtime"] = transit_times.reindex(good_df["transit_index"].values).values
    good_df = good_df.dropna(subset=["_vtime"]).sort_values("_vtime")

    if good_df.empty:
        print("[WARN]  No good volume cells with timing — nothing to pair.")
        return []

    vol_times  = good_df["_vtime"].values
    breaks     = np.where(np.diff(vol_times) > 10)[0]
    seg_starts = np.concatenate([[0], breaks + 1])
    seg_ends   = np.concatenate([breaks + 1, [len(good_df)]])

    seg_lags: list[dict] = []
    total_matched = 0

    for s, e in zip(seg_starts, seg_ends):
        seg = good_df.iloc[s:e]
        if len(seg) < 20:
            continue

        vols   = pd.to_numeric(seg["calibrated_weighted_volume"], errors="coerce").values
        vtimes = seg["_vtime"].values

        v_axis, vi, v_blurred = build_vol_signal(vtimes, vols, timebase, kern)
        offset_idx, offset_s, fine_lag_s, clock_offset_s = xcorr_lag(
            mass_blurred, v_blurred, t_axis, v_axis, timebase
        )

        if clock_offset_s != 0:
            print(
                f"[INFO]   Segment {len(seg_lags)+1}  [{s}:{e}]  "
                f"{len(seg)} cells  start={vtimes[0]:.1f} s  "
                f"clock offset={clock_offset_s/3600:+.0f} h  fine lag={fine_lag_s:+.3f} s"
            )
        else:
            print(
                f"[INFO]   Segment {len(seg_lags)+1}  [{s}:{e}]  "
                f"{len(seg)} cells  start={vtimes[0]:.1f} s  lag={fine_lag_s:+.3f} s"
            )

        vol_idx_sig = make_vol_idx_signal(vi, offset_idx, len(t_axis))
        seg_matched = 0

        for mi, vi2 in find_match_pairs(mass_idx_sig, vol_idx_sig):
            cell_row = seg.iloc[vi2]
            mass_row = mass_sorted.iloc[mi]
            orig_idx = cell_row.name

            vol_val      = float(cell_row["volume"]) if pd.notna(cell_row["volume"]) else float("nan")
            matched_mass = float(mass_row["mass_pg"])
            bdens        = (matched_mass / vol_val
                            if (vol_val and not np.isnan(vol_val)) else float("nan"))

            vol_df.loc[orig_idx, "matched_mass"]    = matched_mass
            vol_df.loc[orig_idx, "buoyant_density"] = bdens
            vol_df.loc[orig_idx, "node_dev_mean"]   = float(mass_row.get("node_dev_mean", float("nan")))
            vol_df.loc[orig_idx, "avg_baseline"]    = float(mass_row.get("avg_baseline", float("nan")))
            vol_df.loc[orig_idx, "volume_time"]     = float(vtimes[vi2])
            vol_df.loc[orig_idx, "mass_time"]       = float(mass_row["real_time_s"])
            vol_df.loc[orig_idx, "sample"]          = str(mass_row.get("sample", ""))
            vol_df.loc[orig_idx, "sample_id"]       = str(mass_row.get("sample_ID", ""))
            vol_df.loc[orig_idx, "condition"]       = str(mass_row.get("condition", ""))
            vol_df.loc[orig_idx, "run_number"]      = float(mass_row.get("run_number", float("nan")))
            vol_df.loc[orig_idx, "mass_table_row"]  = int(mi)
            vol_df.loc[orig_idx, "mass_csv_row"]    = int(mass_row["_orig_row"])
            seg_matched += 1

        total_matched += seg_matched
        seg_lags.append({
            "seg_index":       len(seg_lags) + 1,
            "seg_start":       s,
            "seg_end":         e,
            "n_cells":         len(seg),
            "lag_s":           offset_s,
            "fine_lag_s":      fine_lag_s,
            "clock_offset_s":  clock_offset_s,
            "seg_start_time":  float(vtimes[0]),
            "n_matched":       seg_matched,
        })

    n_good = len(good_df)
    print(
        f"[INFO]  Pairing complete: {total_matched}/{n_good} good cells matched "
        f"({100*total_matched/n_good:.1f}%)."
    )
    return seg_lags


# =============================================================================
# Diagnostic figures (identical to pair_smr_volumes.py)
# =============================================================================

def _paired_counts(vol_df: pd.DataFrame, mass_df: pd.DataFrame) -> dict:
    n_vol_good   = int((vol_df["error_code"].fillna("") == "").sum())
    paired_mask  = pd.to_numeric(vol_df["mass_table_row"], errors="coerce") >= 0
    n_vol_paired = int(paired_mask.sum())

    n_mass_total  = len(mass_df)
    paired_rows   = pd.to_numeric(vol_df["mass_table_row"], errors="coerce")
    paired_rows   = paired_rows[paired_rows >= 0]
    n_mass_paired = int(paired_rows.nunique())

    return dict(
        n_vol_good=n_vol_good,
        n_vol_paired=n_vol_paired,
        n_vol_unpaired=n_vol_good - n_vol_paired,
        n_mass_total=n_mass_total,
        n_mass_paired=n_mass_paired,
        n_mass_unpaired=n_mass_total - n_mass_paired,
        pct_vol=100 * n_vol_paired / n_vol_good if n_vol_good else 0.0,
        pct_mass=100 * n_mass_paired / n_mass_total if n_mass_total else 0.0,
    )


def plot_pairing_stats(vol_df: pd.DataFrame, mass_df: pd.DataFrame, output_path: str) -> None:
    c = _paired_counts(vol_df, mass_df)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("SMR-FXM Pairing Statistics", fontsize=13, fontweight="bold")

    labels    = ["Volume\ntransits", "Mass\nmeasurements"]
    totals    = [c["n_vol_good"],     c["n_mass_total"]]
    paired    = [c["n_vol_paired"],   c["n_mass_paired"]]
    unmatched = [c["n_vol_unpaired"], c["n_mass_unpaired"]]
    x = np.arange(2)
    w = 0.25
    ax1.bar(x - w, totals,    width=w, label="Total",     color="#4C72B0")
    ax1.bar(x,     paired,    width=w, label="Paired",    color="#55A868")
    ax1.bar(x + w, unmatched, width=w, label="Unmatched", color="#C44E52")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Count")
    ax1.set_title("Counts")
    ax1.legend(fontsize=8)
    for xi, (p, t) in enumerate(zip(paired, totals)):
        pct = 100 * p / t if t else 0.0
        ax1.text(xi, p + max(totals) * 0.01, f"{pct:.1f}%",
                 ha="center", va="bottom", fontsize=9, color="#55A868")

    pie_colors = ["#55A868", "#C44E52"]
    ax2.pie(
        [c["n_vol_paired"], c["n_vol_unpaired"]],
        labels=[f"Paired\n({c['n_vol_paired']})", f"Unmatched\n({c['n_vol_unpaired']})"],
        colors=pie_colors, autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 9},
    )
    ax2.set_title(
        f"Volume transits (good cells)\n"
        f"n={c['n_vol_good']}  paired={c['pct_vol']:.1f}%"
    )
    ax3.pie(
        [c["n_mass_paired"], c["n_mass_unpaired"]],
        labels=[f"Paired\n({c['n_mass_paired']})", f"Unmatched\n({c['n_mass_unpaired']})"],
        colors=pie_colors, autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 9},
    )
    ax3.set_title(
        f"Mass measurements\n"
        f"n={c['n_mass_total']}  paired={c['pct_mass']:.1f}%"
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO]  Pairing stats figure: {output_path}")


def plot_pairing_lags(vol_df: pd.DataFrame, seg_lags: list[dict], output_path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle("SMR-FXM Pairing Lag Diagnostics", fontsize=13, fontweight="bold")

    if seg_lags:
        labels        = [f"Seg {d['seg_index']}\n(t~{d['seg_start_time']:.0f} s)" for d in seg_lags]
        fine_lags     = [d["fine_lag_s"]    for d in seg_lags]
        clock_offsets = [d["clock_offset_s"] for d in seg_lags]
        n_matched     = [d["n_matched"]      for d in seg_lags]
        colors        = ["#4C72B0" if abs(l) < 5 else "#C44E52" for l in fine_lags]
        bars = ax1.bar(labels, fine_lags, color=colors, edgecolor="white")
        span = max(fine_lags) - min(fine_lags) if len(fine_lags) > 1 else 1.0
        for bar, nm, lv, co in zip(bars, n_matched, fine_lags, clock_offsets):
            label = f"n={nm}"
            if co != 0:
                label += f"\n({co/3600:+.0f} h clock)"
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                lv + 0.02 * (span + 1e-6),
                label, ha="center", va="bottom", fontsize=8,
            )
        ax1.axhline(0, color="k", linewidth=0.8, linestyle="--")
        ax1.set_ylabel("Fine alignment lag (s)")
        ax1.set_xlabel("Segment")
        ax1.set_title(
            "Per-segment alignment lag (integer-hour clock offset stripped)\n"
            "(red = |fine lag| > 5 s)"
        )
    else:
        ax1.text(0.5, 0.5, "No segments processed (all < 20 cells or no good cells)",
                 transform=ax1.transAxes, ha="center", va="center", fontsize=10)
        ax1.set_title("Per-segment alignment lag")

    mean_clock  = np.mean([d["clock_offset_s"] for d in seg_lags]) if seg_lags else 0.0
    paired_mask = pd.to_numeric(vol_df["mass_table_row"], errors="coerce") >= 0
    if paired_mask.sum() > 0:
        vt        = pd.to_numeric(vol_df.loc[paired_mask, "volume_time"], errors="coerce")
        mt        = pd.to_numeric(vol_df.loc[paired_mask, "mass_time"],   errors="coerce") % 86400
        residuals = (vt - mt + mean_clock).dropna()
        mean_r    = residuals.mean()
        ax2.hist(residuals, bins=min(50, len(residuals) // 5 + 1),
                 color="#4C72B0", edgecolor="white", linewidth=0.5)
        ax2.axvline(0,      color="k",       linestyle="--", linewidth=0.8, label="zero")
        ax2.axvline(mean_r, color="#C44E52", linestyle="-",  linewidth=1.2,
                    label=f"mean = {mean_r:.4f} s")
        clock_label = f"{mean_clock/3600:+.0f} h" if mean_clock != 0 else "none"
        ax2.set_xlabel(f"volume_time - (mass_time_midnight - clock_offset)  [s]   (clock offset: {clock_label})")
        ax2.set_ylabel("Count")
        ax2.set_title(
            f"Per-cell timing difference  (n = {len(residuals)} pairs)\n"
            f"Width reflects matching window precision"
        )
        ax2.legend(fontsize=9)
    else:
        ax2.text(0.5, 0.5, "No paired cells to show residuals for",
                 transform=ax2.transAxes, ha="center", va="center", fontsize=10)
        ax2.set_title("Per-cell residual timing offset")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO]  Lag diagnostics figure: {output_path}")


def plot_pairing_histograms(vol_df: pd.DataFrame, output_path: str) -> None:
    """Three-panel figure: mass, volume, and buoyant density distributions for paired cells."""
    paired_mask = pd.to_numeric(vol_df["mass_table_row"], errors="coerce") >= 0
    paired = vol_df[paired_mask]
    n_paired = int(paired_mask.sum())

    mass_vals   = pd.to_numeric(paired["matched_mass"],    errors="coerce").dropna()
    vol_vals    = pd.to_numeric(paired["volume"],          errors="coerce").dropna()
    bdens_vals  = pd.to_numeric(paired["buoyant_density"], errors="coerce").dropna()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Paired cell distributions  (n = {n_paired} pairs)", fontsize=13, fontweight="bold")

    def _hist_panel(ax, data, xlabel, color):
        if len(data) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            ax.set_xlabel(xlabel)
            return
        n_bins = min(60, max(10, len(data) // 20))
        ax.hist(data, bins=n_bins, color=color, edgecolor="white", linewidth=0.4)
        mean_v   = data.mean()
        median_v = data.median()
        ax.axvline(mean_v,   color="#C44E52", linewidth=1.2, label=f"mean   {mean_v:.4g}")
        ax.axvline(median_v, color="#8172B3", linewidth=1.2, linestyle="--",
                   label=f"median {median_v:.4g}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    _hist_panel(axes[0], mass_vals,  "Matched mass (pg)",       "#4C72B0")
    _hist_panel(axes[1], vol_vals,   "Volume (fL)",             "#55A868")
    _hist_panel(axes[2], bdens_vals, "Buoyant density (pg/fL)", "#DD8452")

    axes[0].set_title(f"Mass  (n = {len(mass_vals)})")
    axes[1].set_title(f"Volume  (n = {len(vol_vals)})")
    axes[2].set_title(f"Buoyant density  (n = {len(bdens_vals)})")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO]   Histogram figure: {output_path}")


# =============================================================================
# Paired/unpaired split helpers
# =============================================================================

def _prep_paired_split(vol_df: pd.DataFrame, mass_df: pd.DataFrame):
    """
    Returns (good_vol, vol_pb, mass_sorted, mass_pb):
      good_vol    — vol_df rows with no error and valid volume
      vol_pb      — boolean ndarray over good_vol positions (True = paired)
      mass_sorted — mass_df sorted by real_time_s, reset index
      mass_pb     — boolean ndarray over mass_sorted positions (True = paired)
    """
    good_mask = (
        (vol_df["error_code"].fillna("") == "")
        & pd.to_numeric(vol_df["calibrated_weighted_volume"], errors="coerce").notna()
    )
    good_vol = vol_df[good_mask].copy()
    vol_pb   = (pd.to_numeric(good_vol["mass_table_row"], errors="coerce") >= 0).values

    mass_sorted = mass_df.sort_values("real_time_s").reset_index(drop=True)
    paired_pos  = set(
        pd.to_numeric(vol_df["mass_table_row"], errors="coerce")
        .dropna().astype(int).tolist()
    )
    mass_pb = np.array([i in paired_pos for i in range(len(mass_sorted))], dtype=bool)
    return good_vol, vol_pb, mass_sorted, mass_pb


def _stats_annotation(all_vals: np.ndarray, paired_vals: np.ndarray) -> str:
    mu_all    = np.nanmean(all_vals)    if len(all_vals)    > 0 else np.nan
    mu_paired = np.nanmean(paired_vals) if len(paired_vals) > 0 else np.nan
    pct_diff  = (
        100 * (mu_paired - mu_all) / abs(mu_all)
        if (mu_all and not np.isnan(mu_all)) else np.nan
    )
    return (
        f"All mean:    {mu_all:.4g}\n"
        f"Paired mean: {mu_paired:.4g}\n"
        f"Δ: {pct_diff:+.1f}%"
    )


def plot_stacked_histograms(
    vol_df: pd.DataFrame, mass_df: pd.DataFrame, output_path: str
) -> None:
    """Stacked bar histograms: unpaired (bottom, red) + paired (top, green)."""
    good_vol, vol_pb, mass_sorted, mass_pb = _prep_paired_split(vol_df, mass_df)
    mass_all = pd.to_numeric(mass_sorted["mass_pg"], errors="coerce").values
    vol_all  = pd.to_numeric(good_vol["volume"],     errors="coerce").values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Mass and Volume — Stacked Histograms (paired vs unpaired)",
                 fontsize=12, fontweight="bold")

    def _panel(ax, vals, pb, xlabel, title):
        ok = ~np.isnan(vals)
        v_all = vals[ok]; p = pb[ok]
        vp = v_all[p]; vu = v_all[~p]
        if len(v_all) == 0:
            return
        lo, hi = np.nanpercentile(v_all, 1), np.nanpercentile(v_all, 99)
        bins = np.linspace(lo, hi, 51)
        cx   = (bins[:-1] + bins[1:]) / 2
        w    = bins[1] - bins[0]
        cp, _ = np.histogram(vp, bins=bins)
        cu, _ = np.histogram(vu, bins=bins)
        ax.bar(cx, cu, width=w, color="#C44E52",
               label=f"Unpaired  n={len(vu)}", align="center")
        ax.bar(cx, cp, width=w, color="#55A868",
               label=f"Paired  n={len(vp)}", align="center", bottom=cu)
        ax.set_xlabel(xlabel); ax.set_ylabel("Count"); ax.set_title(title)
        ax.legend(fontsize=8)
        ax.text(0.97, 0.95, _stats_annotation(v_all, vp), transform=ax.transAxes,
                fontsize=8, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

    _panel(ax1, mass_all, mass_pb, "Mass (pg)",    f"Mass  (n = {(~np.isnan(mass_all)).sum()})")
    _panel(ax2, vol_all,  vol_pb,  "Volume (fL)",  f"Volume  (n = {(~np.isnan(vol_all)).sum()})")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO]   Stacked histogram figure: {output_path}")


def plot_overlaid_histograms(
    vol_df: pd.DataFrame, mass_df: pd.DataFrame, output_path: str
) -> None:
    """Overlaid semi-transparent histograms: unpaired (red) vs paired (green)."""
    good_vol, vol_pb, mass_sorted, mass_pb = _prep_paired_split(vol_df, mass_df)
    mass_all = pd.to_numeric(mass_sorted["mass_pg"], errors="coerce").values
    vol_all  = pd.to_numeric(good_vol["volume"],     errors="coerce").values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Mass and Volume — Overlaid Histograms (paired vs unpaired)",
                 fontsize=12, fontweight="bold")

    def _panel(ax, vals, pb, xlabel, title):
        ok = ~np.isnan(vals)
        v_all = vals[ok]; p = pb[ok]
        vp = v_all[p]; vu = v_all[~p]
        if len(v_all) == 0:
            return
        lo, hi = np.nanpercentile(v_all, 1), np.nanpercentile(v_all, 99)
        bins   = np.linspace(lo, hi, 51)
        ax.hist(vu, bins=bins, color="#C44E52", alpha=0.55,
                label=f"Unpaired  n={len(vu)}", edgecolor="none")
        ax.hist(vp, bins=bins, color="#55A868", alpha=0.65,
                label=f"Paired  n={len(vp)}", edgecolor="none")
        ax.set_xlabel(xlabel); ax.set_ylabel("Count"); ax.set_title(title)
        ax.legend(fontsize=8)
        ax.text(0.97, 0.95, _stats_annotation(v_all, vp), transform=ax.transAxes,
                fontsize=8, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

    _panel(ax1, mass_all, mass_pb, "Mass (pg)",   f"Mass  (n = {(~np.isnan(mass_all)).sum()})")
    _panel(ax2, vol_all,  vol_pb,  "Volume (fL)", f"Volume  (n = {(~np.isnan(vol_all)).sum()})")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO]   Overlaid histogram figure: {output_path}")


def plot_scatter_vs_time(
    vol_df: pd.DataFrame, mass_df: pd.DataFrame, output_path: str
) -> None:
    """Volume and mass scatter vs measurement time; unpaired in red."""
    good_vol, vol_pb, mass_sorted, mass_pb = _prep_paired_split(vol_df, mass_df)
    vt   = pd.to_numeric(good_vol["_vtime"], errors="coerce").values
    vvol = pd.to_numeric(good_vol["volume"], errors="coerce").values
    mt   = mass_sorted["real_time_s"].values % 86400
    mpg  = pd.to_numeric(mass_sorted["mass_pg"], errors="coerce").values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Value vs Time  (red = unpaired)", fontsize=12, fontweight="bold")
    mk = dict(s=3, alpha=0.35, linewidths=0)

    ok_v = ~np.isnan(vt) & ~np.isnan(vvol)
    ax1.scatter(vt[ok_v & ~vol_pb],  vvol[ok_v & ~vol_pb],  color="#C44E52",
                label=f"Unpaired  n={np.sum(ok_v & ~vol_pb)}", **mk)
    ax1.scatter(vt[ok_v & vol_pb],   vvol[ok_v & vol_pb],   color="#4C72B0",
                label=f"Paired  n={np.sum(ok_v & vol_pb)}", **mk)
    ax1.set_xlabel("Transit time (s since midnight)")
    ax1.set_ylabel("Volume (fL)")
    ax1.set_title(f"Volume vs transit time  (n = {ok_v.sum()} good cells)")
    ax1.legend(fontsize=8, markerscale=4)

    ok_m = ~np.isnan(mpg)
    ax2.scatter(mt[ok_m & ~mass_pb],  mpg[ok_m & ~mass_pb],  color="#C44E52",
                label=f"Unpaired  n={np.sum(ok_m & ~mass_pb)}", **mk)
    ax2.scatter(mt[ok_m & mass_pb],   mpg[ok_m & mass_pb],   color="#4C72B0",
                label=f"Paired  n={np.sum(ok_m & mass_pb)}", **mk)
    ax2.set_xlabel("Measurement time (s since midnight)")
    ax2.set_ylabel("Mass (pg)")
    ax2.set_title(f"Mass vs time  (n = {len(mass_sorted)} measurements)")
    ax2.legend(fontsize=8, markerscale=4)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO]   Scatter vs time figure: {output_path}")


def plot_particle_time_histograms(
    vol_df: pd.DataFrame, mass_df: pd.DataFrame, output_path: str
) -> None:
    """Histograms of measurement times, overlaid paired/unpaired; mass/volume stats annotated."""
    good_vol, vol_pb, mass_sorted, mass_pb = _prep_paired_split(vol_df, mass_df)
    vt   = pd.to_numeric(good_vol["_vtime"], errors="coerce").values
    vvol = pd.to_numeric(good_vol["volume"], errors="coerce").values
    mt   = mass_sorted["real_time_s"].values % 86400
    mpg  = pd.to_numeric(mass_sorted["mass_pg"], errors="coerce").values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Particle Time Distributions (paired vs unpaired)",
                 fontsize=12, fontweight="bold")

    def _panel(ax, times, pb, vals, val_name, val_unit, time_xlabel, title):
        ok = ~np.isnan(times)
        t  = times[ok]; p = pb[ok]; v = vals[ok]
        tp = t[p];  vp = v[p]
        tu = t[~p]
        if len(t) == 0:
            return
        bins = np.linspace(t.min(), t.max(), 51)
        ax.hist(tu, bins=bins, color="#C44E52", alpha=0.55,
                label=f"Unpaired  n={len(tu)}", edgecolor="none")
        ax.hist(tp, bins=bins, color="#4C72B0", alpha=0.65,
                label=f"Paired  n={len(tp)}", edgecolor="none")
        ax.set_xlabel(time_xlabel); ax.set_ylabel("Count"); ax.set_title(title)
        ax.legend(fontsize=8)
        ok_v = ~np.isnan(v)
        txt = f"{val_name} ({val_unit}):\n" + _stats_annotation(v[ok_v], vp[~np.isnan(vp)])
        ax.text(0.97, 0.95, txt, transform=ax.transAxes, fontsize=8,
                va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

    _panel(ax1, mt, mass_pb, mpg, "Mass", "pg",
           "Measurement time (s since midnight)",
           f"Mass measurement times  (n = {len(mass_sorted)})")
    _panel(ax2, vt, vol_pb, vvol, "Volume", "fL",
           "Transit time (s since midnight)",
           f"Volume transit times  (n = {len(good_vol)})")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO]   Particle time histogram figure: {output_path}")


def plot_transit_time_scatters(
    vol_df: pd.DataFrame, mass_df: pd.DataFrame, output_path: str
) -> None:
    """Volume vs transit time and mass vs transit_t scatter; unpaired in red."""
    good_vol, vol_pb, mass_sorted, mass_pb = _prep_paired_split(vol_df, mass_df)
    vt   = pd.to_numeric(good_vol["_vtime"], errors="coerce").values
    vvol = pd.to_numeric(good_vol["volume"], errors="coerce").values
    mpg  = pd.to_numeric(mass_sorted["mass_pg"], errors="coerce").values

    t_col = "transit_t" if "transit_t" in mass_sorted.columns else "real_time_s"
    mt    = pd.to_numeric(mass_sorted[t_col], errors="coerce").values
    if t_col == "real_time_s":
        mt = mt % 86400
    mt_lbl = f"{t_col} (s)" if t_col != "real_time_s" else "Measurement time since midnight (s)"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Transit Time Scatters  (red = unpaired)", fontsize=12, fontweight="bold")
    mk = dict(s=3, alpha=0.35, linewidths=0)

    ok_v = ~np.isnan(vt) & ~np.isnan(vvol)
    ax1.scatter(vt[ok_v & ~vol_pb],  vvol[ok_v & ~vol_pb],  color="#C44E52",
                label=f"Unpaired  n={np.sum(ok_v & ~vol_pb)}", **mk)
    ax1.scatter(vt[ok_v & vol_pb],   vvol[ok_v & vol_pb],   color="#4C72B0",
                label=f"Paired  n={np.sum(ok_v & vol_pb)}", **mk)
    ax1.set_xlabel("Transit time (s since midnight)")
    ax1.set_ylabel("Volume (fL)")
    ax1.set_title(f"Volume vs transit time  (n = {ok_v.sum()} good cells)")
    ax1.legend(fontsize=8, markerscale=4)

    ok_m = ~np.isnan(mt) & ~np.isnan(mpg)
    ax2.scatter(mt[ok_m & ~mass_pb],  mpg[ok_m & ~mass_pb],  color="#C44E52",
                label=f"Unpaired  n={np.sum(ok_m & ~mass_pb)}", **mk)
    ax2.scatter(mt[ok_m & mass_pb],   mpg[ok_m & mass_pb],   color="#4C72B0",
                label=f"Paired  n={np.sum(ok_m & mass_pb)}", **mk)
    ax2.set_xlabel(mt_lbl)
    ax2.set_ylabel("Mass (pg)")
    ax2.set_title(f"Mass vs {t_col}  (n = {ok_m.sum()} measurements)")
    ax2.legend(fontsize=8, markerscale=4)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO]   Transit time scatter figure: {output_path}")


# =============================================================================
# Per-folder pairing entry point
# =============================================================================

def pair_one(
    analysis_dir: str,
    vol_dir_name: Optional[str] = None,
    mass_dir_name: Optional[str] = None,
    timebase: float = 1e-3,
    peak_tolerance: int = 11,
    gaussian_width: int = 15,
    utc_offset_hours: float = 0.0,
) -> str:
    """
    Run full pairing for a single experiment folder.  Returns the path of the
    output CSV.  Raises on any error.
    """
    analysis_dir = os.path.abspath(analysis_dir)

    if vol_dir_name is None:
        vol_dir_name = find_most_recent_dir(analysis_dir, "_imaging_fxm_results")
        print(f"[INFO]   Auto-selected imaging dir : {vol_dir_name}")

    vol_dir    = os.path.join(analysis_dir, vol_dir_name)
    stage2_dir = os.path.join(vol_dir, "stage2_analysis")
    if not os.path.isdir(stage2_dir):
        raise FileNotFoundError(
            f"stage2_analysis not found in {vol_dir}"
        )

    if mass_dir_name is None:
        mass_dir_name = find_most_recent_dir(analysis_dir, "_mass_results")
        print(f"[INFO]   Auto-selected mass dir    : {mass_dir_name}")

    mass_dir = os.path.join(analysis_dir, mass_dir_name)

    vol_csv_path   = find_file_in_dir(stage2_dir, "*_ProcessedVolumes.csv")
    frame_csv_path = find_file_in_dir(stage2_dir, "*_FrameVolumes.csv")
    mass_csv_path  = find_mass_csv(mass_dir)

    print(f"[INFO]   ProcessedVolumes : {vol_csv_path}")
    print(f"[INFO]   FrameVolumes     : {frame_csv_path}")
    print(f"[INFO]   Mass CSV         : {mass_csv_path}")

    vol_df   = pd.read_csv(vol_csv_path)
    frame_df = pd.read_csv(frame_csv_path)
    mass_df  = pd.read_csv(mass_csv_path)

    transit_times = frame_df.groupby("transit_index")["frame_time"].min()

    print(
        f"[INFO]   Pairing {len(vol_df)} volume rows against "
        f"{len(mass_df)} mass measurements..."
    )
    seg_lags = pair_mass_and_volumes(
        vol_df, transit_times, mass_df,
        timebase=timebase,
        peak_tolerance=peak_tolerance,
        gaussian_width=gaussian_width,
        utc_offset_hours=utc_offset_hours,
    )

    # attach timing to all vol_df rows so scatter/time-histogram plots can use it
    vol_df["_vtime"] = transit_times.reindex(vol_df["transit_index"].values).values

    ts_now   = datetime.now().strftime("%Y%m%d.%H%M%S")
    out_dir  = os.path.join(analysis_dir, f"{ts_now}_pairing_results")
    os.makedirs(out_dir, exist_ok=True)
    prefix   = os.path.basename(analysis_dir)

    out_csv            = os.path.join(out_dir, f"{prefix}_PairedSMRVolumes.csv")
    stats_fig          = os.path.join(out_dir, f"{prefix}_PairingStats_fig.png")
    lags_fig           = os.path.join(out_dir, f"{prefix}_PairingLags_fig.png")
    hist_fig           = os.path.join(out_dir, f"{prefix}_PairingHistograms_fig.png")
    stacked_fig        = os.path.join(out_dir, f"{prefix}_StackedHistograms_fig.png")
    overlaid_fig       = os.path.join(out_dir, f"{prefix}_OverlaidHistograms_fig.png")
    scatter_time_fig   = os.path.join(out_dir, f"{prefix}_ScatterVsTime_fig.png")
    particle_time_fig  = os.path.join(out_dir, f"{prefix}_ParticleTimeHistograms_fig.png")
    transit_scatter_fig= os.path.join(out_dir, f"{prefix}_TransitTimeScatter_fig.png")

    vol_df["vol_csv_row"] = vol_df.index
    vol_df.drop(columns=["_vtime"], errors="ignore").to_csv(out_csv, index=False)
    print(f"[INFO]   Paired CSV: {out_csv}")

    plot_pairing_stats(vol_df, mass_df, stats_fig)
    plot_pairing_lags(vol_df, seg_lags, lags_fig)
    plot_pairing_histograms(vol_df, hist_fig)
    plot_stacked_histograms(vol_df, mass_df, stacked_fig)
    plot_overlaid_histograms(vol_df, mass_df, overlaid_fig)
    plot_scatter_vs_time(vol_df, mass_df, scatter_time_fig)
    plot_particle_time_histograms(vol_df, mass_df, particle_time_fig)
    plot_transit_time_scatters(vol_df, mass_df, transit_scatter_fig)

    return out_csv


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch SMR mass / FXM volume pairing across multiple experiment folders.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "root_dir",
        help="Root directory containing date-named experiment superfolders.",
    )

    sel = parser.add_argument_group("folder selection")
    sel.add_argument(
        "--from", dest="date_from", metavar="YYYY-MM-DD",
        help="Only process superfolders dated on or after this date.",
    )
    sel.add_argument(
        "--to", dest="date_to", metavar="YYYY-MM-DD",
        help="Only process superfolders dated on or before this date.",
    )
    sel.add_argument(
        "--last", dest="last_n", type=int, metavar="N",
        help="Only process the N most recently dated superfolders.",
    )
    sel.add_argument(
        "--from-file", metavar="FILE",
        help=(
            "Text file listing experiment folder paths (absolute or relative to "
            "root_dir), one per line.  Lines starting with # are ignored."
        ),
    )
    sel.add_argument(
        "--skip-paired", action="store_true",
        help="Skip experiment folders that already contain a *_PairedSMRVolumes.csv file.",
    )
    sel.add_argument(
        "--no-recursive", dest="recursive", action="store_false",
        help="Restrict discovery to fixed depth=2 (default is recursive search).",
    )
    parser.set_defaults(recursive=True)
    sel.add_argument(
        "--dry-run", action="store_true",
        help="Print discovered folders without running any pairing.",
    )

    pair = parser.add_argument_group("pairing options")
    pair.add_argument(
        "--timebase", type=float, default=1e-3,
        help="Time axis resolution in seconds (default: 1e-3).",
    )
    pair.add_argument(
        "--peak-tolerance", type=int, default=11,
        help="Index window around each mass peak for matching (default: 11).",
    )
    pair.add_argument(
        "--gaussian-width", type=int, default=15,
        help="Gaussian blur sigma in samples (default: 15).",
    )
    pair.add_argument(
        "--utc-offset", type=float, default=0.0, metavar="HOURS",
        help=(
            "Hours to add to mass real_time_s timestamps to convert to the same "
            "timezone as FXM frame_times (e.g. -4 for EDT when LabVIEW logs UTC). "
            "Default: 0."
        ),
    )

    args     = parser.parse_args()
    root_dir = os.path.normpath(args.root_dir)

    if not os.path.isdir(root_dir):
        print(f"[ERROR] root_dir does not exist: {root_dir}", file=sys.stderr)
        sys.exit(1)

    # ── discover experiment folders ──────────────────────────────────────
    if args.from_file:
        experiments = _load_from_file(root_dir, args.from_file)
    else:
        date_from = (
            datetime.strptime(args.date_from, "%Y-%m-%d").date()
            if args.date_from else None
        )
        date_to = (
            datetime.strptime(args.date_to, "%Y-%m-%d").date()
            if args.date_to else None
        )
        experiments = list(discover_experiments(
            root_dir,
            date_from=date_from,
            date_to=date_to,
            last_n=args.last_n,
            skip_paired=args.skip_paired,
            recursive=args.recursive,
        ))

    if not experiments:
        print("[INFO] No experiment folders found matching the given criteria.")
        return

    print(f"[INFO] Found {len(experiments)} experiment folder(s):")
    for folder in experiments:
        print(f"  {folder}")

    if args.dry_run:
        print("[INFO] Dry run -- no pairing performed.")
        return

    # ── run pairing per folder ───────────────────────────────────────────
    results  = {}
    n_exp    = len(experiments)
    for i, folder in enumerate(experiments):
        print(f"\n{'#' * 60}")
        print(f"# [{i+1}/{n_exp}] {folder}")
        print(f"{'#' * 60}")
        try:
            out_csv = pair_one(
                folder,
                timebase=args.timebase,
                peak_tolerance=args.peak_tolerance,
                gaussian_width=args.gaussian_width,
                utc_offset_hours=args.utc_offset,
            )
            results[folder] = ("OK", out_csv)
        except Exception as exc:
            print(f"[ERROR] {exc}")
            results[folder] = ("FAILED", str(exc))

    # ── summary ──────────────────────────────────────────────────────────
    n_ok   = sum(1 for s, _ in results.values() if s == "OK")
    n_fail = len(results) - n_ok
    print(f"\n{'=' * 60}")
    print("BULK PAIRING SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {n_ok}/{len(results)} succeeded, {n_fail} failed\n")
    for folder, (status, detail) in results.items():
        print(f"  [{status:6s}] {folder}")
        if status == "FAILED":
            print(f"           {detail}")


if __name__ == "__main__":
    main()
