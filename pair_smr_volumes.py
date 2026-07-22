#!/usr/bin/env python3
"""
pair_smr_volumes.py — Standalone SMR mass / FXM volume pairing script.

Reads a pre-computed ProcessedVolumes.csv (from a volume-only analysis) and a
mass CSV from a *_mass_results directory, runs a cross-correlation pairing of
the calibrated-volume-vs-time and SMR-mass signals, and writes a fully-populated
ProcessedVolumes-format CSV plus diagnostic figures.

Usage:
    python pair_smr_volumes.py <analysis_dir>
        [--vol-dir  DIRNAME]     name of imaging results dir (auto-detected if omitted)
        [--mass-dir DIRNAME]     name of mass results dir (auto-detected if omitted)
        [--timebase FLOAT]       time axis resolution in seconds (default: 1e-3)
        [--peak-tolerance INT]   index window around each mass peak (default: 11)
        [--gaussian-width INT]   Gaussian blur sigma in samples (default: 15)
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from datetime import datetime
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


# ──────────────────────────────────────────────────────────────────────────────
# Directory / file discovery
# ──────────────────────────────────────────────────────────────────────────────

_TS_PATTERNS = [
    (re.compile(r"^(\d{8}_\d{6})_"), "%Y%m%d_%H%M%S"),   # 20260606_025232_…
    (re.compile(r"^(\d{8}\.\d{6})_"), "%Y%m%d.%H%M%S"),  # 20260607.150735_…
]


def _parse_timestamp(dirname: str) -> Optional[datetime]:
    for pat, fmt in _TS_PATTERNS:
        m = pat.match(os.path.basename(dirname))
        if m:
            try:
                return datetime.strptime(m.group(1), fmt)
            except ValueError:
                pass
    return None


def find_most_recent_dir(parent: str, suffix: str) -> str:
    """Return the name of the subdirectory of *parent* whose name ends with
    *suffix* and has the most-recent timestamp prefix."""
    candidates = [
        d for d in os.listdir(parent)
        if d.endswith(suffix) and os.path.isdir(os.path.join(parent, d))
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No directory ending with '{suffix}' found in:\n  {parent}"
        )

    def _key(d: str) -> datetime:
        ts = _parse_timestamp(d)
        return ts if ts is not None else datetime.min

    return max(candidates, key=_key)


def find_file_in_dir(directory: str, pattern: str) -> str:
    """Glob for *pattern* inside *directory*; return the first match or raise."""
    matches = glob.glob(os.path.join(directory, pattern))
    if not matches:
        raise FileNotFoundError(
            f"No file matching '{pattern}' found in:\n  {directory}"
        )
    return matches[0]


def find_mass_csv(mass_dir: str) -> str:
    """Return the main mass CSV in *mass_dir*, skipping curation_index.csv."""
    csvs = [
        f for f in glob.glob(os.path.join(mass_dir, "*.csv"))
        if not os.path.basename(f).startswith("curation_index")
    ]
    if not csvs:
        raise FileNotFoundError(f"No mass CSV found in:\n  {mass_dir}")
    # prefer largest file if multiple candidates exist
    return max(csvs, key=os.path.getsize)


# ──────────────────────────────────────────────────────────────────────────────
# Pairing algorithm  (uses shared primitives from pipeline.stage2.pairing_utils)
# ──────────────────────────────────────────────────────────────────────────────

def pair_mass_and_volumes(
    vol_df: pd.DataFrame,
    transit_times: pd.Series,   # Series[float]: index=transit_index → first frame_time (s since midnight)
    mass_df: pd.DataFrame,
    timebase: float = 1e-3,
    peak_tolerance: int = 11,
    gaussian_width: int = 15,
    utc_offset_hours: float = 0.0,
) -> list[dict]:
    """
    Populate mass-related columns in *vol_df* (modified in place) by
    cross-correlating the calibrated-volume-vs-time signal with the SMR mass
    signal.

    Returns a list of per-segment lag diagnostic dicts.
    """
    # coerce mass-field columns to correct dtypes so .loc assignments work cleanly
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
    # string columns: ensure object dtype so str values can be assigned
    for col in ("sample", "sample_id", "condition"):
        if col in vol_df.columns:
            vol_df[col] = vol_df[col].astype(object)

    kern = _gaussian_kernel(gaussian_width)

    # ── build mass signal on a fixed time axis ────────────────────────────
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

    # ── select good volume rows, attach timing ────────────────────────────
    good_mask = (
        (vol_df["error_code"].fillna("") == "")
        & pd.to_numeric(vol_df["calibrated_weighted_volume"], errors="coerce").notna()
    )
    good_df = vol_df[good_mask].copy()
    good_df["_vtime"] = transit_times.reindex(good_df["transit_index"].values).values

    # keep original vol_df index (needed for .loc writes) — do NOT reset_index
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
        seg = good_df.iloc[s:e]   # retains original vol_df index
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
            cell_row     = seg.iloc[vi2]
            mass_row     = mass_sorted.iloc[mi]
            orig_idx     = cell_row.name  # original vol_df index

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


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostic figures
# ──────────────────────────────────────────────────────────────────────────────

def _paired_counts(vol_df: pd.DataFrame, mass_df: pd.DataFrame) -> dict:
    n_vol_good    = int((vol_df["error_code"].fillna("") == "").sum())
    paired_mask   = pd.to_numeric(vol_df["mass_table_row"], errors="coerce") >= 0
    n_vol_paired  = int(paired_mask.sum())

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


def plot_pairing_stats(
    vol_df: pd.DataFrame, mass_df: pd.DataFrame, output_path: str
) -> None:
    """Three-panel figure: grouped bar + two pie charts."""
    c = _paired_counts(vol_df, mass_df)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("SMR–FXM Pairing Statistics", fontsize=13, fontweight="bold")

    # ── left: grouped bar ────────────────────────────────────────────────
    labels    = ["Volume\ntransits", "Mass\nmeasurements"]
    totals    = [c["n_vol_good"],    c["n_mass_total"]]
    paired    = [c["n_vol_paired"],  c["n_mass_paired"]]
    unmatched = [c["n_vol_unpaired"],c["n_mass_unpaired"]]
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

    # ── middle: volume pie ───────────────────────────────────────────────
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

    # ── right: mass pie ──────────────────────────────────────────────────
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


def plot_pairing_lags(
    vol_df: pd.DataFrame, seg_lags: list[dict], output_path: str
) -> None:
    """Two-panel figure: per-segment cross-correlation lag + per-cell residuals."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle("SMR–FXM Pairing Lag Diagnostics", fontsize=13, fontweight="bold")

    # ── top: per-segment fine lag (clock/timezone offset stripped) ──────
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

    # ── bottom: per-cell timing difference (clock offset stripped) ──────
    # volume_time - (mass_time_midnight + clock_offset) centres the
    # distribution near fine_lag_s, showing the actual matching scatter.
    mean_clock = (
        np.mean([d["clock_offset_s"] for d in seg_lags]) if seg_lags else 0.0
    )
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
        ax.set_title(f"n = {len(data)}")

    _hist_panel(axes[0], mass_vals,  "Matched mass (pg)",         "#4C72B0")
    _hist_panel(axes[1], vol_vals,   "Volume (fL)",               "#55A868")
    _hist_panel(axes[2], bdens_vals, "Buoyant density (pg/fL)",   "#DD8452")

    axes[0].set_title(f"Mass  (n = {len(mass_vals)})")
    axes[1].set_title(f"Volume  (n = {len(vol_vals)})")
    axes[2].set_title(f"Buoyant density  (n = {len(bdens_vals)})")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO]  Histogram figure: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Pair pre-computed SMR mass data with FXM volume data.\n"
            "Produces a paired ProcessedVolumes CSV and diagnostic figures."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "analysis_dir",
        help="Directory containing the *_imaging_fxm_results and *_mass_results subdirs.",
    )
    p.add_argument(
        "--vol-dir", default=None, metavar="DIRNAME",
        help="Name of the imaging/FXM results dir (auto-detected if omitted).",
    )
    p.add_argument(
        "--mass-dir", default=None, metavar="DIRNAME",
        help="Name of the mass results dir (auto-detected if omitted).",
    )
    p.add_argument(
        "--timebase", type=float, default=1e-3,
        help="Time axis resolution in seconds (default: 1e-3).",
    )
    p.add_argument(
        "--peak-tolerance", type=int, default=11,
        help="Index window around each mass peak for matching (default: 11).",
    )
    p.add_argument(
        "--gaussian-width", type=int, default=15,
        help="Gaussian blur sigma in samples (default: 15).",
    )
    p.add_argument(
        "--utc-offset", type=float, default=0.0, metavar="HOURS",
        help=(
            "Hours to add to mass real_time_s timestamps to convert to the same "
            "timezone as FXM frame_times (e.g. -4 for EDT when LabVIEW logs UTC). "
            "Default: 0."
        ),
    )
    return p.parse_args()


def main() -> None:
    args         = parse_args()
    analysis_dir = os.path.abspath(args.analysis_dir)

    if not os.path.isdir(analysis_dir):
        print(f"[ERROR] analysis_dir does not exist: {analysis_dir}", file=sys.stderr)
        sys.exit(1)

    # ── resolve volume dir ───────────────────────────────────────────────
    if args.vol_dir:
        vol_dir_name = args.vol_dir
    else:
        vol_dir_name = find_most_recent_dir(analysis_dir, "_imaging_fxm_results")
        print(f"[INFO] Auto-selected imaging dir : {vol_dir_name}")

    vol_dir    = os.path.join(analysis_dir, vol_dir_name)
    stage2_dir = os.path.join(vol_dir, "stage2_analysis")

    if not os.path.isdir(stage2_dir):
        print(
            f"[ERROR] Expected stage2_analysis subdir not found in:\n  {vol_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── resolve mass dir ─────────────────────────────────────────────────
    if args.mass_dir:
        mass_dir_name = args.mass_dir
    else:
        mass_dir_name = find_most_recent_dir(analysis_dir, "_mass_results")
        print(f"[INFO] Auto-selected mass dir    : {mass_dir_name}")

    mass_dir = os.path.join(analysis_dir, mass_dir_name)

    # ── locate input files ───────────────────────────────────────────────
    vol_csv_path   = find_file_in_dir(stage2_dir, "*_ProcessedVolumes.csv")
    frame_csv_path = find_file_in_dir(stage2_dir, "*_FrameVolumes.csv")
    mass_csv_path  = find_mass_csv(mass_dir)

    print(f"[INFO] ProcessedVolumes : {vol_csv_path}")
    print(f"[INFO] FrameVolumes     : {frame_csv_path}")
    print(f"[INFO] Mass CSV         : {mass_csv_path}")

    # ── load data ────────────────────────────────────────────────────────
    print("[INFO] Loading data...")
    vol_df   = pd.read_csv(vol_csv_path)
    frame_df = pd.read_csv(frame_csv_path)
    mass_df  = pd.read_csv(mass_csv_path)

    # first frame_time per transit → used as the transit's time coordinate
    transit_times = frame_df.groupby("transit_index")["frame_time"].min()

    # ── run pairing ──────────────────────────────────────────────────────
    print(
        f"[INFO] Pairing {len(vol_df)} volume rows against "
        f"{len(mass_df)} mass measurements..."
    )
    seg_lags = pair_mass_and_volumes(
        vol_df,
        transit_times,
        mass_df,
        timebase=args.timebase,
        peak_tolerance=args.peak_tolerance,
        gaussian_width=args.gaussian_width,
        utc_offset_hours=args.utc_offset,
    )

    # ── create output subdir ─────────────────────────────────────────────
    ts_now   = datetime.now().strftime("%Y%m%d.%H%M%S")
    out_dir  = os.path.join(analysis_dir, f"{ts_now}_pairing_results")
    os.makedirs(out_dir, exist_ok=True)
    prefix   = os.path.basename(analysis_dir)

    # ── write output CSV ─────────────────────────────────────────────────
    out_csv = os.path.join(out_dir, f"{prefix}_PairedSMRVolumes.csv")
    vol_df["vol_csv_row"] = vol_df.index
    vol_df.to_csv(out_csv, index=False)
    print(f"[INFO] Paired CSV: {out_csv}")

    # ── diagnostic figures ───────────────────────────────────────────────
    stats_fig = os.path.join(out_dir, f"{prefix}_PairingStats_fig.png")
    lags_fig  = os.path.join(out_dir, f"{prefix}_PairingLags_fig.png")
    hist_fig  = os.path.join(out_dir, f"{prefix}_PairingHistograms_fig.png")
    plot_pairing_stats(vol_df, mass_df, stats_fig)
    plot_pairing_lags(vol_df, seg_lags, lags_fig)
    plot_pairing_histograms(vol_df, hist_fig)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
