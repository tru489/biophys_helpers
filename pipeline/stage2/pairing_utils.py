"""
pairing_utils.py — Shared cross-correlation pairing primitives.

Provides the signal-building and cross-correlation logic used by the
pair_smr_volumes and bulk_pair_smr_volumes scripts, keeping it in one place
so the two scripts don't duplicate it.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import correlate, fftconvolve


def gaussian_kernel(gw: int) -> np.ndarray:
    t_g = np.arange(-3 * gw, 3 * gw + 1, dtype=float)
    return np.exp(-(t_g ** 2) / (gw ** 2))


def build_mass_signal(
    all_times: np.ndarray,
    all_mass: np.ndarray,
    timebase: float,
    peak_tolerance: int,
    kern: np.ndarray,
):
    """
    Build a blurred mass-presence signal and a 1-based mass-index signal on a
    uniform time axis.

    Returns:
        t_axis        – 1-D time axis array
        mass_blurred  – Gaussian-blurred binary presence signal
        mass_idx_sig  – 1-based index signal (window of ±peak_tolerance around each peak)
        t_idx         – integer bin indices of each mass measurement
    """
    start_t = float(np.nanmin(all_times)) - 0.1
    end_t   = float(np.nanmax(all_times)) + 0.1
    if end_t <= start_t:
        raise ValueError(
            f"Mass time axis is empty after % 86400 (start={start_t:.1f}, end={end_t:.1f}). "
            f"This usually means mass timestamps span midnight — check real_time_s values."
        )
    t_axis  = np.arange(start_t, end_t, timebase)

    mass_sig  = np.zeros(len(t_axis))
    t_idx     = np.round((all_times - start_t) / timebase).astype(int)
    t_idx     = np.clip(t_idx, 0, len(t_axis) - 1)
    mass_vals = np.where(np.isnan(all_mass), 0.0, all_mass)
    for i, ti in enumerate(t_idx):
        mass_sig[ti] = mass_vals[i]

    norm_mass_sig = (mass_sig > 0).astype(float)
    mass_blurred  = fftconvolve(norm_mass_sig, kern, mode="same")

    mass_idx_sig = np.zeros(len(t_axis), dtype=int)
    pt = peak_tolerance
    for ii, ti in enumerate(t_idx):
        lo = max(0, ti - pt)
        hi = min(len(t_axis) - 1, ti + pt)
        mass_idx_sig[lo : hi + 1] = ii + 1

    return t_axis, mass_blurred, mass_idx_sig, t_idx


def build_vol_signal(
    vtimes: np.ndarray,
    vols: np.ndarray,
    timebase: float,
    kern: np.ndarray,
):
    """
    Build a blurred volume-presence signal on a local time axis.

    Returns:
        v_axis    – 1-D time axis for this segment
        vi        – integer bin indices of each volume measurement
        v_blurred – Gaussian-blurred binary presence signal
    """
    sv     = vtimes[0] - 1.0
    ev     = vtimes[-1] + 1.0
    v_axis = np.arange(sv, ev, timebase)
    v_sig  = np.zeros(len(v_axis))
    vi     = np.round((vtimes - sv) / timebase).astype(int)
    vi     = np.clip(vi, 0, len(v_axis) - 1)
    v_vals = np.where(np.isnan(vols), 0.0, vols)
    for i, tii in enumerate(vi):
        v_sig[tii] = v_vals[i]

    norm_v_sig = (v_sig > 0).astype(float)
    v_blurred  = fftconvolve(norm_v_sig, kern, mode="same")
    return v_axis, vi, v_blurred


def xcorr_lag(
    mass_blurred: np.ndarray,
    v_blurred: np.ndarray,
    t_axis: np.ndarray,
    v_axis: np.ndarray,
    timebase: float,
):
    """
    Cross-correlate mass and volume blurred signals to find the best-fit
    temporal offset.

    Returns:
        offset_idx     – integer lag in samples
        offset_s       – total time offset in seconds (includes clock offset)
        fine_lag_s     – sub-hour residual lag (integer-hour component stripped)
        clock_offset_s – nearest integer-hour component of offset_s
    """
    xcorr         = correlate(mass_blurred, v_blurred, mode="full")
    lags_arr      = np.arange(-(len(v_blurred) - 1), len(mass_blurred))
    offset_idx    = int(lags_arr[np.argmax(xcorr)])
    offset_s      = offset_idx * timebase + t_axis[0] - v_axis[0]
    clock_offset_s = round(offset_s / 3600) * 3600
    fine_lag_s     = offset_s - clock_offset_s
    return offset_idx, offset_s, fine_lag_s, clock_offset_s


def make_vol_idx_signal(
    vi: np.ndarray,
    offset_idx: int,
    t_axis_len: int,
) -> np.ndarray:
    """
    Build a 1-based volume-index signal on the mass time axis by shifting vi
    by offset_idx samples.
    """
    vol_idx_sig = np.zeros(t_axis_len, dtype=int)
    for i, tii in enumerate(vi):
        shifted = tii + offset_idx
        if 0 <= shifted < t_axis_len:
            vol_idx_sig[shifted] = i + 1
    return vol_idx_sig


def find_match_pairs(
    mass_idx_sig: np.ndarray,
    vol_idx_sig: np.ndarray,
) -> list[tuple[int, int]]:
    """
    Return a list of (mi, vi2) 0-based index pairs where both signals are
    non-zero (i.e. a mass peak and a volume peak overlap after alignment).
    """
    positions = np.where((mass_idx_sig != 0) & (vol_idx_sig != 0))[0]
    return [(int(mass_idx_sig[p]) - 1, int(vol_idx_sig[p]) - 1) for p in positions]
