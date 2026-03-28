"""
crop_smr_data.py

Interactive GUI for cropping junk data from SMR binary run files. A directory
containing three binary files (prefix_frequencies, prefix_valvestates,
prefix_time) is loaded. The data is displayed chunk-by-chunk so that large
datasets (millions to tens of millions of data points) can be navigated
efficiently. The user marks one or more removal regions by clicking on the
frequency-vs-time plot, then saves the cropped binary files to a new output
directory.

Workflow:
    1. A directory is supplied on the command line.
    2. The three binary files are discovered and loaded into memory.
    3. An interactive GUI shows the frequency signal in fixed-size chunks.
       Prev / Next buttons navigate between chunks.
    4. To mark a removal region, click "Set lower boundary" then click on the
       plot to place the lower boundary (red dashed line). Then click "Set
       upper boundary" and click on the plot to place the upper boundary.
       The region between them is shaded red.
    5. Multiple regions may be added across any chunks. All committed regions
       are shown as shaded spans on whichever chunk they intersect.
    6. "Delete" removes any committed region.
    7. "Crop & Save" writes the three cropped binary files (and copies any
       .json / image files) to a new directory named <input_dir>_cropped/ at
       the same level as the input directory.

Usage:
    python crop_smr_data.py <directory> [--chunk-size N]

    <directory>       Path containing the three SMR binary files
    --chunk-size N    Number of data points per displayed chunk (default: 100000)
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import numpy as np
import tkinter as tk
from tkinter import messagebox


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_cli_args():
    """
    Parses CLI arguments.

    Raises:
        FileNotFoundError: directory does not exist

    Returns:
        tuple[Path, int]: (input directory, chunk size)
    """
    parser = argparse.ArgumentParser(
        description="Interactively crop junk data from SMR binary run files."
    )
    parser.add_argument('directory', type=str,
                        help='Path to directory containing SMR binary files')
    parser.add_argument('--chunk-size', type=int, default=100_000,
                        metavar='N',
                        help='Number of data points per displayed chunk '
                             '(default: 100000)')
    args = parser.parse_args()
    d = Path(args.directory)
    if not d.is_dir():
        raise FileNotFoundError(f"Directory not found: {d}")
    return d, args.chunk_size


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

_FILE_PATTERN = re.compile(r'^\d+\.\d+_(frequencies|valvestates|time)$')


def _find_smr_files(directory: Path):
    """
    Scans directory for the three SMR binary files.

    Expected filename format: <digits>.<digits>_<suffix>
    where suffix is one of: frequencies, valvestates, time

    Raises:
        FileNotFoundError: one or more expected file types are absent
        ValueError:        more than one file found for a given suffix

    Returns:
        tuple[Path, Path, Path]: (frequencies_path, valvestates_path, time_path)
    """
    found = {'frequencies': [], 'valvestates': [], 'time': []}
    for f in directory.iterdir():
        m = _FILE_PATTERN.match(f.name)
        if m:
            found[m.group(1)].append(f)

    for suffix, matches in found.items():
        if len(matches) == 0:
            raise FileNotFoundError(
                f"No '{suffix}' file found in {directory}")
        if len(matches) > 1:
            raise ValueError(
                f"Multiple '{suffix}' files found in {directory}: "
                + ', '.join(p.name for p in sorted(matches)))

    return found['frequencies'][0], found['valvestates'][0], found['time'][0]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_data(freq_path: Path, vs_path: Path, time_path: Path):
    """
    Reads all three binary files into numpy arrays.

    File formats (matching MATLAB fread conventions in SMR software):
        frequencies  — big-endian float64, 8 bytes/value
        valvestates  — uint8,              1 byte/value
        time         — big-endian float64, 8 bytes/value

    Raises:
        ValueError: arrays do not have equal length

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: (freq, valvestates, time)
    """
    print(f"Loading {freq_path.name} ...", end=' ', flush=True)
    freq = np.fromfile(freq_path, dtype='>f8')
    print(f"{len(freq):,} points")

    print(f"Loading {vs_path.name} ...", end=' ', flush=True)
    vs = np.fromfile(vs_path, dtype=np.uint8)
    print(f"{len(vs):,} points")

    print(f"Loading {time_path.name} ...", end=' ', flush=True)
    time = np.fromfile(time_path, dtype='>f8')
    print(f"{len(time):,} points")

    if not (len(freq) == len(vs) == len(time)):
        raise ValueError(
            f"Array length mismatch: frequencies={len(freq)}, "
            f"valvestates={len(vs)}, time={len(time)}")
    return freq, vs, time


# ---------------------------------------------------------------------------
# Main GUI window
# ---------------------------------------------------------------------------

class CropWindow:
    """
    Main application window for interactive cropping of SMR binary data.

    Displays frequency vs time in fixed-size chunks. The user navigates
    between chunks and marks removal regions by clicking on the plot. Each
    region is defined by a lower and upper index boundary (mapped from the
    nearest time value to a global array index). Committed regions are
    displayed as red shaded spans on all intersecting chunks.

    State machine for boundary setting:
        0 — idle:            "Set lower" enabled,  "Set upper" disabled
        1 — awaiting lower:  both disabled  (next plot click sets lower index)
        2 — lower set:       "Set lower" disabled,  "Set upper" enabled
        3 — awaiting upper:  both disabled  (next plot click sets upper index)
        After upper click:   region appended, refresh regions list, → state 0
    """

    def __init__(self, root: tk.Tk, directory: Path,
                 freq: np.ndarray, vs: np.ndarray, time: np.ndarray,
                 chunk_size: int,
                 freq_path: Path, vs_path: Path, time_path: Path):
        self._root = root
        self._directory = directory
        self._freq = freq
        self._vs = vs
        self._time = time
        self._chunk_size = chunk_size
        self._freq_path = freq_path
        self._vs_path = vs_path
        self._time_path = time_path

        self._n = len(freq)
        self._n_chunks = max(1, (self._n + chunk_size - 1) // chunk_size)
        self._chunk = 0
        # Subtract global time offset so x-axis values are small floats,
        # preventing matplotlib's tick locator from overflowing.
        self._t0 = float(time[0]) if len(time) > 0 else 0.0

        self._state = 0
        self._pending_lower_idx = None

        # Committed regions: list of [lower_idx, upper_idx] (inclusive)
        self._regions = []

        root.title(f"crop_smr_data  —  {directory.name}")
        self._build_ui()
        self._draw_chunk()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        """Builds all tkinter widgets and embeds the matplotlib canvas."""
        # --- Plot ---
        self._fig, self._ax = plt.subplots(figsize=(14, 4))
        canvas = FigureCanvasTkAgg(self._fig, master=self._root)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=6, pady=(6, 0))
        self._canvas = canvas
        self._cid = canvas.mpl_connect('button_press_event', self._on_click)

        # --- Navigation ---
        nav_frame = tk.Frame(self._root)
        nav_frame.pack(fill=tk.X, padx=8, pady=4)
        self._prev_btn = tk.Button(nav_frame, text="< Prev",
                                   command=self._prev_chunk)
        self._prev_btn.pack(side=tk.LEFT)
        self._chunk_label = tk.Label(nav_frame, text='', width=18)
        self._chunk_label.pack(side=tk.LEFT, padx=8)
        self._next_btn = tk.Button(nav_frame, text="Next >",
                                   command=self._next_chunk)
        self._next_btn.pack(side=tk.LEFT)

        # --- Boundary controls ---
        ctrl_frame = tk.Frame(self._root)
        ctrl_frame.pack(fill=tk.X, padx=8, pady=2)
        self._lower_btn = tk.Button(ctrl_frame, text="Set lower boundary",
                                    command=self._activate_lower)
        self._lower_btn.pack(side=tk.LEFT, padx=(0, 6))
        self._upper_btn = tk.Button(ctrl_frame, text="Set upper boundary",
                                    state=tk.DISABLED,
                                    command=self._activate_upper)
        self._upper_btn.pack(side=tk.LEFT)
        self._pending_var = tk.StringVar(value='')
        tk.Label(ctrl_frame, textvariable=self._pending_var,
                 fg='grey', anchor='w').pack(side=tk.LEFT, padx=12)

        # --- Removal regions panel ---
        regions_outer = tk.LabelFrame(self._root, text="Removal regions")
        regions_outer.pack(fill=tk.X, padx=8, pady=4)

        # Scrollable inner frame
        regions_canvas = tk.Canvas(regions_outer, height=100, bd=0,
                                   highlightthickness=0)
        scrollbar = tk.Scrollbar(regions_outer, orient='vertical',
                                 command=regions_canvas.yview)
        self._regions_inner = tk.Frame(regions_canvas)
        self._regions_inner.bind(
            '<Configure>',
            lambda e: regions_canvas.configure(
                scrollregion=regions_canvas.bbox('all')))
        regions_canvas.create_window((0, 0), window=self._regions_inner,
                                     anchor='nw')
        regions_canvas.configure(yscrollcommand=scrollbar.set)
        regions_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                            padx=4, pady=4)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._regions_canvas = regions_canvas

        self._refresh_regions_list()

        # --- Crop & Save ---
        bottom_frame = tk.Frame(self._root)
        bottom_frame.pack(fill=tk.X, padx=8, pady=(4, 8))
        tk.Button(bottom_frame, text="Crop & Save",
                  command=self._crop_and_save).pack(side=tk.RIGHT)

    # ------------------------------------------------------------------
    # Chunk navigation
    # ------------------------------------------------------------------

    def _prev_chunk(self):
        if self._chunk > 0:
            self._chunk -= 1
            self._draw_chunk()

    def _next_chunk(self):
        if self._chunk < self._n_chunks - 1:
            self._chunk += 1
            self._draw_chunk()

    def _draw_chunk(self):
        """Renders the current chunk: frequency vs time, plus region overlays."""
        lo = self._chunk * self._chunk_size
        hi = min(lo + self._chunk_size, self._n)

        t_slice = self._time[lo:hi] - self._t0
        f_slice = self._freq[lo:hi]

        ax = self._ax
        ax.cla()
        ax.plot(t_slice, f_slice, linewidth=0.5, color='steelblue')
        ax.set_xlabel('Time since start (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(
            f'Chunk {self._chunk + 1} / {self._n_chunks}   '
            f'(indices {lo:,} – {hi - 1:,})',
            fontsize=9)

        if len(t_slice) == 0:
            self._canvas.draw_idle()
            return

        chunk_t_lo = t_slice[0]
        chunk_t_hi = t_slice[-1]

        # Overlay committed removal regions that intersect this chunk
        for r_lo_idx, r_hi_idx in self._regions:
            r_t_lo = self._time[r_lo_idx] - self._t0
            r_t_hi = self._time[min(r_hi_idx, self._n - 1)] - self._t0
            if r_t_hi >= chunk_t_lo and r_t_lo <= chunk_t_hi:
                ax.axvspan(max(r_t_lo, chunk_t_lo),
                           min(r_t_hi, chunk_t_hi),
                           alpha=0.25, color='red')

        # Overlay pending lower boundary if it falls within this chunk
        if self._pending_lower_idx is not None:
            p_t = self._time[self._pending_lower_idx] - self._t0
            if chunk_t_lo <= p_t <= chunk_t_hi:
                ax.axvline(p_t, color='red', linestyle='--', linewidth=1.2)

        self._fig.tight_layout(pad=1.5)
        self._canvas.draw_idle()

        # Update navigation controls
        self._chunk_label.config(
            text=f'Chunk {self._chunk + 1} / {self._n_chunks}')
        self._prev_btn.config(
            state=tk.NORMAL if self._chunk > 0 else tk.DISABLED)
        self._next_btn.config(
            state=tk.NORMAL if self._chunk < self._n_chunks - 1 else tk.DISABLED)

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _set_state(self, state: int):
        """Updates button enable/disable states and pending label."""
        self._state = state
        if state == 0:
            self._lower_btn.config(state=tk.NORMAL)
            self._upper_btn.config(state=tk.DISABLED)
            self._pending_var.set('')
        elif state == 1:
            self._lower_btn.config(state=tk.DISABLED)
            self._upper_btn.config(state=tk.DISABLED)
            self._pending_var.set('Click on plot to set lower boundary…')
        elif state == 2:
            self._lower_btn.config(state=tk.DISABLED)
            self._upper_btn.config(state=tk.NORMAL)
            lo_t = self._time[self._pending_lower_idx]
            self._pending_var.set(
                f'Lower: {lo_t:.6g} s  (idx {self._pending_lower_idx:,})')
        elif state == 3:
            self._lower_btn.config(state=tk.DISABLED)
            self._upper_btn.config(state=tk.DISABLED)
            lo_t = self._time[self._pending_lower_idx]
            self._pending_var.set(
                f'Lower: {lo_t:.6g} s  (idx {self._pending_lower_idx:,})'
                f'  — Click on plot to set upper boundary…')

    def _activate_lower(self):
        self._set_state(1)

    def _activate_upper(self):
        self._set_state(3)

    # ------------------------------------------------------------------
    # Click handler
    # ------------------------------------------------------------------

    def _on_click(self, event):
        """
        Handles matplotlib canvas click events.

        Only active in states 1 and 3. Converts the clicked x (time) value to
        the nearest global array index using binary search. In state 1, stores
        the index as the pending lower boundary and transitions to state 2. In
        state 3, validates that the upper index is greater than the lower,
        appends the region, refreshes the display, and transitions to state 0.
        """
        if event.inaxes is None or event.xdata is None:
            return
        if self._state not in (1, 3):
            return

        idx = self._time_to_idx(event.xdata + self._t0)

        if self._state == 1:
            self._pending_lower_idx = idx
            self._set_state(2)
            self._draw_chunk()

        elif self._state == 3:
            if idx <= self._pending_lower_idx:
                lo_t = self._time[self._pending_lower_idx]
                self._pending_var.set(
                    f'Lower: {lo_t:.6g} s  (idx {self._pending_lower_idx:,})'
                    f'  — Upper must be after lower. Click again.')
                return
            self._regions.append([self._pending_lower_idx, idx])
            self._pending_lower_idx = None
            self._refresh_regions_list()
            self._set_state(0)
            self._draw_chunk()

    def _time_to_idx(self, t: float) -> int:
        """Returns the index of the array element with the closest time to t."""
        idx = int(np.searchsorted(self._time, t))
        idx = int(np.clip(idx, 0, self._n - 1))
        if idx > 0 and (abs(self._time[idx - 1] - t)
                        <= abs(self._time[idx] - t)):
            idx -= 1
        return idx

    # ------------------------------------------------------------------
    # Regions list
    # ------------------------------------------------------------------

    def _refresh_regions_list(self):
        """Rebuilds the removal-regions panel from self._regions."""
        for widget in self._regions_inner.winfo_children():
            widget.destroy()

        if not self._regions:
            tk.Label(self._regions_inner, text='(none)', fg='grey',
                     anchor='w').pack(fill=tk.X)
            return

        for i, (lo_idx, hi_idx) in enumerate(self._regions):
            row = tk.Frame(self._regions_inner)
            row.pack(fill=tk.X, pady=1)
            lo_t = self._time[lo_idx]
            hi_t = self._time[min(hi_idx, self._n - 1)]
            n_pts = hi_idx - lo_idx + 1
            label_text = (f"Region {i + 1}:  "
                          f"{lo_t:.6g} s – {hi_t:.6g} s  "
                          f"(idx {lo_idx:,} – {hi_idx:,},  {n_pts:,} pts)")
            tk.Label(row, text=label_text, anchor='w').pack(side=tk.LEFT)
            tk.Button(row, text="Delete",
                      command=lambda i=i: self._delete_region(i)
                      ).pack(side=tk.RIGHT, padx=4)

    def _delete_region(self, region_index: int):
        """Removes the region at region_index, refreshes the panel and plot."""
        del self._regions[region_index]
        self._refresh_regions_list()
        self._draw_chunk()

    # ------------------------------------------------------------------
    # Crop & Save
    # ------------------------------------------------------------------

    def _crop_and_save(self):
        """
        Builds a boolean keep-mask from all committed regions, writes the three
        cropped binary files, copies auxiliary files (.json, images), prints a
        summary, and closes the application.

        If no regions have been defined the user is asked to confirm before
        writing an identical copy of the data.

        The output directory is <input_dir>_cropped/ at the same level as the
        input directory. If it already exists the user is asked whether to
        overwrite.
        """
        if not self._regions:
            if not messagebox.askyesno(
                    "No regions defined",
                    "No removal regions have been defined.\n\n"
                    "Save an unmodified copy of the data?"):
                return

        out_dir = self._directory.parent / (self._directory.name + '_cropped')
        if out_dir.exists():
            if not messagebox.askyesno(
                    "Output exists",
                    f"Output directory already exists:\n{out_dir}\n\n"
                    "Overwrite?"):
                return
            shutil.rmtree(out_dir)
        out_dir.mkdir()

        # Build keep-mask (True = keep, False = remove)
        mask = np.ones(self._n, dtype=bool)
        for lo_idx, hi_idx in self._regions:
            mask[lo_idx:hi_idx + 1] = False

        n_removed = int((~mask).sum())
        n_kept = int(mask.sum())

        # Write cropped binary files
        self._freq[mask].tofile(out_dir / self._freq_path.name)
        self._vs[mask].tofile(out_dir / self._vs_path.name)
        self._time[mask].tofile(out_dir / self._time_path.name)
        for p in (self._freq_path, self._vs_path, self._time_path):
            print(f"Written: {out_dir / p.name}")

        # Copy auxiliary files
        _AUX_EXT = {'.json', '.txt', '.png', '.jpg', '.jpeg', '.tif', '.tiff'}
        for f in sorted(self._directory.iterdir()):
            if f.is_file() and f.suffix.lower() in _AUX_EXT:
                shutil.copy2(f, out_dir / f.name)
                print(f"Copied:  {f.name}")

        summary = (
            f"Cropping complete.\n\n"
            f"Original:  {self._n:,} points\n"
            f"Removed:   {n_removed:,} points "
            f"({100 * n_removed / self._n:.2f}%)\n"
            f"Kept:      {n_kept:,} points\n\n"
            f"Output: {out_dir}"
        )
        print(summary)
        messagebox.showinfo("Done", summary)
        self._root.destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    directory, chunk_size = parse_cli_args()
    freq_path, vs_path, time_path = _find_smr_files(directory)
    freq, vs, time = _load_data(freq_path, vs_path, time_path)

    root = tk.Tk()
    root.protocol('WM_DELETE_WINDOW', lambda: sys.exit(0))
    CropWindow(root, directory, freq, vs, time, chunk_size,
               freq_path, vs_path, time_path)
    root.mainloop()


if __name__ == '__main__':
    main()
