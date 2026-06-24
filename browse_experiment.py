"""
browse_experiment.py

Interactive GUI for browsing compiled experiment data produced by
compile_experiment.py. Displays three per-sample boxplots (volume, buoyant
mass, density) with clickable scatter overlays; clicking any data point loads
that transit's BF image frames into a scrollable panel at the bottom.

All three metrics are sourced from the volume DataFrame so every scatter point
has a transit_index and can show images. Buoyant mass uses `matched_mass` (only
cells paired with an imaging transit), and density uses `buoyant_density`.

Bottom area holds up to 3 transit panels stacked vertically. Selecting a 4th
transit evicts the oldest (FIFO). Each panel is individually closeable.

Usage:
    python browse_experiment.py <compiled_dir>

    <compiled_dir>  Path to a *_compiled/ directory containing
                    experiment_data.h5 and (optionally) images.h5
"""
import argparse
import collections
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox
import warnings

warnings.filterwarnings('ignore', message='object name is not a valid Python identifier')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> Path:
    parser = argparse.ArgumentParser(
        description='Browse compiled experiment data interactively.')
    parser.add_argument('compiled_dir', type=str,
                        help='Path to a *_compiled/ directory')
    args = parser.parse_args()
    p = Path(args.compiled_dir)
    if not p.is_dir():
        raise FileNotFoundError(f'Directory not found: {p}')
    h5 = p / 'experiment_data.h5'
    if not h5.is_file():
        raise FileNotFoundError(f'experiment_data.h5 not found in {p}')
    return p


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_samples(compiled_dir: Path) -> list:
    """
    Load per-sample arrays for each metric from experiment_data.h5.

    Returns a list of dicts, one per sample that has volume data:
        name, key, has_images,
        volume:  (values, transit_indices),
        mass:    (values, transit_indices),   # matched_mass rows only
        density: (values, transit_indices),   # buoyant_density rows only
    """
    h5_path = compiled_dir / 'experiment_data.h5'
    samples = []

    with pd.HDFStore(str(h5_path), mode='r') as store:
        meta = store['/metadata']

        for _, row in meta.iterrows():
            if not row['has_volume']:
                continue
            key = row['hdf5_key']
            vol_df = store[f'/samples/{key}/volume']

            def _extract(df, val_col, filter_col=None):
                subset = df.dropna(subset=[val_col])
                if filter_col:
                    subset = subset.dropna(subset=[filter_col])
                tidx = subset['transit_index'].values.astype(int)
                vals = subset[val_col].values.astype(float)
                return vals, tidx

            vol_vals,  vol_tidx  = _extract(vol_df, 'volume')
            mass_vals, mass_tidx = _extract(vol_df, 'matched_mass')
            dens_vals, dens_tidx = _extract(vol_df, 'buoyant_density')

            samples.append({
                'name':       row['sample_name'],
                'key':        key,
                'has_images': bool(row['has_images']),
                'volume':     (vol_vals,  vol_tidx),
                'mass':       (mass_vals, mass_tidx),
                'density':    (dens_vals, dens_tidx),
            })

    return samples


# ---------------------------------------------------------------------------
# Transit panel widget
# ---------------------------------------------------------------------------

class TransitPanel(tk.Frame):
    """
    A single-transit display panel: title bar + horizontal strip of BF frames.
    """

    _BF_HEIGHT = 120   # display height in pixels for each BF frame

    def __init__(self, parent, sample_name: str, transit_idx: int,
                 bf_stack: np.ndarray, on_close):
        super().__init__(parent, relief=tk.GROOVE, borderwidth=2)
        self._photo_refs = []   # keep PhotoImage objects alive

        # Title bar
        title_bar = tk.Frame(self)
        title_bar.pack(fill=tk.X, padx=4, pady=(4, 2))
        tk.Label(
            title_bar,
            text=f'{sample_name}  —  Transit {transit_idx:05d}',
            font=('TkDefaultFont', 9, 'bold'), anchor='w',
        ).pack(side=tk.LEFT)
        tk.Button(
            title_bar, text='✕', font=('TkDefaultFont', 9),
            relief=tk.FLAT, padx=4,
            command=lambda: on_close(self),
        ).pack(side=tk.RIGHT)

        # BF frame strip (horizontally scrollable)
        strip_frame = tk.Frame(self)
        strip_frame.pack(fill=tk.X, padx=4, pady=(0, 4))

        h_canvas = tk.Canvas(strip_frame, height=self._BF_HEIGHT + 20)
        h_scroll = tk.Scrollbar(strip_frame, orient=tk.HORIZONTAL,
                                command=h_canvas.xview)
        h_canvas.configure(xscrollcommand=h_scroll.set)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        h_canvas.pack(side=tk.TOP, fill=tk.X)

        inner = tk.Frame(h_canvas)
        h_canvas.create_window((0, 0), window=inner, anchor='nw')
        inner.bind('<Configure>', lambda e: h_canvas.configure(
            scrollregion=h_canvas.bbox('all')))

        tk.Label(inner, text='BF', font=('TkDefaultFont', 8),
                 fg='grey').pack(side=tk.LEFT, padx=(0, 4))

        n_frames = bf_stack.shape[0]
        h_src    = bf_stack.shape[1]
        w_src    = bf_stack.shape[2]
        dh       = self._BF_HEIGHT
        dw       = max(1, round(w_src / h_src * dh))

        for i in range(n_frames):
            frame = bf_stack[i]
            img = Image.fromarray(frame, mode='L').resize(
                (dw, dh), Image.NEAREST)
            photo = ImageTk.PhotoImage(img)
            self._photo_refs.append(photo)
            tk.Label(inner, image=photo, borderwidth=1,
                     relief=tk.SOLID).pack(side=tk.LEFT, padx=1)

        # Mousewheel on strip → horizontal scroll
        for w in (h_canvas, inner):
            w.bind('<MouseWheel>',
                   lambda e, c=h_canvas: c.xview_scroll(
                       -1 * (e.delta // 120), 'units'))


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class BrowseApp:

    def __init__(self, root: tk.Tk, compiled_dir: Path, samples: list):
        self._root        = root
        self._compiled_dir = compiled_dir
        self._samples     = samples
        self._panels: collections.deque = collections.deque(maxlen=3)

        root.title(f'browse_experiment — {compiled_dir.name}')
        root.geometry('1200x800')

        self._build_plots()
        self._build_bottom()

    # ------------------------------------------------------------------
    # Top section: three boxplots
    # ------------------------------------------------------------------

    def _build_plots(self):
        plot_frame = tk.Frame(self._root)
        plot_frame.pack(fill=tk.X, padx=0, pady=0)

        fig, self._axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.subplots_adjust(bottom=0.3, wspace=0.35)

        self._draw_boxplots()

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.X)
        canvas.mpl_connect('pick_event', self._on_pick)
        self._fig_canvas = canvas

    def _draw_boxplots(self):
        specs = [
            ('volume',  'volume',        'Volume (fL)'),
            ('mass',    'matched_mass',   'Buoyant Mass (pg)'),
            ('density', 'buoyant_density','Buoyant Density (g/mL)'),
        ]
        rng = np.random.default_rng(0)

        for ax, (metric_key, col_label, title) in zip(self._axes, specs):
            ax.clear()
            ax.set_title(title, fontsize=9)

            positions = []
            box_data  = []
            names     = []

            for pos, sample in enumerate(self._samples):
                vals, tidx = sample[metric_key]
                if len(vals) == 0:
                    continue
                positions.append(pos)
                box_data.append(vals)
                names.append(sample['name'])

                # Scatter jitter
                jitter = rng.uniform(-0.25, 0.25, size=len(vals))
                sc = ax.scatter(pos + jitter, vals, s=5, alpha=0.4,
                                color='steelblue', picker=5, zorder=3)
                sc.sample_name     = sample['name']
                sc.hdf5_key        = sample['key']
                sc.has_images      = sample['has_images']
                sc.transit_indices = tidx

            if box_data:
                ax.boxplot(box_data, positions=positions,
                           widths=0.5, showfliers=False,
                           medianprops={'color': 'red', 'linewidth': 1.5},
                           boxprops={'linewidth': 0.8},
                           whiskerprops={'linewidth': 0.8},
                           capprops={'linewidth': 0.8})
                ax.set_xticks(positions)
                ax.set_xticklabels(names, rotation=45, ha='right', fontsize=6)
            ax.set_xlim(-0.7, len(self._samples) - 0.3)

        self._fig_canvas.draw() if hasattr(self, '_fig_canvas') else None

    # ------------------------------------------------------------------
    # Bottom section: scrollable transit panels
    # ------------------------------------------------------------------

    def _build_bottom(self):
        bottom = tk.Frame(self._root)
        bottom.pack(fill=tk.BOTH, expand=True)

        self._v_canvas = tk.Canvas(bottom, bg='#f0f0f0')
        vsb = tk.Scrollbar(bottom, orient=tk.VERTICAL,
                           command=self._v_canvas.yview)
        self._v_canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._v_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._inner = tk.Frame(self._v_canvas, bg='#f0f0f0')
        self._canvas_win = self._v_canvas.create_window(
            (0, 0), window=self._inner, anchor='nw')

        self._inner.bind('<Configure>', lambda e: self._v_canvas.configure(
            scrollregion=self._v_canvas.bbox('all')))
        self._v_canvas.bind('<Configure>', lambda e: self._v_canvas.itemconfig(
            self._canvas_win, width=e.width))

        # Mousewheel → vertical scroll
        for w in (self._v_canvas, self._inner):
            w.bind('<MouseWheel>',
                   lambda e: self._v_canvas.yview_scroll(
                       -1 * (e.delta // 120), 'units'))

        # Placeholder label when no panels are open
        self._placeholder = tk.Label(
            self._inner, text='Click a data point above to view transit images.',
            fg='grey', bg='#f0f0f0', font=('TkDefaultFont', 10))
        self._placeholder.pack(pady=20)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_pick(self, event):
        sc = event.artist
        if not hasattr(sc, 'transit_indices'):
            return
        i    = event.ind[0]
        tidx = int(sc.transit_indices[i])
        self._open_transit(sc.sample_name, sc.hdf5_key, tidx, sc.has_images)

    def _open_transit(self, sample_name: str, hdf5_key: str,
                      transit_idx: int, has_images: bool):
        if not has_images:
            messagebox.showinfo(
                'No images',
                f'{sample_name}: images were not compiled for this sample.')
            return

        images_h5 = self._compiled_dir / 'images.h5'
        if not images_h5.is_file():
            messagebox.showwarning(
                'Missing file', 'images.h5 not found in the compiled directory.')
            return

        tkey = f'{hdf5_key}/{transit_idx:05d}/bf'
        try:
            with h5py.File(images_h5, 'r') as f:
                if tkey not in f:
                    messagebox.showwarning(
                        'Not found',
                        f'{sample_name}: transit {transit_idx:05d} not in images.h5.')
                    return
                bf = f[tkey][:]
        except Exception as exc:
            messagebox.showerror('Error', f'Could not load images: {exc}')
            return

        # Evict oldest if at capacity
        if len(self._panels) == self._panels.maxlen:
            old = self._panels.popleft()
            old.destroy()

        # Hide placeholder
        self._placeholder.pack_forget()

        panel = TransitPanel(
            self._inner, sample_name, transit_idx, bf,
            on_close=self._close_panel)
        panel.pack(fill=tk.X, padx=6, pady=(0, 6))
        self._panels.append(panel)

        # Scroll to bottom so newest panel is visible
        self._root.update_idletasks()
        self._v_canvas.yview_moveto(1.0)

    def _close_panel(self, panel: TransitPanel):
        if panel in self._panels:
            self._panels.remove(panel)
        panel.destroy()
        if not self._panels:
            self._placeholder.pack(pady=20)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    compiled_dir = _parse_args()
    print(f'Loading {compiled_dir.name}...')
    samples = _load_samples(compiled_dir)
    if not samples:
        print('No samples with volume data found.')
        sys.exit(1)
    print(f'Loaded {len(samples)} sample(s).')

    root = tk.Tk()
    BrowseApp(root, compiled_dir, samples)
    root.mainloop()


if __name__ == '__main__':
    main()
