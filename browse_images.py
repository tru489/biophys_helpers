"""
browse_images.py

Paginated GUI viewer for images.h5 files produced by compile_experiment.py.
Displays BF transit images 5 rows (transits) per page. Each row shows all
frames for that transit in a horizontally scrollable strip. Use Next / Prev
to page through transits.

If the file contains more than one sample, a dropdown at the top lets you
switch between them; each sample has its own independent page position.

Usage:
    python browse_images.py <images_h5>

    <images_h5>  Path to an images.h5 file (from compile_experiment.py)
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk


_ROWS_PER_PAGE = 5
_FRAME_HEIGHT  = 80   # display height (px) for each BF frame — width is scaled to aspect


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> Path:
    parser = argparse.ArgumentParser(
        description='Paginated viewer for images.h5 files from compile_experiment.py.')
    parser.add_argument('images_h5', type=str,
                        help='Path to images.h5')
    args = parser.parse_args()
    p = Path(args.images_h5)
    if not p.is_file():
        raise FileNotFoundError(f'File not found: {p}')
    return p


# ---------------------------------------------------------------------------
# Index loading
# ---------------------------------------------------------------------------

def _load_index(h5_path: Path) -> dict:
    """
    Read the group structure of images.h5 and return:
        {sample_name: [transit_key, ...]}   — keys sorted numerically
    Only samples with at least one transit are included.
    """
    index = {}
    with h5py.File(str(h5_path), 'r') as f:
        for sample in sorted(f.keys()):
            transits = sorted(f[sample].keys())
            if transits:
                index[sample] = transits
    return index


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class BrowseImagesApp:

    _ROWS         = _ROWS_PER_PAGE
    _FH           = _FRAME_HEIGHT
    _LABEL_FONT   = ('TkDefaultFont', 11)
    _NAV_FONT     = ('TkDefaultFont', 12)
    _NAV_FONT_B   = ('TkDefaultFont', 12, 'bold')
    _BG           = '#f0f0f0'

    def __init__(self, root: tk.Tk, h5_path: Path, index: dict):
        self._root    = root
        self._h5_path = h5_path
        self._index   = index
        self._samples = list(index.keys())

        # Per-sample page memory so switching samples and back preserves position
        self._pages: dict[str, int] = {s: 0 for s in self._samples}
        self._sample  = self._samples[0]

        # Keep PhotoImage objects alive between redraws
        self._photo_refs: list = []

        root.title(f'browse_images — {h5_path.name}')
        root.state('zoomed')

        self._build_ui()
        self._show_page()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ---- Top bar: sample selector + transit count ----
        top = tk.Frame(self._root)
        top.pack(fill=tk.X, padx=12, pady=(10, 4))

        if len(self._samples) > 1:
            tk.Label(top, text='Sample:', font=self._NAV_FONT).pack(side=tk.LEFT)
            self._sample_var = tk.StringVar(value=self._sample)
            cb = ttk.Combobox(top, values=self._samples,
                              textvariable=self._sample_var,
                              state='readonly', width=32,
                              font=self._NAV_FONT)
            cb.pack(side=tk.LEFT, padx=(6, 20))
            cb.bind('<<ComboboxSelected>>', self._on_sample_change)
        else:
            self._sample_var = None

        self._info_label = tk.Label(top, text='', font=self._NAV_FONT)
        self._info_label.pack(side=tk.LEFT)

        # ---- Middle: _ROWS row slots ----
        mid = tk.Frame(self._root, bg=self._BG)
        mid.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

        self._row_widgets: list[dict] = []
        for _ in range(self._ROWS):
            self._row_widgets.append(self._make_row(mid))

        # ---- Bottom: navigation ----
        bot = tk.Frame(self._root)
        bot.pack(fill=tk.X, padx=12, pady=(4, 10))

        self._prev_btn = tk.Button(bot, text='← Prev', font=self._NAV_FONT_B,
                                   width=10, command=self._prev)
        self._prev_btn.pack(side=tk.LEFT)

        self._nav_label = tk.Label(bot, text='', font=self._NAV_FONT)
        self._nav_label.pack(side=tk.LEFT, padx=24)

        self._next_btn = tk.Button(bot, text='Next →', font=self._NAV_FONT_B,
                                   width=10, command=self._next)
        self._next_btn.pack(side=tk.LEFT)

    def _make_row(self, parent: tk.Frame) -> dict:
        """Build one transit row and return its widget references."""
        bg = self._BG
        row = tk.Frame(parent, bg=bg, relief=tk.GROOVE, bd=1)
        row.pack(fill=tk.X, pady=3)

        # Transit ID label (left column, fixed width)
        lbl = tk.Label(row, text='', width=8, anchor='w',
                       font=self._LABEL_FONT, bg=bg)
        lbl.pack(side=tk.LEFT, padx=(6, 2))

        # Horizontal scrollable canvas for frames
        h_canvas = tk.Canvas(row, height=self._FH + 18, bg=bg,
                             highlightthickness=0)
        h_scroll = tk.Scrollbar(row, orient=tk.HORIZONTAL,
                                command=h_canvas.xview)
        h_canvas.configure(xscrollcommand=h_scroll.set)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        h_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True)

        inner = tk.Frame(h_canvas, bg=bg)
        win_id = h_canvas.create_window((0, 0), window=inner, anchor='nw')

        inner.bind('<Configure>',
                   lambda e, c=h_canvas: c.configure(scrollregion=c.bbox('all')))
        h_canvas.bind('<Configure>',
                      lambda e, c=h_canvas, w=win_id: c.itemconfig(w, height=e.height))

        for w in (h_canvas, inner):
            w.bind('<MouseWheel>',
                   lambda e, c=h_canvas: c.xview_scroll(
                       -1 * (e.delta // 120), 'units'))

        return {'label': lbl, 'canvas': h_canvas, 'inner': inner}

    # ------------------------------------------------------------------
    # Page rendering
    # ------------------------------------------------------------------

    @property
    def _page(self) -> int:
        return self._pages[self._sample]

    @_page.setter
    def _page(self, value: int):
        self._pages[self._sample] = value

    def _transit_list(self) -> list:
        return self._index.get(self._sample, [])

    def _max_page(self) -> int:
        n = len(self._transit_list())
        return max(0, (n - 1) // self._ROWS)

    def _show_page(self):
        self._photo_refs.clear()

        transits  = self._transit_list()
        n_total   = len(transits)
        start     = self._page * self._ROWS
        end       = min(start + self._ROWS, n_total)
        page_keys = transits[start:end]

        # Update header and nav
        self._info_label.config(
            text=f'{self._sample}   —   {n_total} transit(s)')
        self._nav_label.config(
            text=f'Page {self._page + 1} / {self._max_page() + 1}'
                 f'   (transits {start + 1}–{end})')
        self._prev_btn.config(
            state=tk.NORMAL if self._page > 0 else tk.DISABLED)
        self._next_btn.config(
            state=tk.NORMAL if self._page < self._max_page() else tk.DISABLED)

        # Render each row
        with h5py.File(str(self._h5_path), 'r') as f:
            for i, rw in enumerate(self._row_widgets):
                # Clear previous contents
                for child in rw['inner'].winfo_children():
                    child.destroy()

                if i >= len(page_keys):
                    rw['label'].config(text='')
                    continue

                t_key = page_keys[i]
                rw['label'].config(text=t_key)

                bf_key = f'{self._sample}/{t_key}/bf'
                if bf_key not in f:
                    tk.Label(rw['inner'], text='(no bf data)',
                             fg='grey', bg=self._BG,
                             font=self._LABEL_FONT).pack(
                        side=tk.LEFT, padx=8, pady=4)
                    continue

                bf_stack = f[bf_key][()]          # (n_frames, H, W) uint8
                n_frames, h_src, w_src = bf_stack.shape
                dh = self._FH
                dw = max(1, round(w_src / h_src * dh))

                for fi in range(n_frames):
                    img = Image.fromarray(bf_stack[fi], mode='L').resize(
                        (dw, dh), Image.NEAREST)
                    photo = ImageTk.PhotoImage(img)
                    self._photo_refs.append(photo)
                    tk.Label(rw['inner'], image=photo, bg=self._BG,
                             relief=tk.SOLID, bd=1).pack(
                        side=tk.LEFT, padx=2, pady=3)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _prev(self):
        if self._page > 0:
            self._page -= 1
            self._show_page()

    def _next(self):
        if self._page < self._max_page():
            self._page += 1
            self._show_page()

    def _on_sample_change(self, _event=None):
        self._sample = self._sample_var.get()
        self._show_page()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    h5_path = _parse_args()
    print(f'Loading index from {h5_path.name}...')
    index = _load_index(h5_path)

    if not index:
        print('No transit data found in file.')
        sys.exit(1)

    n_total = sum(len(v) for v in index.values())
    print(f'{len(index)} sample(s), {n_total} transit(s) total.')

    root = tk.Tk()
    BrowseImagesApp(root, h5_path, index)
    root.mainloop()


if __name__ == '__main__':
    main()
