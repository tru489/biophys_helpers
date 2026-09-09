"""
annotate_sickle_cells.py

Interactive per-transit annotation tool for compile_experiment.py's compiled
output. Shows one transit (every crop frame belonging to one cell_id) at a
time and lets the user classify it as sickled / unsickled / indeterminate via
buttons or key presses (S / U / I), automatically advancing to the next
unreviewed transit. A Back button steps to the previous transit to revise its
classification; an End button marks every not-yet-reviewed transit
indeterminate and writes the final results.

Transits are reviewed in a shuffled order spanning every sample, not sample by
sample, so that ending the session early (as expected — curating every cell is
not the goal) still leaves a reasonably representative subset from every
sample rather than only the alphabetically-first ones. The shuffle is
generated once and persisted (<compiled_dir>/sickle_review_order.json), so
resuming a session continues along the same order instead of reshuffling.

Reuses browse_pt.py's zip/pickle .pt reader (_read_pt) and seek-based pixel
reader (CropReader), so paging through a multi-hundred-thousand-crop cache
stays instant without a torch install. Transit grouping (by cell_id, within
experiment/sample_name, ordered by frame_in_cell) is done locally rather than
via browse_pt.TransitIndex, since the xlsx summary needs the raw typed
grouping values, not just a formatted display label.

Input: an experiment_data.xlsx written by compile_experiment.py (or the
*_compiled/ dir containing it). Its sibling concat_vqvae_cache.pt and
concat_vqvae_cache_metadata.parquet are located automatically; the tool exits
with a clear error if a sample was compiled with --no-vqvae (no crop cache to
annotate).

Persistence:
    - Every classification is appended to a sidecar
      <compiled_dir>/sickle_annotations.jsonl (one JSON line per decision,
      keyed by experiment/sample_name/cell_id), so the session is crash-safe
      and resumable without forcing an early End: closing the window simply
      leaves the sidecar for the next launch to pick up.
    - On startup, any sickle_class column already baked into the metadata
      parquet by a previous End is read as a baseline, then the sidecar is
      replayed on top of it (sidecar wins on conflict) — so relaunching
      always resumes exactly where a previous session left off, whether it
      ended normally, was closed early, or had already finished once and is
      being revised.
    - "End" fills every remaining unreviewed transit as indeterminate, then
      merges the full annotation set into:
        * concat_vqvae_cache_metadata.parquet — new column sickle_class, one
          value per crop row, rewritten streaming batch-by-batch (the same
          approach concat_vqvae_caches.write_concat_parquet uses) so a huge
          cache stays cheap to update.
        * experiment_data.xlsx — a sickle_annotations sheet, one row per
          transit (experiment, sample_name, cell_id, n_frames, sickle_class),
          added/replaced without touching any other sheet.
      Only End performs this full-file rewrite; every other action just
      appends to the sidecar.

Usage:
    python annotate_sickle_cells.py <compiled_dir_or_experiment_data.xlsx>
"""
import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageTk

import browse_pt as bpt
from concat_vqvae_caches import _BATCH_ROWS

_ANNOTATION_COLUMN = 'sickle_class'
_SIDECAR_NAME       = 'sickle_annotations.jsonl'
_ORDER_NAME         = 'sickle_review_order.json'
_SHEET_NAME         = 'sickle_annotations'

_CLASS_KEYS = {'s': 'sickled', 'u': 'unsickled', 'i': 'indeterminate'}

_STRIP_HEIGHT = 80    # display height (px) per crop
_GRID_PAD     = 2     # px padding around each crop in the wrapped grid
_GRID_MARGIN  = 24    # px reserved (scrollbar etc.) when fitting columns to width


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> tuple[Path, Path, Path]:
    parser = argparse.ArgumentParser(
        description="Manually classify each transit in a compiled "
                    "experiment's crop cache as sickled / unsickled / "
                    "indeterminate.")
    parser.add_argument('target', type=str,
                        help="Path to a compile_experiment.py *_compiled/ "
                             "dir, or directly to its experiment_data.xlsx")
    args = parser.parse_args()

    p = Path(args.target)
    xlsx_path = (p / 'experiment_data.xlsx') if p.is_dir() else p
    if not xlsx_path.is_file():
        raise FileNotFoundError(f"experiment_data.xlsx not found: {xlsx_path}")

    compiled_dir = xlsx_path.parent
    pt_path = compiled_dir / 'concat_vqvae_cache.pt'
    parquet_path = compiled_dir / 'concat_vqvae_cache_metadata.parquet'
    if not pt_path.is_file() or not parquet_path.is_file():
        raise FileNotFoundError(
            f"{compiled_dir} has no concat_vqvae_cache.pt / "
            f"concat_vqvae_cache_metadata.parquet beside {xlsx_path.name} — "
            f"was this experiment compiled with --no-vqvae, or does it have "
            f"no samples with a crop cache?")
    return xlsx_path, pt_path, parquet_path


# ---------------------------------------------------------------------------
# Transit grouping
# ---------------------------------------------------------------------------

class Transit:
    __slots__ = ('experiment', 'sample_name', 'cell_id', 'rows')

    def __init__(self, experiment: str, sample_name: str, cell_id,
                rows: list[int]):
        self.experiment   = experiment
        self.sample_name  = sample_name
        self.cell_id      = cell_id
        self.rows         = rows   # tensor/parquet row indices, frame-ordered

    @property
    def key(self) -> tuple:
        return (self.experiment, self.sample_name, self.cell_id)


def _load_transits(parquet_path: Path, n_rows: int) -> tuple[list[Transit], dict]:
    """
    Group the concatenated cache's rows into transits — one per (experiment,
    sample_name, cell_id) — ordered by frame_in_cell within each transit and
    sorted by (sample_name, cell_id) overall. This is the canonical list
    order (stable across runs, and what the xlsx summary sheet is written
    in) — the shuffled *review* order used on screen is a separate sequence
    built in SickleAnnotator, see _load_or_build_order.

    Returns (transits, baseline) where baseline maps a transit's key to
    whatever sickle_class value a previous End already wrote for it.
    """
    pf = pq.ParquetFile(str(parquet_path))
    names = [f.name for f in pf.schema_arrow]
    required = ['cell_id', 'experiment', 'sample_name']
    missing = [c for c in required if c not in names]
    if missing:
        raise ValueError(
            f"{parquet_path.name} is missing {missing} — this doesn't look "
            f"like a compile_experiment.py concatenated crop-cache metadata "
            f"parquet")
    if pf.metadata.num_rows != n_rows:
        raise ValueError(
            f"{parquet_path.name} has {pf.metadata.num_rows:,} rows but the "
            f"crop cache has {n_rows:,} rows — they are out of sync")

    order_col = 'frame_in_cell' if 'frame_in_cell' in names else None
    has_existing = _ANNOTATION_COLUMN in names
    wanted = required + ([order_col] if order_col else []) \
        + ([_ANNOTATION_COLUMN] if has_existing else [])
    table = pf.read(columns=wanted)
    pf.close()

    cell_id      = table.column('cell_id').to_pylist()
    experiment   = table.column('experiment').to_pylist()
    sample_name  = table.column('sample_name').to_pylist()
    order        = table.column(order_col).to_pylist() if order_col else None
    existing     = (table.column(_ANNOTATION_COLUMN).to_pylist()
                    if has_existing else None)

    groups: dict[tuple, list[int]] = {}
    for row in range(n_rows):
        groups.setdefault(
            (experiment[row], sample_name[row], cell_id[row]), []).append(row)

    baseline: dict[tuple, str] = {}
    transits: list[Transit] = []
    for key, rows in groups.items():
        if order is not None:
            rows.sort(key=lambda r: (
                order[r] if order[r] is not None else float('inf'), r))
        transits.append(Transit(*key, rows))
        if existing is not None:
            val = existing[rows[0]]
            if val:
                baseline[key] = val

    transits.sort(key=lambda t: (t.sample_name, t.cell_id))
    return transits, baseline


# ---------------------------------------------------------------------------
# Sidecar (crash-safe, resumable autosave)
# ---------------------------------------------------------------------------

def _sidecar_path(compiled_dir: Path) -> Path:
    return compiled_dir / _SIDECAR_NAME


def _load_sidecar(path: Path) -> dict:
    """Replay the sidecar JSONL; the last line for a key wins."""
    out: dict[tuple, str] = {}
    if not path.is_file():
        return out
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = (rec['experiment'], rec['sample_name'], rec['cell_id'])
                out[key] = rec['sickle_class']
            except Exception:
                continue   # tolerate a truncated last line from a crash
    return out


def _append_sidecar(path: Path, transit: Transit, cls: str):
    with open(path, 'a', encoding='utf-8') as fh:
        fh.write(json.dumps({
            'experiment':   transit.experiment,
            'sample_name':  transit.sample_name,
            'cell_id':      transit.cell_id,
            'sickle_class': cls,
        }) + '\n')
        fh.flush()


# ---------------------------------------------------------------------------
# Review order (shuffled, persisted so a resumed session doesn't reshuffle)
# ---------------------------------------------------------------------------

def _order_path(compiled_dir: Path) -> Path:
    return compiled_dir / _ORDER_NAME


def _load_order(path: Path) -> list[tuple] | None:
    """The persisted review order as a list of transit keys, or None."""
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
        return [tuple(entry) for entry in raw]
    except Exception:
        return None


def _save_order(path: Path, keys: list[tuple]):
    path.write_text(json.dumps([list(k) for k in keys]), encoding='utf-8')


def _load_or_build_order(path: Path, transits: list[Transit]) -> list[int]:
    """
    Indices into `transits` giving the on-screen review order: shuffled across
    every sample (rather than sample by sample) so that ending the session
    early still leaves a reasonably representative subset from each sample,
    not just the alphabetically-first ones.

    The shuffle is generated once and written to `path`; a later call (a
    resumed session) reads it back so Back/Next keep walking the same
    sequence instead of jumping to a freshly-shuffled one. Persisted keys no
    longer present in `transits` are dropped; transits with no persisted
    entry (e.g. the cache was regenerated) are shuffled in and appended.
    """
    by_key = {t.key: i for i, t in enumerate(transits)}

    persisted = _load_order(path) or []
    order_keys = [k for k in persisted if k in by_key]

    known = set(order_keys)
    new_keys = [t.key for t in transits if t.key not in known]
    if new_keys:
        random.shuffle(new_keys)
        order_keys += new_keys

    if order_keys != persisted:
        _save_order(path, order_keys)

    return [by_key[k] for k in order_keys]


# ---------------------------------------------------------------------------
# Final merge into the parquet + xlsx
# ---------------------------------------------------------------------------

def _merge_outputs(parquet_path: Path, xlsx_path: Path,
                   transits: list[Transit], classes: dict, log=print):
    n_rows = sum(len(t.rows) for t in transits)
    col: list = [None] * n_rows
    summary_rows = []
    for t in transits:
        cls = classes.get(t.key)
        for row in t.rows:
            col[row] = cls
        summary_rows.append({
            'experiment':   t.experiment,
            'sample_name':  t.sample_name,
            'cell_id':      t.cell_id,
            'n_frames':     len(t.rows),
            'sickle_class': cls,
        })

    # --- rewrite the metadata parquet, streaming batch-by-batch ---
    pf = pq.ParquetFile(str(parquet_path))
    fields = [f for f in pf.schema_arrow if f.name != _ANNOTATION_COLUMN]
    fields.append(pa.field(_ANNOTATION_COLUMN, pa.string()))
    out_schema = pa.schema(fields)

    tmp_path = parquet_path.with_name(parquet_path.name + '.tmp')
    writer = pq.ParquetWriter(str(tmp_path), out_schema)
    written = 0
    try:
        for batch in pf.iter_batches(batch_size=_BATCH_ROWS):
            n = batch.num_rows
            chunk = col[written:written + n]
            arrays = [
                pa.array(chunk, type=pa.string()) if field.name == _ANNOTATION_COLUMN
                else batch.column(field.name)
                for field in out_schema
            ]
            writer.write_table(pa.Table.from_arrays(arrays, schema=out_schema))
            written += n
    finally:
        writer.close()
        pf.close()
    tmp_path.replace(parquet_path)
    log(f'Wrote {_ANNOTATION_COLUMN!r} for {n_rows:,} row(s) -> '
        f'{parquet_path.name}')

    # --- add/replace the summary sheet in the xlsx, other sheets untouched ---
    summary_df = pd.DataFrame(summary_rows, columns=[
        'experiment', 'sample_name', 'cell_id', 'n_frames', 'sickle_class'])
    with pd.ExcelWriter(str(xlsx_path), engine='openpyxl', mode='a',
                        if_sheet_exists='replace') as xw:
        summary_df.to_excel(xw, sheet_name=_SHEET_NAME, index=False)
    log(f'Wrote {len(summary_df):,} transit(s) -> {xlsx_path.name} '
        f'[{_SHEET_NAME}]')


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class SickleAnnotator:

    def __init__(self, root: tk.Tk, pt_path: Path, parquet_path: Path,
                xlsx_path: Path):
        self._root = root
        self._pt_path = pt_path
        self._parquet_path = parquet_path
        self._xlsx_path = xlsx_path
        self._sidecar_path = _sidecar_path(pt_path.parent)
        self._order_path = _order_path(pt_path.parent)

        obj, _info = bpt._read_pt(pt_path)
        viewable = dict(bpt.viewable_tensors(obj))
        if 'bf_u8' not in viewable:
            raise ValueError(f"{pt_path.name} has no viewable bf_u8 tensor")
        self._bf_reader = bpt.CropReader(pt_path, viewable['bf_u8'])
        self._fl_reader = (bpt.CropReader(pt_path, viewable['fl_u8'])
                           if 'fl_u8' in viewable else None)

        self._transits, baseline = _load_transits(parquet_path, self._bf_reader.n)
        if not self._transits:
            raise ValueError(f"{parquet_path.name} has no rows to annotate")
        self._classes: dict[tuple, str] = dict(baseline)
        self._classes.update(_load_sidecar(self._sidecar_path))

        self._order = _load_or_build_order(self._order_path, self._transits)

        self._cursor = 0
        while (self._cursor < len(self._order)
               and self._transits[self._order[self._cursor]].key in self._classes):
            self._cursor += 1

        self._photo_refs: list = []
        root.title('Annotate sickle cells')
        root.geometry('1000x750')
        self._build_ui()
        root.update()   # so the canvas has its real, laid-out width to wrap to
        self._show_current()
        root.protocol('WM_DELETE_WINDOW', self._on_close)
        root.bind('<Key>', self._on_key)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        top = tk.Frame(self._root)
        top.pack(fill=tk.X, padx=10, pady=(10, 2))
        self._progress_var = tk.StringVar()
        tk.Label(top, textvariable=self._progress_var,
                font=('TkDefaultFont', 11, 'bold')).pack(side=tk.LEFT)

        self._info_var = tk.StringVar()
        tk.Label(self._root, textvariable=self._info_var,
                font=('TkDefaultFont', 10)).pack(fill=tk.X, padx=10, pady=(0, 4))

        body = tk.Frame(self._root, bg='#f0f0f0')
        body.pack(fill=tk.BOTH, expand=True, padx=10)
        canvas = tk.Canvas(body, bg='#f0f0f0', highlightthickness=0)
        vsb = tk.Scrollbar(body, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = tk.Frame(canvas, bg='#f0f0f0')
        win = canvas.create_window((0, 0), window=inner, anchor='nw')
        inner.bind('<Configure>',
                   lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        # Pin the inner frame's width to the canvas's, not its height: crops
        # wrap onto new rows to fill the width, growing the content downward,
        # so it's the height that must stay free to expand past the canvas.
        canvas.bind('<Configure>',
                    lambda e: canvas.itemconfig(win, width=e.width))
        for w in (canvas, inner):
            w.bind('<MouseWheel>',
                  lambda e, c=canvas: c.yview_scroll(-1 * (e.delta // 120), 'units'))
        self._canvas = canvas
        self._strip = inner

        btn_frame = tk.Frame(self._root)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        self._back_btn = tk.Button(btn_frame, text='← Back (⌫)',
                                   width=14, command=self._go_back)
        self._back_btn.pack(side=tk.LEFT)

        class_frame = tk.Frame(btn_frame)
        class_frame.pack(side=tk.LEFT, expand=True)
        tk.Button(class_frame, text='Sickled (S)', width=16, height=2,
                 command=lambda: self._classify('sickled')).pack(
            side=tk.LEFT, padx=6)
        tk.Button(class_frame, text='Unsickled (U)', width=16, height=2,
                 command=lambda: self._classify('unsickled')).pack(
            side=tk.LEFT, padx=6)
        tk.Button(class_frame, text='Indeterminate (I)', width=16, height=2,
                 command=lambda: self._classify('indeterminate')).pack(
            side=tk.LEFT, padx=6)

        self._end_btn = tk.Button(btn_frame, text='End', width=10, fg='#900',
                                  command=self._end)
        self._end_btn.pack(side=tk.RIGHT)

        tk.Label(self._root, fg='#666', font=('TkDefaultFont', 9),
                 justify=tk.LEFT, wraplength=900, anchor='w', text=(
                     'Progress is saved automatically after every decision — '
                     'closing this window is safe, relaunch to resume. '
                     'End marks every unreviewed transit indeterminate and '
                     'writes the final sickle_class column into the metadata '
                     'parquet and a summary sheet into the xlsx.')
                 ).pack(fill=tk.X, padx=10, pady=(0, 8))

    # ------------------------------------------------------------------
    # Key handling
    # ------------------------------------------------------------------

    def _on_key(self, event):
        ch = event.char.lower()
        if ch in _CLASS_KEYS:
            self._classify(_CLASS_KEYS[ch])
        elif event.keysym in ('BackSpace', 'Left'):
            self._go_back()

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _current_transit(self) -> Transit | None:
        if 0 <= self._cursor < len(self._order):
            return self._transits[self._order[self._cursor]]
        return None

    def _show_current(self):
        for child in self._strip.winfo_children():
            child.destroy()
        self._photo_refs.clear()

        n = len(self._order)
        done = sum(1 for t in self._transits if t.key in self._classes)
        self._progress_var.set(
            f'{min(self._cursor, n):,} / {n:,} reviewed   '
            f'({done:,} classified total)')
        self._back_btn.config(state=tk.NORMAL if self._cursor > 0 else tk.DISABLED)

        t = self._current_transit()
        if t is None:
            self._info_var.set('All transits reviewed — press End to finish.')
            tk.Label(self._strip, text='(nothing left to review)',
                    bg='#f0f0f0', fg='#666').pack(padx=20, pady=40)
            return

        cur_cls = self._classes.get(t.key)
        tag = f'   [currently: {cur_cls}]' if cur_cls else ''
        self._info_var.set(
            f'{t.sample_name}   cell_id={t.cell_id}   '
            f'{len(t.rows)} frame(s){tag}')

        self._draw_strip(t, self._bf_reader, 'BF')
        if self._fl_reader is not None:
            self._draw_strip(t, self._fl_reader, 'FL')

    def _draw_strip(self, t: Transit, reader: 'bpt.CropReader', label: str):
        """One channel's crops, wrapped onto as many rows as the window
        needs so every frame of the transit is visible without scrolling
        sideways (only vertical scrolling, if the transit is very long)."""
        block = tk.Frame(self._strip, bg='#f0f0f0')
        block.pack(anchor='w', fill=tk.X, pady=(4, 8))
        tk.Label(block, text=label, bg='#f0f0f0',
                font=('TkDefaultFont', 9, 'bold')).pack(anchor='w')

        h, w = reader.frame_shape
        dh = _STRIP_HEIGHT
        dw = max(1, round(w / h * dh)) if h else dh
        cell_w = dw + 2 * _GRID_PAD
        avail = max(cell_w, self._canvas.winfo_width() - _GRID_MARGIN)
        cols = max(1, avail // cell_w)

        grid = tk.Frame(block, bg='#f0f0f0')
        grid.pack(anchor='w')
        for i, row in enumerate(t.rows):
            r, c = divmod(i, cols)
            cell = tk.Frame(grid, bg='#f0f0f0')
            cell.grid(row=r, column=c, padx=_GRID_PAD, pady=_GRID_PAD)
            try:
                frame = reader.read(row)
                photo = ImageTk.PhotoImage(
                    Image.fromarray(frame, mode='L').resize(
                        (dw, dh), Image.NEAREST))
                self._photo_refs.append(photo)
                tk.Label(cell, image=photo, bg='#f0f0f0',
                        relief=tk.SOLID, bd=1).pack()
            except Exception:
                tk.Label(cell, text='(err)', fg='#b00', bg='#f0f0f0',
                        width=6).pack()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _classify(self, cls: str):
        t = self._current_transit()
        if t is None:
            return
        self._classes[t.key] = cls
        _append_sidecar(self._sidecar_path, t, cls)
        self._cursor += 1
        self._show_current()

    def _go_back(self):
        if self._cursor > 0:
            self._cursor -= 1
            self._show_current()

    def _end(self):
        for t in self._transits:
            if t.key not in self._classes:
                self._classes[t.key] = 'indeterminate'
                _append_sidecar(self._sidecar_path, t, 'indeterminate')

        try:
            _merge_outputs(self._parquet_path, self._xlsx_path,
                          self._transits, self._classes,
                          log=lambda m: print(f'  {m}'))
        except Exception as exc:
            messagebox.showerror('Error writing results', str(exc),
                                 parent=self._root)
            return

        messagebox.showinfo(
            'Done',
            f'{len(self._transits):,} transit(s) written to:\n'
            f'  {self._parquet_path.name}  (column "{_ANNOTATION_COLUMN}")\n'
            f'  {self._xlsx_path.name}  (sheet "{_SHEET_NAME}")',
            parent=self._root)
        self._close_readers()
        self._root.destroy()

    def _on_close(self):
        self._close_readers()
        self._root.destroy()

    def _close_readers(self):
        self._bf_reader.close()
        if self._fl_reader is not None:
            self._fl_reader.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    xlsx_path, pt_path, parquet_path = _parse_args()
    print(f'Loading {pt_path.name} / {parquet_path.name}...')

    root = tk.Tk()
    try:
        SickleAnnotator(root, pt_path, parquet_path, xlsx_path)
    except Exception as exc:
        root.withdraw()
        messagebox.showerror('Cannot open', str(exc))
        sys.exit(1)
    root.mainloop()


if __name__ == '__main__':
    main()
