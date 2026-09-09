"""
make_sickle_montage.py

Builds a montage figure of BF crops from randomly chosen sickled/unsickled
transits, from a compile_experiment.py output previously annotated by
annotate_sickle_cells.py — specifically the sickle_class column that tool's
End action writes into the crop cache's metadata parquet. Transits with no
sickle_class (unreviewed) or class 'indeterminate' are excluded throughout.

Three modes:
    grid (default)  One representative crop (the middle frame) per transit,
                     tiled into a grid — a morphology-at-a-glance comparison —
                     with the sample stacked "SICKLED" section above
                     "UNSICKLED" section.
    filmstrip        Every frame of each chosen transit, one transit per
                     (wrapped) block, several sickled and unsickled transits
                     stacked vertically, sectioned the same way as grid mode.
    mixed            One representative crop per transit as in grid mode, but
                     sickled and unsickled crops are shuffled together into a
                     single, unlabeled n x n grid (a blind comparison — no
                     per-cell class tag, only a small position number so a
                     cell can be looked up in the manifest CSV afterwards).

A companion CSV lists exactly which sample/cell_id (and, in grid/mixed mode,
which frame) landed at each position, so a montage image can always be traced
back to its source data — this is the only place mixed mode's ground-truth
class per cell is recorded.

Usage:
    python make_sickle_montage.py <compiled_dir> <out_path.png>
        [--mode {grid,filmstrip,mixed}] [--n-per-class N]
        [--grid-size N] [--seed SEED]

    <compiled_dir>   compile_experiment.py *_compiled/ dir (or its
                     experiment_data.xlsx) already annotated with sickle_class
    <out_path.png>   where to save the montage; a sibling
                     <stem>_manifest.csv is written alongside it
    --mode           grid (default), filmstrip, or mixed
    --n-per-class    transits to sample per class (default: 48 for grid, 4
                     for filmstrip; ignored for mixed — see --grid-size).
                     Fewer are used if a class doesn't have that many.
    --grid-size      mixed mode only: the grid is --grid-size x --grid-size,
                     split as evenly as possible between the two classes then
                     shuffled together (default: 6, i.e. a 6x6 grid of 36)
    --seed           random seed for a reproducible sample (default: random)
"""
import argparse
import csv
import math
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import browse_pt as bpt
from annotate_sickle_cells import Transit, _load_transits

_DISPLAY_SIZE = 96     # px per crop in grid mode (upscaled from the native
                       # crop size, same NEAREST convention as the annotator)
_FILM_HEIGHT  = 72     # px crop height in filmstrip mode (more, smaller crops
                       # per row than grid mode)
_FILM_MAX_WIDTH = 1100  # px target width a transit's frames wrap within, so
                        # a long transit grows down instead of very wide
_PAD          = 4      # px gap between crops
_MARGIN       = 12     # px outer margin around each class grid / the filmstrip
_BG           = 255    # white background
_TITLE_H      = 40     # px title bar height per class section (grid mode)
_ROW_LABEL_H  = 20     # px label height above each row (filmstrip mode)
_TITLES       = {'sickled': 'SICKLED', 'unsickled': 'UNSICKLED'}
_DEFAULT_N_PER_CLASS = {'grid': 48, 'filmstrip': 4}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Montage of random sickled vs unsickled cell crops from "
                    "an annotate_sickle_cells.py session.")
    parser.add_argument('compiled', type=str,
                        help="compile_experiment.py *_compiled/ dir, or its "
                             "experiment_data.xlsx")
    parser.add_argument('out_path', type=str,
                        help="Output .png path")
    parser.add_argument('--mode', choices=('grid', 'filmstrip', 'mixed'),
                        default='grid',
                        help="grid: one representative crop per transit, "
                             "sectioned by class. filmstrip: every frame of "
                             "each chosen transit, sectioned by class. "
                             "mixed: one representative crop per transit, "
                             "both classes shuffled into a single unlabeled "
                             "grid. (default: grid)")
    parser.add_argument('--n-per-class', type=int, default=None,
                        help="Transits to sample per class (default: 48 for "
                             "grid, 4 for filmstrip; ignored for mixed)")
    parser.add_argument('--grid-size', type=int, default=6,
                        help="mixed mode only: the grid is N x N (default: 6)")
    parser.add_argument('--seed', type=int, default=None,
                        help="Random seed, for a reproducible sample "
                             "(default: a new random sample each run)")
    args = parser.parse_args()
    if args.n_per_class is None and args.mode != 'mixed':
        args.n_per_class = _DEFAULT_N_PER_CLASS[args.mode]
    return args


def _resolve_compiled_paths(target: Path) -> tuple[Path, Path]:
    """(pt_path, parquet_path) for a *_compiled/ dir or its xlsx."""
    compiled_dir = target if target.is_dir() else target.parent
    pt_path = compiled_dir / 'concat_vqvae_cache.pt'
    parquet_path = compiled_dir / 'concat_vqvae_cache_metadata.parquet'
    if not pt_path.is_file() or not parquet_path.is_file():
        raise FileNotFoundError(
            f"{compiled_dir} has no concat_vqvae_cache.pt / "
            f"concat_vqvae_cache_metadata.parquet")
    return pt_path, parquet_path


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _sample_by_class(transits: list[Transit], classes: dict[tuple, str],
                     cls: str, n: int, rng: random.Random) -> list[Transit]:
    pool = [t for t in transits if classes.get(t.key) == cls]
    if not pool:
        return []
    k = min(n, len(pool))
    return rng.sample(pool, k)


def _representative_row(t: Transit) -> int:
    """The middle frame of a transit — a single crop stands in for the cell."""
    return t.rows[len(t.rows) // 2]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _read_crop(reader: 'bpt.CropReader', row: int,
               size: tuple[int, int] = None) -> Image.Image:
    size = size or (_DISPLAY_SIZE, _DISPLAY_SIZE)
    frame = reader.read(row)
    return Image.fromarray(frame, mode='L').resize(size, Image.NEAREST)


def _grid_shape(n: int) -> tuple[int, int]:
    cols = max(1, math.ceil(math.sqrt(n)))
    rows = max(1, math.ceil(n / cols))
    return rows, cols


def _render_section(crops: list[Image.Image], title: str,
                    cols: int = None,
                    captions: list[str] = None) -> Image.Image:
    """
    One titled grid of crops, as a single PIL image.

    `cols` forces an exact column count (used by mixed mode's n x n grid);
    left as None it falls back to _grid_shape's as-square-as-possible layout.
    `captions`, if given, draws a small label under each crop (mixed mode
    uses this for a position number, without revealing the class).
    """
    n = len(crops)
    if cols is None:
        rows, cols = _grid_shape(max(n, 1))
    else:
        rows = max(1, math.ceil(n / cols))
    cell_h = _DISPLAY_SIZE + (10 if captions else 0)
    cell = _DISPLAY_SIZE + _PAD
    grid_w = cols * cell + _PAD
    grid_h = rows * (cell_h + _PAD) + _PAD

    font = ImageFont.load_default(size=24)
    title_text = f'{title}  (n={n})'
    title_w = font.getlength(title_text) + 2 * _PAD
    img_w = max(grid_w, math.ceil(title_w))

    img = Image.new('L', (img_w, grid_h + _TITLE_H), color=_BG)
    draw = ImageDraw.Draw(img)
    draw.text((_PAD, 8), title_text, fill=0, font=font)

    cap_font = ImageFont.load_default(size=11) if captions else None
    for i, crop in enumerate(crops):
        r, c = divmod(i, cols)
        x = _PAD + c * cell
        y = _TITLE_H + _PAD + r * (cell_h + _PAD)
        img.paste(crop, (x, y))
        if captions:
            draw.text((x, y + _DISPLAY_SIZE), captions[i], fill=0, font=cap_font)

    return img


def _stack_sections(top: Image.Image, bottom: Image.Image) -> Image.Image:
    """Combine two class sections into one image, left-aligned, with a rule
    between them."""
    w = max(top.width, bottom.width)
    rule = 3
    out = Image.new('L', (w + 2 * _MARGIN,
                          top.height + rule + bottom.height + 2 * _MARGIN),
                    color=_BG)
    out.paste(top, (_MARGIN, _MARGIN))
    ImageDraw.Draw(out).rectangle(
        [(_MARGIN, _MARGIN + top.height + 1),
         (_MARGIN + w, _MARGIN + top.height + rule - 1)], fill=180)
    out.paste(bottom, (_MARGIN, _MARGIN + top.height + rule))
    return out


def _film_cell_size(reader: 'bpt.CropReader') -> tuple[int, int]:
    h, w = reader.frame_shape
    dh = _FILM_HEIGHT
    dw = max(1, round(w / h * dh)) if h else dh
    return dw, dh


def _render_filmstrip_transit(t: Transit, cls: str, reader: 'bpt.CropReader',
                              dw: int, dh: int, cols: int,
                              font: ImageFont.ImageFont) -> Image.Image:
    """
    One transit's full frame sequence, wrapped onto as many rows as needed to
    stay within _FILM_MAX_WIDTH (as annotate_sickle_cells.py's live viewer
    wraps a transit to the window, but here against a fixed target width
    since there is no window to measure), labeled above with its sample/
    cell_id.
    """
    n = len(t.rows)
    n_rows = math.ceil(n / cols)
    block_w = cols * (dw + _PAD) + _PAD
    block_h = n_rows * (dh + _PAD) + _PAD

    img = Image.new('L', (block_w, _ROW_LABEL_H + block_h), color=_BG)
    draw = ImageDraw.Draw(img)
    draw.text((_PAD, 2), f'{t.sample_name}  cell_id={t.cell_id}  '
                         f'({n} frames)', fill=0, font=font)

    for i, row in enumerate(t.rows):
        r, c = divmod(i, cols)
        crop = _read_crop(reader, row, (dw, dh))
        x = _PAD + c * (dw + _PAD)
        y = _ROW_LABEL_H + _PAD + r * (dh + _PAD)
        img.paste(crop, (x, y))

    return img


def _stack_filmstrip_section(blocks: list[Image.Image], title: str,
                             title_font: ImageFont.ImageFont) -> Image.Image:
    """One class's titled stack of wrapped transit blocks."""
    w = max((b.width for b in blocks), default=1)
    h = sum(b.height for b in blocks) + max(0, len(blocks) - 1) * _PAD

    img = Image.new('L', (w, _TITLE_H + h), color=_BG)
    ImageDraw.Draw(img).text((_PAD, 8), title, fill=0, font=title_font)

    y = _TITLE_H
    for b in blocks:
        img.paste(b, (0, y))
        y += b.height + _PAD
    return img


def _stack_filmstrip_montage(sickled: Image.Image,
                             unsickled: Image.Image) -> Image.Image:
    """Sickled section above unsickled, separated by the same rule
    _stack_sections uses for grid mode."""
    return _stack_sections(sickled, unsickled)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    target = Path(args.compiled)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pt_path, parquet_path = _resolve_compiled_paths(target)
    print(f'Loading {pt_path.name} / {parquet_path.name}...')

    obj, _info = bpt._read_pt(pt_path)
    viewable = dict(bpt.viewable_tensors(obj))
    if 'bf_u8' not in viewable:
        print(f'error: {pt_path.name} has no bf_u8 tensor', file=sys.stderr)
        sys.exit(1)
    reader = bpt.CropReader(pt_path, viewable['bf_u8'])

    transits, classes = _load_transits(parquet_path, reader.n)
    if not any(classes.values()):
        print(f'error: {parquet_path.name} has no sickle_class annotations '
              f'— run annotate_sickle_cells.py and press End first',
              file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)
    manifest_rows = []

    if args.mode == 'grid':
        sections = []
        for cls in ('sickled', 'unsickled'):
            chosen = _sample_by_class(transits, classes, cls, args.n_per_class, rng)
            print(f'  {cls}: {len(chosen)} transit(s) sampled '
                  f'(of {sum(1 for v in classes.values() if v == cls)} available)')
            crops = []
            for pos, t in enumerate(chosen):
                row = _representative_row(t)
                crops.append(_read_crop(reader, row))
                manifest_rows.append({
                    'sickle_class': cls, 'position': pos,
                    'experiment': t.experiment, 'sample_name': t.sample_name,
                    'cell_id': t.cell_id, 'n_frames': len(t.rows),
                    'representative_row': row,
                })
            sections.append(_render_section(crops, _TITLES[cls]))
        montage = _stack_sections(sections[0], sections[1])

    elif args.mode == 'mixed':
        n_total = args.grid_size * args.grid_size
        n_sickled = n_total // 2
        n_unsickled = n_total - n_sickled

        picks = []   # (cls, transit)
        for cls, k in (('sickled', n_sickled), ('unsickled', n_unsickled)):
            chosen = _sample_by_class(transits, classes, cls, k, rng)
            print(f'  {cls}: {len(chosen)} transit(s) sampled '
                  f'(of {sum(1 for v in classes.values() if v == cls)} available)')
            picks += [(cls, t) for t in chosen]
        rng.shuffle(picks)

        crops, captions = [], []
        for pos, (cls, t) in enumerate(picks):
            row = _representative_row(t)
            crops.append(_read_crop(reader, row))
            captions.append(str(pos))
            manifest_rows.append({
                'sickle_class': cls, 'position': pos,
                'experiment': t.experiment, 'sample_name': t.sample_name,
                'cell_id': t.cell_id, 'n_frames': len(t.rows),
                'representative_row': row,
            })
        title = (f'Sickled & unsickled - randomly mixed, unlabeled '
                f'(ground truth in the manifest CSV)')
        montage = _render_section(crops, title, cols=args.grid_size,
                                  captions=captions)

    else:   # filmstrip
        row_font = ImageFont.load_default(size=14)
        section_font = ImageFont.load_default(size=20)
        dw, dh = _film_cell_size(reader)
        cols = max(1, _FILM_MAX_WIDTH // (dw + _PAD))

        section_imgs = []
        for cls in ('sickled', 'unsickled'):
            chosen = _sample_by_class(transits, classes, cls, args.n_per_class, rng)
            print(f'  {cls}: {len(chosen)} transit(s) sampled '
                  f'(of {sum(1 for v in classes.values() if v == cls)} available)')
            blocks = []
            for pos, t in enumerate(chosen):
                blocks.append(_render_filmstrip_transit(
                    t, cls, reader, dw, dh, cols, row_font))
                manifest_rows.append({
                    'sickle_class': cls, 'position': pos,
                    'experiment': t.experiment, 'sample_name': t.sample_name,
                    'cell_id': t.cell_id, 'n_frames': len(t.rows),
                    'representative_row': '',
                })
            section_imgs.append(_stack_filmstrip_section(
                blocks, f'{_TITLES[cls]}  (n={len(blocks)})', section_font))
        montage = _stack_filmstrip_montage(section_imgs[0], section_imgs[1])

    reader.close()

    montage.save(str(out_path))
    print(f'Wrote {out_path}')

    manifest_path = out_path.with_name(out_path.stem + '_manifest.csv')
    with open(manifest_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            'sickle_class', 'position', 'experiment', 'sample_name',
            'cell_id', 'n_frames', 'representative_row'])
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f'Wrote {manifest_path}')


if __name__ == '__main__':
    main()
