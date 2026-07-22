# biophys_helpers

Helper scripts for analyzing biophysical single-cell data from **Coulter counter**
(cell volume), **SMR / suspended microchannel resonator** (buoyant mass), and
**imaging fluorescence exclusion (iFXM)** instruments.

The scripts fall into two kinds:

- **Batch / CLI** tools that process directories and write output non-interactively.
- **Interactive GUI** tools (tkinter / matplotlib) for gating, pairing, annotation,
  and browsing. These open a window and must be run on a machine with a display.

---

## Installation

The project ships a conda environment spec ([`environment.yaml`](environment.yaml)):

```bash
conda env create -f environment.yaml
conda activate biophys_helpers
```

This installs Python 3.12 plus `numpy`, `scipy`, `pandas`, `openpyxl`, `matplotlib`,
`pyyaml`, `h5py`, `pytables`, and `pillow`.

> **Notes**
> - When adding or referencing a new third-party dependency, add it to
>   [`environment.yaml`](environment.yaml) in the same change — including backends an
>   existing library needs (e.g. `openpyxl` for pandas `.xlsx` I/O). See
>   [`CLAUDE.md`](CLAUDE.md) for the full policy.
> - **openpyxl** is the backend `pandas` uses to write the `experiment_data.xlsx`
>   workbook from `compile_experiment.py`. It is an optional pandas extra, so a missing
>   install only surfaces at write time.
> - **PyTables** (the `tables` module) is what `pandas` uses to write the `data.h5`
>   HDFStore file from `pair_bm_runs.py`. Also an optional pandas backend; install via
>   conda (preferred on Windows: `conda install -n biophys_helpers pytables`) rather than pip.
> - **SciPy** is used by the SMR/FXM cross-correlation pairing
>   (`pair_smr_volumes.py` / `bulk_pair_smr_volumes.py`).
> - **Pillow** (`PIL`) is required by `browse_experiment.py` for image display.
> - The GUI tools (`crop_smr_timeseries`, `gate_*`, `pair_bm_runs`,
>   `annotate_coulter_samples`, `browse_experiment`) require a display. They are
>   cross-platform (macOS / Windows / Linux); on macOS the tkinter theme is forced to
>   `clam` so table row colors render correctly. Scripts that scan external
>   (exFAT/FAT) drives skip macOS AppleDouble sidecar files (`._*`) via
>   [`fsutil.py`](fsutil.py).

---

## Typical workflow

```
Coulter .#m4 files ──> extract_coulter_data.py ──> *_sc_volumes.csv ──> annotate_coulter_samples.py

SMR binaries ──> crop_smr_timeseries.py ──> (SMR software) ──> *_mass_results/       (buoyant mass)
iFXM images  ──> (FXM software) ─────────────────────────────> *_imaging_fxm_results/ (volume)
        │                                                            │
        │        pair_smr_volumes.py / bulk_pair_smr_volumes.py  <───┘
        │        (per-cell mass↔volume cross-correlation pairing)
        │                              │
        │                              ▼
        │                *_pairing_results/*_PairedSMRVolumes.csv
        │                              │
        ├─> aggregate_bm_vol_files.py ─┤  (collect BM + iFXM volumes ──> mass_pg.csv)
        │        │                     │
        │        ▼                     │
        │   gate_bm_coulter.py         │  (gate one CSV into a new dir)
        │   gate_experiments_inplace.py│  (gate per-sample, in place ──> gate YAMLs)
        │        │                     │
        │   pair_bm_runs.py            │  (population-level multi-fluid density grouping)
        │                              │
        └─> calculate_baseline_density.py ──> *_baseline_density.csv  (absolute-density offset)
                                       │
                                       ▼
                        compile_experiment.py ──> *_compiled/{experiment_data.xlsx, images.h5}
                                       │
                                       ▼
                           browse_experiment.py  (interactive viewer)

Housekeeping: prune_timestamped_subdirs.py
```

---

## Coulter counter

### `extract_coulter_data.py`  · _batch_

Parses a directory of Coulter counter `.#m4` files and writes output files into a
timestamped `<YYYYMMDD-HHMMSS>_coulter-processed/` directory. Outputs include:

- `<dirname>_sc_volumes.csv` — all single-cell volume measurements (ungated), one column per file, NaN-padded to equal length
- `<dirname>_volume_stats.csv` — summary statistics pre-selected in the Multisizer software, one column per file
- `fig/<sample>.png` — ungated histogram per sample

**Usage**

```
python extract_coulter_data.py <directory> [-stats] [-single-stats] [-r]
```

| Argument | Description |
|---|---|
| `directory` | Path to folder containing `.#m4` files |
| _(no flags)_ | Write single-cell volumes CSV and histograms (default) |
| `-stats` | Write only the stats CSV (no volumes, no histograms) |
| `-single-stats` | Write both the single-cell volumes CSV and the stats CSV + histograms |
| `-r` | Recursively include `.#m4` files from subdirectories; column names are prefixed with the relative subdir path |

**Examples**

```bash
# Write single-cell volumes CSV and histograms (default)
python extract_coulter_data.py "E:/data/my_experiment"

# Write only the stats CSV
python extract_coulter_data.py "E:/data/my_experiment" -stats

# Write both single-cell volumes CSV and stats CSV + histograms
python extract_coulter_data.py "E:/data/my_experiment" -single-stats

# Include .#m4 files from all subdirectories
python extract_coulter_data.py "E:/data/my_experiment" -r
```

---

## SMR / buoyant mass

### `crop_smr_timeseries.py`  · _GUI_

Interactive GUI for cropping junk data from raw SMR binary run files. A directory
containing three binary files (`prefix_frequencies`, `prefix_valvestates`,
`prefix_time`) is loaded and displayed chunk-by-chunk so that large datasets
(millions to tens of millions of points) can be navigated efficiently. You mark one
or more removal regions on the frequency-vs-time plot, then save the cropped binary
files to a new directory.

**Workflow**

1. The three binary files in the supplied directory are discovered and loaded into memory.
2. The frequency signal is shown in fixed-size chunks; **Prev / Next** navigate between chunks.
3. To mark a removal region, click **"Set lower boundary"** then click the plot to place
   the lower boundary (red dashed line); then **"Set upper boundary"** and click to place
   the upper boundary. The region between is shaded red.
4. Multiple regions can be added across any chunks. **Delete** removes a committed region.
5. **Crop & Save** writes the three cropped binary files (and copies any `.json` / image
   files) to `<input_dir>_cropped/` alongside the input directory.

**Usage**

```
python crop_smr_timeseries.py <directory> [--chunk-size N]
```

| Argument | Description |
|---|---|
| `directory` | Path containing the three SMR binary files |
| `--chunk-size N` | Number of data points per displayed chunk (default: `100000`) |

**Examples**

```bash
# Crop a run with the default chunk size
python crop_smr_timeseries.py "E:/data/2026-05-22_run01"

# Use a larger chunk for faster navigation of very long runs
python crop_smr_timeseries.py "E:/data/2026-05-22_run01" --chunk-size 500000
```

---

### `aggregate_bm_vol_files.py`  · _batch_

Aggregates buoyant mass (BM) CSVs from SMR runs and/or FXM volume
(`_ProcessedVolumes.csv`) CSVs from iFXM runs across one or more experiment
superdirectories. Results are copied into a timestamped output directory organised
by data type and superdir name. Also writes a combined `mass_pg.csv` summary and
per-sample histogram PNGs.

```
<YYYYMMDD-HHMMSS>_aggregated/
  smr_data/
    <superdir_name>/
      <csv files>
  imaging_fxm/
    <superdir_name>/
      <csv files>
  mass_pg.csv
  fig/
    mass_pg/
      <sample>.png
```

**Expected directory structure**

BM files (produced by SMR software) — searched recursively for any directory
named `*_mass_results`; all CSVs within (except `curation_index*.csv`) are collected:
```
<superdir>/
  .../<any_depth>/
    <name>_mass_results/
      <csv files>
```

FXM volume files (produced by `FXMAnalysis.py`):
```
<superdir>/
  <sample_subdir>/
    <YYYYMMDD_HHMMSS>_imaging_fxm_results/
      stage2_analysis/
        <sample_name>_ProcessedVolumes.csv
```

**Usage**

```
python aggregate_bm_vol_files.py <superdir1> [superdir2 ...]
python aggregate_bm_vol_files.py --from-file <paths.txt> [--output <output_dir>]
python aggregate_bm_vol_files.py --summary-only <aggr_dir>
```

| Argument | Description |
|---|---|
| `directories` | One or more superdir paths (positional) |
| `--from-file FILE` | Text file listing superdir paths, one per line |
| `--output DIR` | Parent directory for the output folder (default: parent of first superdir) |
| `--summary-only DIR` | Path to an existing aggregated directory; re-generates `mass_pg.csv` from its `smr_data/` contents without re-copying files |

`--from-file` and positional `directories` are mutually exclusive.
`--output` is compatible with both `directories` and `--from-file`.
`--summary-only` is mutually exclusive with all other arguments.

**Examples**

```bash
# Single superdir
python aggregate_bm_vol_files.py "E:/data/6um_silica"

# Multiple superdirs
python aggregate_bm_vol_files.py "E:/data/6um_silica" "E:/data/10um_silica"
find expm_dir -type d -name "l1210_data_*" | xargs python aggregate_bm_vol_files.py

# Superdirs listed in a text file
python aggregate_bm_vol_files.py --from-file my_experiments.txt

# Specify where the output folder is written
python aggregate_bm_vol_files.py "E:/data/6um_silica" --output "E:/results"
python aggregate_bm_vol_files.py --from-file my_experiments.txt --output "E:/results"

# Re-generate mass_pg.csv from an existing aggregated directory
python aggregate_bm_vol_files.py --summary-only "E:/results/20260324-123456_aggregated"
```

**Example path list file (`my_experiments.txt`)**

```
E:/data/6um_silica
E:/data/10um_silica
E:/data/control_run
```

---

### `pair_smr_volumes.py`  · _batch_

Pairs individual SMR **buoyant-mass** measurements with individual iFXM **volume**
measurements for the same cells, by cross-correlating the mass-vs-time and
calibrated-volume-vs-time signals to recover the time lag between the two
instruments, then matching peaks within a tolerance window. Reads a volume-only
`*_ProcessedVolumes.csv` and a `*_mass_results` mass CSV from a single experiment
(analysis) directory and writes a fully-populated ProcessedVolumes-format CSV plus
diagnostic figures.

The output `*_PairedSMRVolumes.csv` is exactly the `*_pairing_results` file that
`compile_experiment.py` (and `gate_experiments_inplace.py` for iFXM) discover
automatically, so pairing feeds straight into compilation.

**Output** (`<analysis_dir>/<YYYYMMDD_HHMMSS>_pairing_results/`)

```
<prefix>_PairedSMRVolumes.csv         paired per-cell mass + volume rows
<prefix>_PairingStats_fig.png         pairing summary
<prefix>_PairingLags_fig.png          cross-correlation lag diagnostics
<prefix>_PairingHistograms_fig.png    matched-population histograms
```

**Usage**

```
python pair_smr_volumes.py <analysis_dir> [options]
```

| Argument | Description |
|---|---|
| `analysis_dir` | Experiment folder containing both a `*_imaging_fxm_results` and a `*_mass_results` subdir |
| `--vol-dir DIRNAME` | Name of the imaging results dir (auto-detected — most recent — if omitted) |
| `--mass-dir DIRNAME` | Name of the mass results dir (auto-detected — most recent — if omitted) |
| `--timebase FLOAT` | Time-axis resolution in seconds (default: `1e-3`) |
| `--peak-tolerance INT` | Index window around each mass peak for matching (default: `11`) |
| `--gaussian-width INT` | Gaussian blur sigma in samples (default: `15`) |

**Example**

```bash
python pair_smr_volumes.py "E:/data/2026-06-03_drugtreat/zota_24h_samp05"
```

---

### `bulk_pair_smr_volumes.py`  · _batch_

Batch driver for `pair_smr_volumes.py`: discovers experiment folders under a root
directory and runs the same pairing on each. By default it searches recursively for
any folder containing both a `*_imaging_fxm_results` and a `*_mass_results` subdir;
`--no-recursive` restricts to a fixed depth-2 layout (`<root>/<date-superfolder>/<experiment>/`).
Depth-1 superfolders beginning with `YYYY-MM-DD` can be filtered by date.

**Usage**

```
python bulk_pair_smr_volumes.py <root_dir> [folder-selection] [pairing-options]
```

**Folder selection**

| Argument | Description |
|---|---|
| `root_dir` | Root directory containing date-named experiment superfolders |
| `--from YYYY-MM-DD` | Only process superfolders dated on or after this date |
| `--to YYYY-MM-DD` | Only process superfolders dated on or before this date |
| `--last N` | Only process the N most recently dated superfolders |
| `--from-file FILE` | Text file of experiment folder paths (one per line; `#` comments allowed) |
| `--skip-paired` | Skip folders that already contain a `*_PairedSMRVolumes.csv` |
| `--no-recursive` | Restrict discovery to fixed depth-2 instead of recursive search |
| `--dry-run` | Print discovered folders without running any pairing |

**Pairing options** (passed through to each pairing run)

| Argument | Description |
|---|---|
| `--timebase FLOAT` | Time-axis resolution in seconds (default: `1e-3`) |
| `--peak-tolerance INT` | Index window around each mass peak (default: `11`) |
| `--gaussian-width INT` | Gaussian blur sigma in samples (default: `15`) |
| `--utc-offset HOURS` | Hours added to mass `real_time_s` timestamps to align with FXM frame times (e.g. `-4` for EDT when LabVIEW logs UTC; default: `0`) |

**Examples**

```bash
# All experiments under root, last 3 dated superfolders
python bulk_pair_smr_volumes.py "E:/experiments" --last 3

# Date range, skip already-paired folders
python bulk_pair_smr_volumes.py "E:/experiments" --from 2026-06-01 --to 2026-06-07 --skip-paired

# Preview what would be processed
python bulk_pair_smr_volumes.py "E:/experiments" --from 2026-06-03 --dry-run
```

---

### `calculate_baseline_density.py`  · _batch_

Computes the **fluid baseline density** for every sample in an experiment superdir
from its buoyant-mass data, and writes a single timestamped summary CSV. For each
sample subdir the newest `*_mass_results` CSV is read (the same discovery convention
as `gate_experiments_inplace.py`), the mean of the per-cell `avg_baseline` column is
taken, and it is converted to a density via a base-frequency / density calibration:

```
baseline_density = (rfreq - mean_avg_baseline - intercept) / slope
```

`slope` and `intercept` come from a calibration JSON; `rfreq` is the experiment's
resonant frequency (Hz). This value is the absolute-density offset to add to the
**relative** `buoyant_density` written by `compile_experiment.py`. Without a
calibration JSON the mean baseline is still reported but `baseline_density` is `NaN`.

**Output** (written inside `<superdir>`, at the same level as the sample subdirs)

```
<YYYYMMDD_HHMMSS>_baseline_density.csv
    one row per sample: sample, n_cells, mean_avg_baseline, baseline_density,
    rfreq, slope, intercept, source_csv
```

**Usage**

```
python calculate_baseline_density.py <superdir> --rfreq <hz> [--calib-json <path>]
```

| Argument | Description |
|---|---|
| `superdir` | Experiment superdir whose immediate subdirs are samples, each with a `*_mass_results` folder |
| `--rfreq HZ` | Resonant frequency in Hz (required) |
| `--calib-json PATH` | Calibration JSON with `slope` and `intercept` keys (optional; `baseline_density` is `NaN` if omitted) |

**Example**

```bash
python calculate_baseline_density.py "E:/data/2026-05-22_tcell_act" \
    --rfreq 1167183 --calib-json "E:/data/density_calibration.json"
```

---

## Gating

Two gating tools share the histogram / cutoff UI in [`gating/common.py`](gating/common.py).
Use `gate_bm_coulter.py` to gate a **single aggregated CSV** into a new output directory,
or `gate_experiments_inplace.py` to gate **per-sample data in place** (writing a YAML gate
file into each sample subfolder, which downstream tools like `pair_bm_runs.py` and
`compile_experiment.py` pick up automatically).

Both cutoff windows include an **X-axis view** control (min/max entry boxes with
**Apply** / **Reset**): typing a range re-bins and redraws the histograms over just
that span, so you can zoom into a single size population before placing cutoffs
without changing the underlying data.

### `gate_bm_coulter.py`  · _GUI_

Interactive GUI for applying upper/lower cutoffs to columns of a `mass_pg.csv` or
single-cell volumes CSV file. Supports both buoyant mass (linear scale) and Coulter
counter volume (log scale) data.

**Workflow**

1. A dialog asks whether you are gating **Buoyant Mass** or **Coulter Counter Volume** data.
   This sets the histogram scale, bin layout, and output directory naming.
2. A scrollable multi-select list of all column names is shown with a running
   counter of columns remaining.
3. Select one or more columns and click **"Set cutoffs for selection"**. A histogram
   window opens showing the selected columns overlaid with shared bin edges.
4. Click once on the histogram to set a lower cutoff (red dashed line), then again
   to set an upper cutoff (blue dashed line). The accepted region is shaded green.
   Use **Reset** to start over, **Accept** to confirm.
5. Repeat steps 3–4 until all columns are assigned. **Done** then becomes available.
6. Output is written to a timestamped directory alongside the input file.

**Output structure**

```
<YYYYMMDD-HHMMSS>_gated_bm_data/    (or _gated_cc_data/ for Coulter Counter Volume)
  <stem>_cutoff.csv         gated data (values outside cutoffs removed, columns NaN-padded)
  <stem>_cutoff_log.txt     per-column removal statistics
  <stem>_cutoff_stats.csv   descriptive statistics on gated data (n, mean, median, mode, std, CV)
  histograms/
    group_01.png            overlaid histogram for each cutoff group, with cutoff lines
    group_02.png
    ...
```

**Usage**

```
python gate_bm_coulter.py <csv_file>
```

| Argument | Description |
|---|---|
| `csv_file` | Path to a CSV file where each column is a dataset (no row index). Typically `mass_pg.csv` from `aggregate_bm_vol_files.py` or a `sc_volumes.csv` from `extract_coulter_data.py`. |

**Examples**

```bash
# Gate buoyant mass data
python gate_bm_coulter.py "E:/results/20260324-123456_aggregated/mass_pg.csv"

# Gate Coulter counter volume data
python gate_bm_coulter.py "E:/data/20260324-123456_coulter-processed/my_experiment_sc_volumes.csv"
```

**Example cutoff log (`mass_pg_cutoff_log.txt`)**

```
apply_bm_cutoffs — Buoyant Mass Cutoff Log
============================================================
Input:   E:/results/.../mass_pg.csv
Output:  mass_pg_cutoff.csv
Run:     2026-03-24 14:32:01

Cutoff groups
------------------------------------------------------------
Group 1   lower = 5.2 pg   upper = 45.7 pg
  tester_6um_silica_rep1_2026-02-10...csv    1842 → 1801  (41 removed, 2.2%)
  tester_6um_silica_rep2_2026-02-10...csv    1956 → 1923  (33 removed, 1.7%)

Total: 3798 → 3724 values retained across 2 column(s)  (74 removed, 1.9%)
```

---

### `gate_experiments_inplace.py`  · _GUI_

Interactive GUI for gating buoyant mass (BM) or iFXM volume data across all sample
subfolders in an experiment superdir, writing the gate bounds **back into each sample
subfolder** as a YAML file (plus a timestamped summary folder in the superdir). One
column per sample is shown in the GUI.

**Workflow**

1. A data-type dialog asks whether you are gating **Buoyant Mass** or **iFXM Volume** data.
2. The script discovers the relevant data file for each sample subdir and loads the
   target column (`mass_pg` for BM, `volume` for iFXM).
3. A scrollable list of sample names is shown. Multi-select a group and click
   **"Set cutoffs for selection"**.
4. A histogram window opens showing all selected samples overlaid with shared bin
   edges. Click to set lower then upper cutoffs.
5. Steps 3–4 repeat until all samples are assigned; **Done** becomes available.
6. **← Back** undoes the last group of cutoffs, restoring those samples to the
   remaining list (can be pressed repeatedly).

**Output**

A YAML gate file written into each sample subfolder, and a summary folder in the superdir:

```
<sample_subdir>/<sample_subdir_name>_<mode>_gate.yaml      (mode = bm | ifxm-vol)
<superdir>/<YYMMDD.HHMMSS>_<mode>_gating_summary/
  cutoff_log.txt
  cutoff_stats.csv
  histograms/group_NN.png
```

**Expected directory structure**

```
BM:    <superdir>/<sample_subdir>/<name>_mass_results/<date>_<name>.csv          (column: mass_pg)
iFXM:  <superdir>/<sample_subdir>/<YYYYMMDD_HHMMSS>_imaging_fxm_results/
           stage2_analysis/<sample>_ProcessedVolumes.csv                          (column: volume)
```

**Usage**

```
python gate_experiments_inplace.py <superdir>
```

| Argument | Description |
|---|---|
| `superdir` | Path to the experiment superdir containing sample subdirs |

**Example**

```bash
python gate_experiments_inplace.py "E:/data/2026-05-22_tcell_act"
```

---

## Pairing, annotation & compilation

### `pair_bm_runs.py`  · _GUI_

Interactive spreadsheet-like GUI for organizing buoyant mass runs from multiple fluid
conditions (h2o, d2o, optiprep) into paired/triplet groups for **population-level SMR
density analysis**. Optionally associates Coulter counter volume data with each sample
group.

**Workflow**

1. Discovers all sample subdirs under the given superdir that contain a `*_mass_results`
   folder with a `mass_pg` CSV. Any `*_bm_gating` folder present is pre-loaded so its
   gate thresholds are saved automatically.
2. If `--coulter` is given, that CSV's column names become a per-row **"Coulter Col"**
   dropdown. Setting a Coulter column on any run in a group auto-fills it for all runs
   in that group.
3. Assign each sample a run type (h2o / d2o / optiprep), optional Coulter column, and
   group ID. Custom metadata columns can be added (free-text, checkbox, or
   shared-within-group). **Set Cells…** bulk-assigns a value to a chosen column across
   all selected rows.
4. **Group Selected** associates multiple runs as one biological sample. **Done** writes output.

**Output** (`<superdir>/YYYYMMDD_HHMMSS_populationlevel_smr_pairing/`)

```
metadata.csv   one row per run; all assigned attributes + gate thresholds
data.h5        /metadata DataFrame
               /data/{gid}/{run_type}   full mass CSV per run
               /data/{gid}/coulter      Coulter volumes per group (if a Coulter column assigned)
```

**Usage**

```
python pair_bm_runs.py <superdir> [--coulter <csv_path>]
```

| Argument | Description |
|---|---|
| `superdir` | Path to the experiment superdir |
| `--coulter CSV` | Optional Coulter counter CSV whose columns are sample names |

**Examples**

```bash
# Pair BM runs only
python pair_bm_runs.py "E:/data/2026-05-22_tcell_act"

# Pair BM runs and associate Coulter volume columns
python pair_bm_runs.py "E:/data/2026-05-22_tcell_act" --coulter "E:/data/coulter_sc_volumes_cutoff.csv"
```

---

### `annotate_coulter_samples.py`  · _GUI_

A stripped-down sibling of `pair_bm_runs.py` for **per-sample annotation of Coulter
samples** — no run types, no grouping. Each column of a summary Coulter CSV becomes one
table row; you add custom metadata columns (free-text or yes/no checkbox), edit cells in
place, and bulk-fill with **Set Cells…**.

**Output** (`<csv_dir>/YYYYMMDD_HHMMSS_coulter_sample_annotation/`)

```
metadata.csv       one row per sample; sample_name + all annotation columns
<input_csv_name>   a copy of the input single-cell Coulter CSV, columns
                   reordered to match the metadata row order
```

The `sample_name` column of `metadata.csv` holds the exact column headers of the
copied CSV, so the two files cross-reference by name (and, after reordering, by
position). Both outputs are plain CSVs, so annotation mistakes can be fixed by
editing `metadata.csv` directly.

**Usage**

```
python annotate_coulter_samples.py <coulter_csv>
```

| Argument | Description |
|---|---|
| `coulter_csv` | Summary Coulter CSV whose columns are sample names and rows are volume measurements |

**Example**

```bash
python annotate_coulter_samples.py "E:/data/coulter-processed/my_experiment_sc_volumes.csv"
```

---

### `compile_experiment.py`  · _batch_

Automatically discovers and compiles **all** per-sample data for an experiment into a
pair of HDF5 files. Given a superdir, it walks each sample subdir, finds every known data
type (BM mass, iFXM volume, pairing, gating, images), loads them, and writes structured
output. Most-recent is used if multiple of a type exist.

**Recognised sub-subdir types** (all optional)

```
*_mass_results          mass CSV with mass_pg column
*_imaging_fxm_results   stage2_analysis/*_ProcessedVolumes.csv;
                        stage1_image_processing/*_CELLGROUPED.hdf5;
                        stage2_analysis/*_Hdf5PathIndex.csv
*_pairing_results       *_PairedSMRVolumes.csv
*_bm_gating             YAML with lower/upper thresholds
*_ifxm-vol_gating       YAML with lower/upper thresholds
```

**Output** (`<superdir>/YYYYMMDD_HHMMSS_compiled/`)

```
experiment_data.xlsx  (Excel workbook — shareable, no HDF5 tooling needed)
  metadata sheet         one row per sample (summary + gate values); the
                         sheet_name column names each sample's worksheet
  <one sheet per sample> three side-by-side blocks, one blank spacer column
                         between them, single header row, columns prefixed so
                         they split back apart in code (df.filter(like='vol_')):
                           VOLUME (vol_)   every FXM cell: volume_au, volume_fL
                           MASS   (mass_)  every SMR cell: mass_pg
                           PAIRED (pair_)  matched cells: mass_pg, volume_au,
                                           volume_fL, buoyant_density
                         buoyant_density is RELATIVE (add the experiment
                         baseline for absolute g/mL)
  README sheet           units, block meaning, and a pandas read recipe

images.h5            (h5py)
  /{safe_name}/{transit_idx:05d}/bf   (n_frames, H, W) uint8
  /{safe_name}/{transit_idx:05d}/fl   (n_frames, H, W) uint16
```

> The `buoyant_density` column is **relative**. Add the per-sample offset from
> `calculate_baseline_density.py` to get absolute density in g/mL.

**Usage**

```
python compile_experiment.py <superdir>
```

| Argument | Description |
|---|---|
| `superdir` | Path to the experiment superdir containing sample subdirs |

**Example**

```bash
python compile_experiment.py "E:/data/2026-05-22_tcell_act"
```

---

### `browse_experiment.py`  · _GUI_

Interactive viewer for a `*_compiled/` directory produced by `compile_experiment.py`.
Displays three per-sample boxplots (volume, buoyant mass, density) with clickable scatter
overlays; clicking any data point loads that transit's BF image frames into a scrollable
panel. Up to 3 transit panels stack at the bottom (FIFO eviction of the oldest).

> Requires **Pillow** (`PIL`) for image display — see Installation notes.

**Usage**

```
python browse_experiment.py <compiled_dir>
```

| Argument | Description |
|---|---|
| `compiled_dir` | Path to a `*_compiled/` directory containing `experiment_data.h5` and (optionally) `images.h5` |

**Example**

```bash
python browse_experiment.py "E:/data/2026-05-22_tcell_act/20260611_235527_compiled"
```

---

## Housekeeping

### `prune_timestamped_subdirs.py`  · _batch_

For each sample subdir inside an experiment superdir, finds all directories whose names
begin with a timestamp prefix (`YYYYMMDD.HHMMSS` or `YYYYMMDD_HHMMSS`), groups them by
suffix (everything after the timestamp), and deletes all but the **most recent** in each
group. Useful for clearing out repeated analysis runs.

**Usage**

```
python prune_timestamped_subdirs.py <superdir> [--dry-run]
```

| Argument | Description |
|---|---|
| `superdir` | Path to the experiment superdir containing sample subdirs |
| `--dry-run` | Print what would be kept/deleted without deleting anything |

**Examples**

```bash
# Preview what would be deleted (recommended first)
python prune_timestamped_subdirs.py "E:/data/2026-05-22_tcell_act" --dry-run

# Actually delete older runs
python prune_timestamped_subdirs.py "E:/data/2026-05-22_tcell_act"
```

---

## Supporting modules

These are imported by the scripts above and are not run directly.

| Module | Role |
|---|---|
| [`CoulterFile.py`](CoulterFile.py) | `CoulterFile` class that parses a single Coulter counter `.#m4` file — selection statistics, histogram bin edges/counts, single-cell diameters/volumes, and start time. Used by `extract_coulter_data.py`. |
| [`gating/common.py`](gating/common.py) | Shared GUI components and output utilities for the gating tools: `CutoffWindow` (click-to-set histogram with X-axis view control), `MainWindow` (scrollable sample list), `ask_data_type_dialog`, `save_group_histograms`, `write_stats_csv`, `write_log`. Used by `gate_bm_coulter.py` and `gate_experiments_inplace.py`. |
| [`pipeline/stage2/pairing_utils.py`](pipeline/stage2/pairing_utils.py) | Cross-correlation pairing primitives (SciPy-based signal building, lag estimation, peak matching) shared by `pair_smr_volumes.py` and `bulk_pair_smr_volumes.py`. |
| [`fsutil.py`](fsutil.py) | Stdlib-only filesystem helpers shared across scripts. `is_appledouble()` detects macOS AppleDouble sidecars (`._*`) so scans of exFAT/FAT external drives skip them. |
