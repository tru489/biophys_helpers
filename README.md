# biophys_helpers

Helper scripts for analyzing biophysical data from Coulter counter, SMR, and imaging fluorescence exclusion (iFXM) instruments.

---

## Scripts

### `extract_coulter_data.py`

Parses a directory of Coulter counter `.#m4` files and writes output files into a
timestamped `<YYYYMMDD-HHMMSS>_coulter-processed/` directory. Outputs include:

- `<dirname>_sc_volumes_ungated.csv` — all single-cell volume measurements, one column per file, NaN-padded to equal length
- `<dirname>_sc_volumes_gated.csv` — single-cell volumes for files that have Multisizer gate data
- `<dirname>_volume_stats.csv` — summary statistics pre-selected in the Multisizer software, one column per file
- `fig/<sample>.png` — histogram per sample (gated where available, ungated as fallback)

**Usage**

```
python extract_coulter_data.py <directory> [-stats] [-single] [-r]
```

| Argument | Description |
|---|---|
| `directory` | Path to folder containing `.#m4` files |
| `-stats` | Write only the stats CSV |
| `-single` | Write only the single-cell volumes CSVs and histograms |
| `-r` | Recursively include `.#m4` files from subdirectories; column names are prefixed with the relative subdir path |

If neither `-stats` nor `-single` is provided, both are written.

**Examples**

```bash
# Write all output files
python extract_coulter_data.py "E:/data/my_experiment"

# Write only the stats CSV
python extract_coulter_data.py "E:/data/my_experiment" -stats

# Write only the single-cell volumes CSV and histograms
python extract_coulter_data.py "E:/data/my_experiment" -single

# Include .#m4 files from all subdirectories
python extract_coulter_data.py "E:/data/my_experiment" -r
```

---

### `aggregate_bm_vol_files.py`

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

### `apply_bm_cutoffs.py`

Interactive GUI for applying upper/lower cutoffs to columns of a `mass_pg.csv` file
(typically produced by `aggregate_bm_vol_files.py`). The user groups columns, sets
shared cutoffs visually via overlaid histograms, and repeats until all columns are
processed.

**Workflow**

1. A scrollable multi-select list of all column names is shown with a running
   counter of columns remaining.
2. Select one or more columns and click **"Set cutoffs for selection"**. A histogram
   window opens showing the selected columns overlaid with shared linear bin edges.
3. Click once on the histogram to set a lower cutoff (red dashed line), then again
   to set an upper cutoff (blue dashed line). The accepted region is shaded green.
   Use **Reset** to start over, **Accept** to confirm.
4. Repeat steps 2–3 until all columns are assigned. **Done** then becomes available.
5. Output is written to a timestamped directory alongside the input file.

**Output structure**

```
<YYYYMMDD-HHMMSS>_gated_bm_data/
  <stem>_cutoff.csv         gated data (values outside cutoffs removed, columns NaN-padded)
  <stem>_cutoff_log.txt     per-column removal statistics
  histograms/
    group_01.png            overlaid histogram for each cutoff group, with cutoff lines
    group_02.png
    ...
```

**Usage**

```
python apply_bm_cutoffs.py <mass_pg.csv>
```

| Argument | Description |
|---|---|
| `mass_pg.csv` | Path to a CSV file where each column is a dataset (no row index) |

**Examples**

```bash
python apply_bm_cutoffs.py "E:/results/20260324-123456_aggregated/mass_pg.csv"
```

**Example cutoff log (`mass_pg_cutoff_log.txt`)**

```
apply_bm_cutoffs — Cutoff Log
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
