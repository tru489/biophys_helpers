# biophys_helpers

Helper scripts for analyzing biophysical data from Coulter counter, SMR, and imaging fluorescence exlusion (iFXM) instruments.

---

## Scripts

### `extract_coulter_data.py`

Parses a directory of Coulter counter `.#m4` files and writes one or both of:

- `<dirname>_single_cell_volumes.csv` — single-cell volume measurements, one column per
  file, NaN-padded to equal length
- `<dirname>_volume_stats.csv` — summary statistics pre-selected in the Multisizer software,
  one column per file

**Usage**

```
python extract_coulter_data.py <directory> [-stats] [-single] [-r]
```

| Argument | Description |
|---|---|
| `directory` | Path to folder containing `.#m4` files |
| `-stats` | Write only the stats CSV |
| `-single` | Write only the single-cell volumes CSV |
| `-r` | Recursively include `.#m4` files from subdirectories; column names are prefixed with the relative subdir path |

If neither `-stats` nor `-single` is provided, both files are written.

**Examples**

```bash
# Write both output files
python extract_coulter_data.py "E:/data/my_experiment"

# Write only the stats CSV
python extract_coulter_data.py "E:/data/my_experiment" -stats

# Write only the single-cell volumes CSV
python extract_coulter_data.py "E:/data/my_experiment" -single

# Include .#m4 files from all subdirectories
python extract_coulter_data.py "E:/data/my_experiment" -r
```

---

### `aggregate_bm_vol_files.py`

Aggregates buoyant mass (BM) CSVs from SMR runs and/or FXM volume
(`_ProcessedVolumes.csv`) CSVs from FXM runs across one or more experiment
superdirectories. Results are copied into a timestamped `<timestamp>_aggregated/`
folder, organised by data type and superdir name. Additionally writes a combined
`mass_pg.csv` and per-sample histogram PNGs.

```
<timestamp>_aggregated/
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
```

| Argument | Description |
|---|---|
| `directories` | One or more superdir paths (positional) |
| `--from-file FILE` | Text file listing superdir paths, one per line |
| `--output DIR` | Parent directory for the `aggregated/` folder |

`--from-file` and positional `directories` are mutually exclusive.
`--output` is compatible with both input modes.

**Examples**

```bash
# Single superdir
python aggregate_bm_vol_files.py "E:/data/6um_silica"

# Multiple superdirs
python aggregate_bm_vol_files.py "E:/data/6um_silica" "E:/data/10um_silica"

# Superdirs listed in a text file
python aggregate_bm_vol_files.py --from-file my_experiments.txt

# Specify where aggregated/ is written
python aggregate_bm_vol_files.py "E:/data/6um_silica" --output "E:/results"
python aggregate_bm_vol_files.py --from-file my_experiments.txt --output "E:/results"
```

**Example path list file (`my_experiments.txt`)**

```
E:/data/6um_silica
E:/data/10um_silica
E:/data/control_run
```
