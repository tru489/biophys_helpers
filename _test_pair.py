from pair_bm_runs import _discover_runs, _load_coulter
from pathlib import Path

runs = _discover_runs(Path(r"E:\2026-05-22_tcell_act"))
print(f"Runs: {len(runs)}")

coulter = _load_coulter(Path(r"E:\2026-05-14_silica_beads_fl5s_coulter\2026-05-14_silica_beads_fl5s_sc_volumes_cutoff.csv"))
print(f"Coulter cols ({len(coulter.columns)}): {list(coulter.columns)}")
print(f"Coulter rows: {len(coulter)}")
print("OK")
