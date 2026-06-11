from pair_bm_runs import _discover_runs
from pathlib import Path
runs = _discover_runs(Path(r"E:\2026-05-22_tcell_act"))
print(f"Found {len(runs)} samples")
for name, v in list(runs.items())[:5]:
    gate = v["bm_gate"]
    gate_str = f"gate=({gate[0]:.3f}, {gate[1]:.3f})" if gate else "no gate"
    print(f"  {name}: {len(v['data'])} rows, {gate_str}")
n_gated = sum(1 for v in runs.values() if v["bm_gate"])
print(f"Total with gate: {n_gated}")
