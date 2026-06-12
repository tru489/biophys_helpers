import pandas as pd

h5 = r"E:\2026-05-22_tcell_act\20260611_235527_compiled\experiment_data.h5"
with pd.HDFStore(h5, mode='r') as store:
    keys = store.keys()
    print(f"Total keys: {len(keys)}")
    meta = store['/metadata']
    print(f"\n/metadata: {len(meta)} rows, columns: {list(meta.columns)}")
    print(meta[['sample_name','has_mass','has_volume','has_pairing','bm_gate_lower','bm_gate_upper']].to_string())

    # spot-check a normal sample
    mass = store['/samples/0h_pt1/mass']
    vol  = store['/samples/0h_pt1/volume']
    pair = store['/samples/0h_pt1/pairing']
    print(f"\n0h_pt1 mass: {mass.shape}, cols: {list(mass.columns[:5])}...")
    print(f"0h_pt1 volume: {vol.shape}, cols: {list(vol.columns[:5])}...")
    print(f"0h_pt1 pairing: {pair.shape}, has matched_mass: {'matched_mass' in pair.columns}")

    # spot-check hyphen-name sample (key sanitised)
    mass30 = store['/samples/30_5h_pt1/mass']
    print(f"\n30-5h_pt1 mass (key 30_5h_pt1): {mass30.shape}")
