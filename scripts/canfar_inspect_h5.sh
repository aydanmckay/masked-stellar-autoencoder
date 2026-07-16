#!/bin/bash
set -eu
pip install -q h5py
python3 << 'PYEOF'
import h5py

f = h5py.File("/arc/projects/k-pop/catalogues/andrae2023/sslset-realmags-full-052725.h5", "r")
keys = list(f.keys())
print(f"Top-level groups: {len(keys)}")

for k in keys[:3]:
    ds = f[k]
    print(f"\n  {k}: shape={ds.shape}, dtype={ds.dtype}")
    if hasattr(ds, "dtype") and ds.dtype.names:
        print(f"    columns ({len(ds.dtype.names)}):")
        for col in ds.dtype.names:
            print(f"      {col}")

print("\n  ...")
total = sum(f[k].shape[0] for k in keys)
print(f"\nTotal rows across all keys: {total:,}")

k0 = keys[0]
arr = f[k0][:]
print(f"\nKey '{k0}': {arr.shape}, {arr.nbytes / 1e6:.1f} MB")
if arr.dtype.names:
    print(f"  All {len(arr.dtype.names)} columns: {list(arr.dtype.names)}")

f.close()
PYEOF
