import glob

import h5py
import numpy as np

source_files = np.sort(glob.glob("gaia/GaiaSource/*"))

# ⚡ Bolt Optimization: Open output HDF5 file once outside the loop instead of repeatedly
with h5py.File("gaia/source_ids_x_file_names.h5", "a") as hf_out:
    for file in source_files:
        with h5py.File(file, "r") as f:
            ids = f["source_id"][:]
            xpq = f["has_xp_continuous"][:]
            filename = file.split("/")[-1].split(".")[0]

            # ⚡ Bolt Optimization: Bypass Pandas overhead by assembling directly to NumPy structured arrays
            dtype = [("source_id", ids.dtype), ("has_xp_coeffs", xpq.dtype)]
            dataset_to_save = np.empty(len(ids), dtype=dtype)
            dataset_to_save["source_id"] = ids
            dataset_to_save["has_xp_coeffs"] = xpq

            hf_out.create_dataset(filename, data=dataset_to_save)
