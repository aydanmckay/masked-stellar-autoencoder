import h5py
import tqdm
from astropy.io import fits
import numpy as np
import glob
import random

row_limit = 2_000_000  # Maximum number of rows per dataset
filelist = glob.glob('partialtable*.fits')
if not filelist:
    raise FileNotFoundError("No FITS files found matching pattern 'partialtable*.fits'")
random.shuffle(filelist)

progress_bar = tqdm.tqdm(filelist, total=len(filelist))

for file in progress_bar:
    try:
        with fits.open(file, memmap=True) as hdul:
            if len(hdul) < 2:
                print(f"Warning: {file} has insufficient HDUs, skipping")
                continue
            data = hdul[1].data # Convert FITS data to NumPy array
            if data is None or len(data) == 0:
                print(f"Warning: {file} contains no data, skipping")
                continue
    except Exception as e:
        print(f"Error processing {file}: {e}, skipping")
        continue

        dataset_base_name = 'sslset' + file.split('.')[0].split('ll')[-1]

        total_rows = data.shape[0]
        num_chunks = (total_rows + row_limit - 1) // row_limit  # Ceiling division

        with h5py.File('pretrain_dataset_incomplete.h5', 'a') as hf:
            for i in range(num_chunks):
                start_idx = i * row_limit
                end_idx = min(start_idx + row_limit, total_rows)

                chunk = data[start_idx:end_idx]  # Extract chunk
                dataset_name = f"{dataset_base_name}_part{i}"

                hf.create_dataset(dataset_name, data=chunk)