import pandas as pd
import h5py
import numpy as np
from astropy.table import Table
import glob
import tqdm
from natsort import natsorted
import time

file = h5py.File('pretrain_dataset_incomplete.h5')
keylist = list(file.keys())
source_files = np.sort(glob.glob('gaia/GaiaSource/*'))

keylist = natsorted(keylist)

source_file_num = 0
pbar = tqdm.tqdm(enumerate(keylist), total=len(keylist))
for it, dataset_name in pbar:  
    # if it < 64:
    #     source_file_num = 849
    #     time.sleep(1)
    # else:
    array1 = file[dataset_name][:]  # Load dataset from single file
    # print(f'Decoding: {dataset_name}')
    new_dtypes = []
    array3 = []
    for name in array1.dtype.names:
        col = array1[name]
        # print(col.dtype.kind)
        if col.dtype.kind in {'S', 'U'}:  # If the column contains byte strings or unicode
            array3.append(np.array([np.nan if v in {b'', ''} else float(v) for v in col], dtype='>f8'))
            new_dtypes.append((name, '>f8'))
        else:
            array3.append((col))  # Convert other numeric types to float3
            new_dtypes.append((name, col.dtype.str))
    structured_rows = list(zip(*array3))
    array1 = np.array(structured_rows, dtype=new_dtypes)
    
    reference_size = array1.shape[0]
    matched_records = []  # Initialize matched records for each dataset
    matched_mask = np.zeros(reference_size, dtype=bool)  # Track matched rows

    undermatched=True
    # print(f"Processing dataset: {dataset_name}")
    while undermatched:

        file_path = source_files[source_file_num]
        
        # print(f"Processing file: {file_path}")

        if np.all(matched_mask):  # Stop early if all rows are matched
            # print("All rows matched. Breaking the loop.")
            break

        with h5py.File(file_path, "r") as f:
            file_data = {
                'source_id': f['source_id'][:],
                'ruwe': f['ruwe'][:],
                'pmra': f['pmra'][:],
                'pmdec': f['pmdec'][:],
                'e_pmra': f['pmra_error'][:],
                'e_pmdec': f['pmdec_error'][:],
                'pmradec_corr': f['pmra_pmdec_corr'][:],
                'g_flux_error': f['phot_g_mean_flux_error'][:],
                'bp_flux_error': f['phot_bp_mean_flux_error'][:],
                'rp_flux_error': f['phot_rp_mean_flux_error'][:],
                'e_parallax': f['parallax_error'][:]
            }

            # print('Loaded data into dict')

            dtype = [(key, file_data[key].dtype) for key in file_data.keys()]
            selected_columns = list(file_data.keys())
            selected_columns.remove('source_id')

            # Create structured array for file_data
            array2 = np.zeros(file_data['source_id'].shape, dtype=dtype)
            for key in file_data.keys():
                array2[key] = file_data[key]

            # print('Converted file data to structured array')

        # Sort arrays by 'source_id' to optimize matching
        array1_sorted_idx = np.argsort(array1["source_id"])
        array2_sorted_idx = np.argsort(array2["source_id"])

        array1 = array1[array1_sorted_idx]
        array2 = array2[array2_sorted_idx]

        # Get the sorted source_id columns
        ref_ids = array1["source_id"]
        file_ids = array2["source_id"]

        # Use np.searchsorted to find matching positions for all rows at once
        file_indices = np.searchsorted(file_ids, ref_ids)

        # Ensure file_indices are within bounds (i.e., < len(file_ids))
        file_indices = np.clip(file_indices, 0, len(file_ids) - 1)

        # Create the valid_matches mask
        valid_matches = file_ids[file_indices] == ref_ids

        # Create the combined dtype for the result
        combined_dtype = array1.dtype.descr + [(key, array2.dtype[key]) for key in selected_columns]

        # Extract the matching records in one go
        matched_array1 = array1[valid_matches]
        matched_array2 = array2[file_indices[valid_matches]]

        # Create the result array and copy data from both arrays
        result = np.empty(len(matched_array1), dtype=combined_dtype)
            
        # Copy from array1 (all columns)
        for col in array1.dtype.names:
            result[col] = matched_array1[col]

        # Copy from array2 (selected columns)
        for col in selected_columns:
            result[col] = matched_array2[col]

        # Store the result
        # print(f"Finished processing file: {file_path}")
        matched_records.append(result)

        # Update matched_mask to reflect matched rows
        matched_mask[valid_matches] = True  # Mark the matched rows as True

        # Stop early if all rows are matched
        if np.all(matched_mask):
            # print("All rows matched. Breaking the loop.")
            undermatched = False
            break

        source_file_num += 1

    # After processing all files for a dataset, combine matched records
    if matched_records:
        final_result = np.concatenate(matched_records, axis=0)
        # print(f"Finished processing dataset: {dataset_name}")
    else:
        final_result = np.empty(0, dtype=combined_dtype)

    # Do something with the final result (e.g., save, analyze, etc.)
    # final_result could be saved or processed further here

    
    with h5py.File('220M_pretrain_data.h5', 'a') as hf:
        hf.create_dataset(dataset_name, data=final_result)