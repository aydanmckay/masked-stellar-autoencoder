from astropy.table import Table
import os
from astroquery.gaia import Gaia
import time
from concurrent.futures import ProcessPoolExecutor
import glob

# pans_gaia_ids.fits was created using another query:
# SELECT source_id, original_ext_source_id
# FROM gaiadr3.panstarrs1_best_neighbour
ids = Table.read('/arc/projects/k-pop/catalogues/adql_matches/pans_gaia_ids.fits')
chunk_size = 2_000_000
total_rows = len(ids)
num_workers = 3

already_done = glob.glob('/arc/projects/k-pop/catalogues/adql_matches/panstarrsdr1_matches/ids*')
ids_done = [int(num.split('_')[-1].split('.')[0]) for num in already_done]

query = """SELECT pans.obj_id,
       pans.g_mean_psf_mag, pans.g_mean_psf_mag_error, pans.r_mean_psf_mag, pans.r_mean_psf_mag_error,
       pans.i_mean_psf_mag, pans.i_mean_psf_mag_error, pans.z_mean_psf_mag, pans.z_mean_psf_mag_error,
       pans.y_mean_psf_mag, pans.y_mean_psf_mag_error, pans.obj_info_flag, pans.quality_flag
FROM gaiadr2.panstarrs1_original_valid AS pans
WHERE pans.obj_id IN (
    SELECT original_ext_source_id FROM user_amckay.chunked_table{chunknum}
)"""

def process_chunk(num, chunk):
    try:
        # access remote and upload chunk
        Gaia.login(credentials_file='/path/to/credentials_file.txt')
        Gaia.upload_table(upload_resource=chunk, table_name=f'chunked_table{num}', verbose=True)

        modified_query = query.format(chunknum=num)
        
        # perform xmatch
        job = Gaia.launch_job_async(modified_query, name=f'pansdr1_{num}',
                                    dump_to_file=True, output_format='fits',
                                    output_file=f'/arc/projects/k-pop/catalogues/adql_matches/panstarrsdr1_matches/ids_pansdr1_{num}.fits.gz')
        
        # login again if xmatch takes a long time
        Gaia.login(credentials_file='/path/to/credentials_file.txt')
        # remove the user table and job for memory
        Gaia.remove_jobs([job.jobid])
        Gaia.delete_user_table(table_name=f"chunked_table{num}")

    except Exception as e:
        print(f"Error processing chunk {num}: {e}")

# start parallel processing
with ProcessPoolExecutor(max_workers=num_workers) as executor:
    num = 0
    futures = []

    for last_idx in range(0, total_rows, chunk_size):
        if num in ids_done:
            # skip ids done already that seemed to break
            num += 1
        else:
            chunk = ids[last_idx : min(last_idx + chunk_size, total_rows)]
            future = executor.submit(process_chunk, num, chunk)
            futures.append(future)
            num += 1

    # Wait for all tasks to complete
    for future in futures:
        future.result()