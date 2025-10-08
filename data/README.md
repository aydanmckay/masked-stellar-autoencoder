# Data

The data fed to the model has a specific structure which is created using the following dataset making scripts. The pre-training dataset has the column format of source_id, CatWISE bands W1 and W2, *Gaia* G, BP, and RP, followed by the 55 BP coefficients, the 55 RP coefficients, magnitudes from any other photometric surveys included, and finally positional information not desired to be reconstructed by the model. To create the fine-tuning dataset, the labels are included between the source_ids and W1, having the format of label, label_error, repeating. More surveys can be included in the datasets given that the columns are updated in the [config](../configs) files.

---

## Pre-Training Dataset Creation Proceedure

- source_ids_x_file_names.py to create the reference file for all the XP continuous and Gaia source files.
- photometric-dataset-handling.ipynb to clean the photometric datasets before combining if needed.
- smssdr4_filtering.ipynb cleaning specifically the Sky-Mapper DR4 dataset as it is in a different format.
- pretraining-partial-table-maker.py to make all the small partial tables for the pre-training dataset in its original form.
- combine-partial-tables.py to crunch all the files together into 1 HDF file.
- add-gaia-features.py example on how to add more features from the gaia source files if desired, like the proper motions.

- data_validator.py provides scripts to validate the data in the pre-training dataset file.

---

## Pre-Training Data

- CatWISE - table_1_catwise.fits comes from the Andrae+2023 data as it is a pre computed xmatch with XP spectra, work is done for us already.
    - [Documentation](https://irsa.ipac.caltech.edu/data/WISE/CatWISE/gator_docs/catwise_colDescriptions.html)
    - [Data](https://zenodo.org/records/7945154)
- *Gaia* DR3 - *Gaia* DR3 source and XP continuous data were provided as 3386 files sliced identically by source_id, such that some XP continuous files contained no information as no stars within a source_id range contained XP continuous measurements.
    - [Gaia Source Documentation](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_main_source_catalogue/ssec_dm_gaia_source.html)
    - [Gaia Source Data](https://sdsc-users.flatironinstitute.org/~gaia/dr3/hdf5/GaiaSource/)
    - [XP Continuous Documentation](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_spectroscopic_tables/ssec_dm_xp_continuous_mean_spectrum.html)
    - [XP Continuous Data](https://cdn.gea.esac.esa.int/Gaia/gdr3/Spectroscopy/xp_continuous_mean_spectrum/)
- Sky-Mapper DR4 - Data was provided as numerous files containing pre-computed crossmatches with *Gaia* DR3 as a subset of columns within the DR4 files. 
    - [Documentation](https://skymapper.anu.edu.au/data-release/)
    - [Data](https://skymapper.anu.edu.au/_data/DR4/)
- Pan-STARRS DR1 - Matched using source_id column via multiple asynchronous ADQL queries ([example](example_adql_match.py)) through the *Gaia* Archive, where pre computed crossmatches have already exist courtesy of the *Gaia* Data Processing and Analysis Consortium (DPAC).
    - [Documentation](https://outerspace.stsci.edu/display/PANSTARRS)
    - [Gaia Archive Documentation](https://gea.esac.esa.int/archive/documentation/GDR3/Catalogue_consolidation/chap_crossmatch/sec_crossmatch_externalCat/ssec_crossmatch_panstarrs.html)
- 2MASS - Matched using source_id column via multiple asynchronous ADQL queries through the *Gaia* Archive.
    - [Documentation](https://irsa.ipac.caltech.edu/Missions/2mass.html)
    - [Gaia Archive Documentation](https://gea.esac.esa.int/archive/documentation/GDR3/Catalogue_consolidation/chap_crossmatch/sec_crossmatch_externalCat/ssec_crossmatch_2mass.html)
- SDSS DR13 - Matched using source_id column via multiple asynchronous ADQL queries through the *Gaia* Archive.
    - [Gaia Archive Documentation](https://gea.esac.esa.int/archive/documentation/GDR3/Catalogue_consolidation/chap_crossmatch/sec_crossmatch_externalCat/ssec_crossmatch_sdss.html)

topcat stilts was used to concatenate the numerous tables matched through adql and astroquery.gaia into singular HDF files for 2MASS, SDSS, and Pan-STARRS.

---

## Fine-Tuning Data

- sdss-apogee-dr17.fits         : Binary table from SDSS APOGEE APSCAP data release
    - [Documentation](https://data.sdss.org/datamodel/files/APOGEE_ASPCAP/APRED_VERS/ASPCAP_VERS/allStar.html)
- apogee_astroNN-DR17.fits      : Binary table from Leung, Bovy, & Mackereth last updated in 2021
    - [Accessible here](https://www.sdss.org/dr18/data_access/value-added-catalogs/?vac_id=85)
- nn_latent_age_dr17.csv.gz     : CSV table from Leung, Bovy, Mackereth & Miglio in 2023
    - [Accessible here](https://github.com/henrysky/astroNN_ages)
- galah_dr4_allstar_240705.fits : Binary table from GALAH 4th data release
    - [Documentation](https://www.galah-survey.org/dr4/the_catalogues/#galah-dr4-main-catalogues)
- li_et_al_x_gaiaids.fits       : Binary table of very metal-poor stars from LAMOST cross-matched with Gaia DR3
    - Matched with astroquery.gaia based on RA and Dec
    - [Paper](https://ui.adsabs.harvard.edu/abs/2022ApJ...931..147L/abstract)

