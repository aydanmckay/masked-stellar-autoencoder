# masked-stellar-autoencoder

The **Masked Stellar Autoencoder (MSA)** is a deep learning foundation model designed to reconstruct and analyze **Gaia low-resolution spectra** for Galactic archaeology. By leveraging masked autoencoding and residual architectures, MSA learns robust representations of *Gaia* BP/RP spectra, photometry, and positional information and enables fine-tuned predictions of key stellar labels.

---

## Features

- **Masked autoencoding**: Learns to reconstruct missing or masked spectral regions, improving robustness to incomplete data.  
- **Residual encoder–decoder architecture**: Captures nonlinear stellar features while preserving fine spectral details.  
- **Multi-label fine-tuning**: Predicts stellar parameters including:  
  - Effective temperature (*T*<sub>eff</sub>)  
  - Metallicity ([Fe/H])  
  - Alpha enhancement ([α/Fe])  
  - Surface gravity (*log g*)  
  - Stellar age
  - Parallax ($\varpi$)  
- **Quantile regression**: Provides uncertainty-aware predictions with enforced quantile ordering at 16<sup>th</sup>, 50<sup>th</sup>, and 84<sup>th</sup> intervals.  
- **Applications beyond Gaia magnitude limits**: Infers stellar parameters even for stars too faint to have low-resolution spectra in *Gaia* DR3.  

---

## Scientific Motivation

- **Galactic archaeology**: Use stellar parameters to trace the formation and evolution of the Milky Way.  
- **Ultra metal-poor stars**: Identify and characterize ancient stellar populations.  
- **Dark matter dominated systems**: Search for chemo-dynamical signatures of dwarf galaxies and globular clusters.  
- **Survey integration**: Bridge *Gaia* with complementary spectroscopic surveys (APOGEE, GALAH, etc.).  

---

## Repository Structure

```bash
data/ # Scripts or instructions for data preprocessing
models/ # Model architectures (Masked Autoencoder, prediction heads)
training/ # Training loops, schedulers, and loss functions
notebooks/ # Metrics, validation scripts, visualization tools, and exploratory analysis
configs/ # Config file examples for running the model
README.md # This file
requirements.txt # Python dependencies
```

---

## Installation

```bash
git clone https://github.com/aydanmckay/masked-stellar-autoencoder.git
cd masked-stellar-autoencoder
pip install -r requirements.txt
```

Requirements include:
* torch (PyTorch)
* sklearn (scikit-learn)
* numpy, pandas, scipy
* astropy, h5py
* matplotlib
* [rtdl_num_embeddings](https://github.com/yandex-research/rtdl-num-embeddings)

---

## Usage

### Pretraining (Masked Autoencoding)
```bash
python training/pretrain_msa.py --config configs/pretrain.yaml
```
### Fine-tuning on Stellar Labels
```bash
python training/finetune_msa.py --config configs/finetune.yaml
```
### Evaluation

Evaluation is performed in notebooks in notebooks/

---

## Results

* Reconstruction of masked *Gaia* XP spectra
* Improved metallicity and age predictions compared to traditional regression models
* Robust generalization across surveys

---

## Citation
If you use this project in your research, please cite:
```bash
@article{mckay2025msa,
  title={Extending the Reach of Gaia with Masked Stellar Autoencoders},
  author={McKay, Aydan and Fabbro, Sebastien},
  year={2025},
  journal={In preparation}
}
```

---

Developed by Aydan McKay, as part of MSc research in Galactic Archaeology and machine learning applications to stellar populations.
