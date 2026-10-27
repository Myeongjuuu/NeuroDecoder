# NeuroDecoder

EEG-based imagined speech decoding for alternative communication in individuals with aphasia.

## Overview

This project investigates whether EEG signals recorded during imagined speech — words mentally rehearsed but never spoken — can be decoded to identify the intended word. The goal is to enable alternative communication for individuals who cannot speak due to conditions such as aphasia.

We recruited 40 healthy participants, collected 16-channel EEG in a shielded soundproof booth using an OpenBCI Cyton system, and built a complete decoding pipeline: frequency-domain feature extraction with a filter bank and CSP, followed by classification with an Echo State Network (ESN).

The public inner-speech EEG dataset (Nieto et al., 2022) was also incorporated to validate the pipeline on higher-density recordings.

## Method

1. **Experiment design** — 40 healthy adults (aged 20–30); 10 imagined-speech target words (8 device-control + 2 condition words); recorded in a shielded soundproof booth with a 16-ch OpenBCI Cyton system
2. **Data** — self-collected dataset (40 subjects, 10 words) + [Nieto et al. (2022)](https://doi.org/10.1038/s41597-022-01147-2) public inner-speech EEG dataset
3. **Preprocessing** — MNE-Python: detrending (baseline drift removal) → min-max normalization per channel
4. **Feature extraction** — filter bank decomposition (0–60 Hz, 30 bands at 2 Hz intervals); Fisher Ratio per band per channel to identify discriminative frequency–channel pairs; Common Spatial Pattern (CSP) spatial filtering
5. **Decoding** — Echo State Network (ESN) with reservoir-based recurrent structure, online learning, winner-take-all readout over 10 word classes

## Repository Contents

```
EEG Inner Speech Decoding/     # Original analysis notebooks
  preprocessing_*.ipynb        # EEG preprocessing pipelines
  filter_bank.ipynb            # Filter bank + Fisher Ratio feature extraction
  filter_bank_csp.ipynb        # Filter bank + CSP + SVM classification
  psd.ipynb                    # Power spectral density analysis
  psd_comparison.ipynb         # PSD comparison across classes
  CSP.ipynb                    # CSP classification
  Wavelet_Transform.ipynb      # Wavelet-based analysis
  check_env.py                 # Dependency check script
SOURCE_MANIFEST.json           # SHA-256 of original source files
```

## Stack

Python · MNE-Python · NumPy · SciPy · scikit-learn · Matplotlib

## Reference

N. Nieto et al., "Thinking out loud, an open-access EEG-based BCI dataset for inner speech recognition," *Scientific Data*, 2022. [https://doi.org/10.1038/s41597-022-01147-2](https://doi.org/10.1038/s41597-022-01147-2)
