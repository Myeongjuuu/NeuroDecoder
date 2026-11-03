# NeuroDecoder

NeuroDecoder is a Python toolkit for offline imagined-speech EEG analysis. It includes a ten-word Echo State Network (ESN) for marked OpenBCI recordings and a four-direction workflow for public EEG. The public workflow compares action and relaxation periods using Fisher ratios and supports optional ESN replay.

## Installation

Run these commands from the repository root.

```powershell
conda create -n neurodecoder python=3.11.9 -y
conda activate neurodecoder
python -m pip install -r requirements.txt
neurodecoder --help
```

## Public EEG dataset

The public workflow uses **Thinking out loud**, the inner-speech EEG dataset by Nieto et al. The example below uses subject 01, session 01. [OpenNeuro ds003626](https://openneuro.org/datasets/ds003626/versions/2.1.2) · [Dataset paper](https://doi.org/10.1038/s41597-022-01147-2)

Save both derivative files in `data/` with their original names.

- [EEG epochs · sub-01_ses-01_eeg-epo.fif](https://s3.amazonaws.com/openneuro.org/ds003626/derivatives/sub-01/ses-01/sub-01_ses-01_eeg-epo.fif)
- [Trial metadata · sub-01_ses-01_events.dat](https://s3.amazonaws.com/openneuro.org/ds003626/derivatives/sub-01/ses-01/sub-01_ses-01_events.dat)

```powershell
neurodecoder public-fisher --epochs data/sub-01_ses-01_eeg-epo.fif --events data/sub-01_ses-01_events.dat --trusted-legacy-events --output results/fisher
```

Add `--esn` to run the four-direction replay. Use `--trusted-legacy-events` only with trusted DAT files.

## Other workflows

For a marked OpenBCI CSV, supply the sampling rate of the recording. The example uses 125 Hz.

```powershell
neurodecoder openbci --csv data/result_06.csv --sfreq 125 --output results/openbci.csv
```

To preprocess a raw BioSemi recording with embedded triggers into 256 Hz FIF epochs, place the BDF in `data/` and run the following command. Add `--iclabel` to enable ICA-based artifact removal.

```powershell
neurodecoder preprocess --bdf data/sub-01_ses-01_task-innerspeech_eeg.bdf --output-dir results/epochs
```

## Source layout

```text
src/
├── cli.py                         CLI entry point
├── data/
│   ├── read_openbci_csv.py         OpenBCI trial extraction from word markers
│   ├── preprocess_biosemi_bdf.py   BDF filtering, resampling, and epoching
│   └── read_public_eeg_epochs.py  Public EEG and trial metadata alignment
├── analysis/
│   ├── bandpass_filters.py        Shared filter-bank definitions
│   └── compute_public_fisher.py   Directional Fisher analysis and ESN replay
└── models/
    ├── decode_openbci_words.py    Feature extraction and ten-word decoding
    ├── ridge_esn.py               Sparse reservoir and ridge readout
    └── online_readout_esn.py      Trial-by-trial ESN readout updates
```
