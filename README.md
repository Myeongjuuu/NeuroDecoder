# NeuroDecoder

Python tools for imagined-speech EEG. The OpenBCI workflow uses an ESN to decode ten word cues from marked recordings. The public EEG workflow analyzes directional Fisher ratios and can replay a four-direction ESN.

## Run

The pinned environment was checked with Python 3.11.9. Place recordings in `data/`.

```powershell
conda create -n neurodecoder python=3.11.9 -y
conda activate neurodecoder
python -m pip install -r requirements.txt
neurodecoder --help

neurodecoder openbci --csv data/result_06.csv --sfreq 125 --output results/openbci.csv
neurodecoder preprocess --bdf data/sub-01_ses-01_task-innerspeech_eeg.bdf --output-dir results/epochs
neurodecoder public-fisher --epochs data/sub-01_ses-01_eeg-epo.fif --events data/sub-01_ses-01_events.dat --trusted-legacy-events --output results/fisher
```

Use the recording's actual sampling rate for OpenBCI. For `public-fisher`, use the matching official FIF and DAT from [OpenNeuro ds003626](https://openneuro.org/datasets/ds003626). The legacy DAT requires a trusted source. Add `--esn` to replay the ESN. Run any command with `--help` for its options. The pinned setup includes the dependencies for `preprocess --iclabel`. For a base install, use `python -m pip install -e .`.

## Code (src/)

| Location | Role |
| --- | --- |
| `cli.py` | Command entry point |
| `data/read_openbci_csv.py` | Read CSV word markers and extract trials |
| `data/preprocess_biosemi_bdf.py` | Filter, resample, and epoch BDF into FIF |
| `data/read_public_eeg_epochs.py` | Match public FIF and events, then extract inner-speech windows |
| `analysis/bandpass_filters.py` | Shared filter-bank definitions and filtering |
| `analysis/compute_public_fisher.py` | Compare action and relaxation EEG by direction |
| `models/decode_openbci_words.py` | Prepare features and run the ten-word decoder |
| `models/ridge_esn.py` | Sparse reservoir with ridge readout |
| `models/online_readout_esn.py` | Reservoir with a readout that updates after each labeled trial |
