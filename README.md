# NeuroDecoder

NeuroDecoder is a Python toolkit for offline imagined-speech EEG analysis with two workflows.

- **Self-collected BEL EEG** uses EGI MFF recordings, sample-aligned word markers, and the BEL130 sensor montage for ten-word ESN decoding. The acquisition protocol assumes 60 seconds of initial rest. The current decoder uses the marked word trials for training and evaluation.
- **Public OpenNeuro EEG** uses four directions from the Thinking out loud dataset. Fisher analysis compares the action windows with a separate 15-second resting baseline. ESN decoding can also be evaluated across sessions.

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

Save these three derivative files in `data/` with their original names.

- [EEG epochs · sub-01_ses-01_eeg-epo.fif](https://s3.amazonaws.com/openneuro.org/ds003626/derivatives/sub-01/ses-01/sub-01_ses-01_eeg-epo.fif)
- [Trial metadata · sub-01_ses-01_events.dat](https://s3.amazonaws.com/openneuro.org/ds003626/derivatives/sub-01/ses-01/sub-01_ses-01_events.dat)
- [Resting baseline · sub-01_ses-01_baseline-epo.fif](https://s3.amazonaws.com/openneuro.org/ds003626/derivatives/sub-01/ses-01/sub-01_ses-01_baseline-epo.fif)

```powershell
neurodecoder public-fisher --epochs data/sub-01_ses-01_eeg-epo.fif --events data/sub-01_ses-01_events.dat --baseline data/sub-01_ses-01_baseline-epo.fif --trusted-legacy-events --output results/fisher
```

The analysis compares 2.5 s task windows with nonoverlapping 2.5 s windows from the same session’s 15 s resting baseline. The baseline windows come from one continuous recording. Add `--esn` to run the four-direction replay. Use `--trusted-legacy-events` only with trusted DAT files.

## ESN evaluation across sessions

Use the EEG FIF and event DAT for each session from the same OpenNeuro dataset. The command selects the input representation and reservoir settings on the training sessions, then evaluates a fixed readout on the test session.

```powershell
neurodecoder public-esn `
  --train data/sub-01_ses-01_eeg-epo.fif data/sub-01_ses-01_events.dat `
  --train data/sub-01_ses-02_eeg-epo.fif data/sub-01_ses-02_events.dat `
  --test data/sub-01_ses-03_eeg-epo.fif data/sub-01_ses-03_events.dat `
  --trusted-legacy-events --output results/public-esn.csv
```

## Other workflows

For BEL recordings, place the MFF directory and its marker CSV in `data/`. The CSV requires a sample-aligned `Data` column with paired word codes 2–11. The sampling rate is read from the recording. Use `--marker-offset-samples` if CSV row zero maps to a later recording sample.

```powershell
neurodecoder bel --recording data/inner_kkj_0226.mff --markers data/result01.csv --montage BEL130.sfp --output results/bel.csv
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
│   ├── read_bel_recording.py      EGI recording, montage, and word marker alignment
│   ├── preprocess_biosemi_bdf.py   BDF filtering, resampling, and epoching
│   └── read_public_eeg_epochs.py  Public EEG and trial metadata alignment
├── analysis/
│   ├── bandpass_filters.py        Shared filter-bank definitions
│   └── compute_public_fisher.py   Directional Fisher analysis and ESN replay
└── models/
    ├── decode_bel_words.py        Feature extraction and ten-word decoding
    ├── ridge_esn.py               Sparse reservoir and ridge readout
    ├── online_readout_esn.py      Trial-by-trial ESN readout updates
    └── tune_public_esn.py          Training-session tuning and held-out evaluation
```
