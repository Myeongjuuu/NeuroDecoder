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

The public workflow uses **Thinking out loud**, the inner-speech EEG dataset by Nieto et al. It contains 10 participants (`sub-01` through `sub-10`), with three sessions per participant. [Download from OpenNeuro](https://openneuro.org/datasets/ds003626/versions/2.1.2) · [Dataset paper](https://doi.org/10.1038/s41597-022-01147-2)

In the dataset's `derivatives/sub-XX/ses-YY/` folders, download the matching `*_eeg-epo.fif` and `*_events.dat` files for each session. For Fisher analysis, also download the matching `*_baseline-epo.fif`. Save the files in the repository's `data/` directory with their original names.

### Fisher analysis

Choose any participant and session with matching files. The analysis compares 2.5 s task windows with nonoverlapping windows of the same length from a single 15 s resting recording.

```powershell
$id = Read-Host 'Participant and session ID (sub-XX_ses-YY)'
neurodecoder public-fisher --epochs "data/${id}_eeg-epo.fif" --events "data/${id}_events.dat" --baseline "data/${id}_baseline-epo.fif" --trusted-legacy-events --output "results/fisher/${id}"
```

Add `--esn` to run a within-session replay. Use `--trusted-legacy-events` only with trusted DAT files.

### ESN evaluation across all participants

The evaluation protocol trains a separate ESN for each participant. It selects the input representation and reservoir settings using sessions 01 and 02, then evaluates the fixed readout on session 03. Use `--compare` to run the standard ESN, filter-bank CSP + ESN, and direction-wise Fisher-selected CSP + ESN on the same split. Both CSP variants use the same reservoir. The number of selected components (8 or 16) is chosen within the training sessions. The loop writes one predictions CSV per participant.

```powershell
foreach ($index in 1..10) {
    $subject = 'sub-{0:00}' -f $index
    neurodecoder public-esn `
      --train "data/${subject}_ses-01_eeg-epo.fif" "data/${subject}_ses-01_events.dat" `
      --train "data/${subject}_ses-02_eeg-epo.fif" "data/${subject}_ses-02_events.dat" `
      --test "data/${subject}_ses-03_eeg-epo.fif" "data/${subject}_ses-03_events.dat" `
      --compare --trusted-legacy-events --output "results/public-esn/${subject}.csv"
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed for $subject" }
}
```

## Other workflows

For BEL recordings, place the MFF directory and its marker CSV in `data/`. The CSV requires a sample-aligned `Data` column with paired word codes 2–11. The sampling rate is read from the recording. Use `--marker-offset-samples` if CSV row zero maps to a later recording sample.

```powershell
neurodecoder bel --recording data/inner_kkj_0226.mff --markers data/result01.csv --montage BEL130.sfp --output results/bel.csv
```

To preprocess a raw BioSemi recording with embedded triggers into 256 Hz FIF epochs, place the BDF in `data/` and run the following command. Add `--iclabel` to enable ICA-based artifact removal.

```powershell
neurodecoder preprocess --bdf data/recording.bdf --output-dir results/epochs
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
