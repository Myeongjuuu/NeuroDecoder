# Inner Speech EEG Analysis

EEG preprocessing, run-held-out CSP classification, and exploratory task/rest connectivity analysis for the [Inner Speech dataset, ds003626 v2.1.2](https://doi.org/10.18112/openneuro.ds003626.v2.1.2).

The original notebooks remain unchanged in their two historical folders. New shared code lives in `src/inner_speech`; new notebooks call that code instead of duplicating it. The group pipeline is implemented, but a validated group-level scientific finding is **not** claimed. See [validation status](docs/validation.md) and [analysis plan and limitations](docs/analysis_plan.md).

## Layout

```text
configs/default.yaml              # Shared analysis settings
src/inner_speech/
  config.py                       # Paths and parameter validation
  events.py                       # Run/condition tracking and strict parsing
  preprocessing.py                # Filtering, resampling, ICA, paired QC
  classification.py               # CSP + LDA/SVM inside grouped CV
  connectivity.py                 # Matched task/rest wPLI-squared, coherence, PLV
  statistics.py                   # Participant differences, max-T, effect estimates
  time_frequency.py               # Multitaper TFR
  cli.py                          # Staged execution
notebooks/
  01_preprocessing.ipynb
  02_csp_classification.ipynb
  03_coherence_analysis.ipynb
  04_plv_analysis.ipynb
  05_time_frequency.ipynb
tests/                            # Synthetic regression/scientific guards
docs/analysis_plan.md
docs/validation.md
data/README.md                    # Dataset preparation, no recordings
results/generated/                # Ignored local outputs and provenance
experiments/README.md             # Historical EEGNet attempt status
EEG Inner Speech Decoding/        # Unmodified original snapshot
Imagined Speech Decoding using Functional Connectivity/  # Original snapshot
SOURCE_MANIFEST.json              # SHA-256 of 17 original source files
```

## Install and configure

Python 3.11 is the tested interpreter. From the repository root:

```bash
python -m venv .venv
# Activate the virtual environment for your shell.
python -m pip install -e ".[dev]"
```

Copy `configs/default.yaml` to `configs/local.yaml`. Set `dataset_root` to the separately downloaded dataset and `output_root` to a local results directory. Relative paths resolve against the YAML file. Set `preprocessing.scratch_dir` if the OS temporary drive has less than approximately 3 GiB free. Filtering/ICA also require substantial RAM and time. Review bad channels/ICA rather than assuming the starting settings suit every recording.

## Execute in stages

```bash
python -m inner_speech.cli audit --config configs/local.yaml
# Review condition counts, event sequences and eligible pairs in the audit CSVs.
python -m inner_speech.cli preprocess --config configs/local.yaml
python -m inner_speech.cli connectivity --config configs/local.yaml
python -m inner_speech.cli classify --config configs/local.yaml
# Only after all planned participants and sessions pass QC:
python -m inner_speech.cli statistics --config configs/local.yaml
python -m pytest -q
```

For a first run, explicitly select `subjects: [sub-01]` and `sessions: [ses-01]` in the local config. This is an execution check, **not** a group analysis. Group inference requires at least five unique participants by default and fails if any planned session is missing. Use a different output directory for each changed configuration; generated files within a run directory are overwritten on rerun.

Open the five notebooks after installation. They use `configs/default.yaml` unless environment variable `INNER_SPEECH_CONFIG` points to a local YAML. New notebooks contain no claimed results or saved executions.

## Original archive

The 17 original top-level `.ipynb`, `.py` and `.m` files retain their exact bytes, saved outputs, empty files and variants. Data, external weights and checkpoints are excluded. Old notebooks contain obsolete paths, possible evaluation leakage and incomplete experiments. External EEGNet inference is excluded from the new execution path. No code license has been selected; dataset licensing does not assign a license to this source code.

Primary sources: [data descriptor](https://doi.org/10.1038/s41597-022-01147-2), [authors' repository](https://github.com/N-Nieto/Inner_Speech_Dataset), [wPLI methods](https://doi.org/10.1016/j.neuroimage.2011.01.055), and [MNE-Connectivity](https://mne.tools/mne-connectivity/stable/generated/mne_connectivity.spectral_connectivity_epochs.html).

