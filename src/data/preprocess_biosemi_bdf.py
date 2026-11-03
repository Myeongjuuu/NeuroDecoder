"""Convert BioSemi BDF recordings to cue-aligned MNE EEG epochs."""

import argparse
from pathlib import Path


EVENTS = {"up": 31, "down": 32, "right": 33, "left": 34}


def convert(source, output, inner_only, iclabel, overwrite):
    import mne

    if output.exists() and not overwrite:
        raise FileExistsError(output)
    raw = mne.io.read_raw_bdf(source, preload=True)
    events = mne.find_events(raw, shortest_event=1, mask=255)
    condition = None
    selected = []
    for event in events:
        code = event[2]
        if code in (21, 22, 23):
            condition = code
        elif code in (12, 16):
            condition = None
        elif code in EVENTS.values() and (not inner_only or condition == 22):
            selected.append(event)
    if not selected:
        raise ValueError(f"No matching cue markers in {source}.")

    import numpy as np

    references = ["EXG1", "EXG2"]
    raw.set_eeg_reference(references)
    raw.drop_channels([name for name in raw.ch_names if name.upper().startswith("EXG")])
    raw.pick("eeg")
    raw.set_montage("biosemi128")
    raw.filter(1 if iclabel else 0.5, 100)
    raw.notch_filter(50)
    raw, aligned = raw.resample(256, events=np.asarray(selected))
    epochs = mne.Epochs(raw, aligned, EVENTS, tmin=-0.5, tmax=4.0, baseline=None, preload=True)
    if iclabel:
        from mne.preprocessing import ICA
        from mne_icalabel import label_components

        ica = ICA(random_state=97, method="infomax", fit_params={"extended": True})
        ica.fit(epochs)
        labels = label_components(epochs, ica, method="iclabel")["labels"]
        ica.exclude = [i for i, label in enumerate(labels) if label not in ("brain", "other")]
        ica.apply(epochs)
    output.parent.mkdir(parents=True, exist_ok=True)
    epochs.save(output, overwrite=overwrite)
    print(f"Saved {len(epochs)} epochs to {output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bdf", required=True, nargs="+", type=Path, help="One or more BioSemi recordings.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--inner-speech-only", action="store_true")
    parser.add_argument("--iclabel", action="store_true", help="Apply extended-infomax ICA and ICLabel.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    names = [source.stem.removesuffix("_task-innerspeech_eeg") + "_eeg-epo.fif" for source in args.bdf]
    if len(set(names)) != len(names):
        parser.error("Inputs must produce distinct output filenames.")
    for source, name in zip(args.bdf, names):
        convert(source, args.output_dir / name, args.inner_speech_only, args.iclabel, args.overwrite)


if __name__ == "__main__":
    main()
