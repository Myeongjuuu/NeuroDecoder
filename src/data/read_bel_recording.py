"""Read BEL EGI recordings and align sample-indexed word markers."""

import csv
import re

import mne
import numpy as np


WORD_CODES = tuple(range(2, 12))


def read_trials(recording, markers_path, montage_path, marker_offset, initial_rest):
    raw = mne.io.read_raw_egi(recording, preload=False)
    aliases = {}
    for name, kind in zip(raw.ch_names, raw.get_channel_types()):
        if kind != "eeg":
            continue
        number = re.fullmatch(r"E?(\d+)(?:\s+.*)?", name.strip(), re.IGNORECASE)
        if number:
            aliases[name] = f"E{int(number[1])}"
        elif name.casefold() == "cz":
            aliases[name] = "Cz"
    raw.rename_channels({name: alias for name, alias in aliases.items() if name != alias})
    channels = list(aliases.values())
    if not channels:
        raise ValueError("No BEL EEG channel names matched E1–E130 or Cz.")
    raw.pick(channels)
    raw.set_montage(mne.channels.read_custom_montage(montage_path), match_case=False)
    sfreq = raw.info["sfreq"]
    if sfreq <= 4:
        raise ValueError("Recording sampling rate must exceed 4 Hz.")

    with markers_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if "Data" not in (reader.fieldnames or ()):
            raise ValueError(f"{markers_path}: expected a sample-aligned Data column.")
        markers = np.asarray([float(row["Data"]) for row in reader])
    if not markers.size or not np.isfinite(markers).all():
        raise ValueError(f"{markers_path}: expected finite marker values.")

    pulses = np.flatnonzero(np.isin(markers, WORD_CODES) & np.r_[True, markers[1:] != markers[:-1]])
    if not len(pulses) or len(pulses) % 2:
        raise ValueError(f"{markers_path}: expected paired word markers.")
    trials, labels = [], []
    for start, end in pulses.reshape(-1, 2):
        code = int(markers[start])
        if code != markers[end]:
            raise ValueError(f"{markers_path}: mismatched markers at samples {start} and {end}.")
        first, stop = start + marker_offset, end + marker_offset + 1
        if first < round(initial_rest * sfreq) or stop > raw.n_times:
            raise ValueError("Word markers fall outside the recording or inside the initial rest.")
        segment = raw.get_data(picks=channels, start=first, stop=stop)
        if not np.isfinite(segment).all():
            raise ValueError(f"{recording}: nonfinite EEG within a word trial.")
        trials.append(segment - segment.mean(axis=0, keepdims=True))
        labels.append(code)
    return trials, np.asarray(labels), sfreq, channels
