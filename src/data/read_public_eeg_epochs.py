"""Load condition-filtered, cue-aligned OpenNeuro EEG epochs."""

import mne
import numpy as np
from scipy.signal import detrend


DIRECTIONS = ("up", "down", "right", "left")
SFREQ = 256
# Times are relative to the cue marker in the official FIF.
ACTION = (0.5, 3.0)
RELAX = (3.0, 4.0)


def load_session(eeg_path, events_path, trusted_legacy_events, *, include_relax=True):
    """Join official event metadata to epochs by recording sample."""
    if trusted_legacy_events:
        events = np.load(events_path, allow_pickle=True)
    else:
        events = np.loadtxt(events_path, delimiter=",", dtype=np.int64, ndmin=2)
    if events.ndim != 2 or events.shape[1] != 4 or not np.issubdtype(events.dtype, np.integer):
        raise ValueError("Expected four integer event columns: sample,class,condition,session.")
    if np.unique(events[:, 0]).size != len(events):
        raise ValueError("Event samples must be unique.")
    if len(np.unique(events[:, 3])) != 1:
        raise ValueError("Expected metadata for exactly one session.")
    metadata = {row[0]: row[1:3] for row in events}
    epochs = mne.read_epochs(eeg_path, preload=False, verbose="error")
    order = np.argsort(epochs.events[:, 0])
    selected, labels = [], []
    for index in order:
        sample, _, code = epochs.events[index]
        if sample not in metadata:
            raise ValueError(f"No matching metadata for FIF event at sample {sample}.")
        label, condition = metadata[sample]
        if not 0 <= label < len(DIRECTIONS) or code not in (label, label + 31):
            raise ValueError(f"Direction code mismatch at sample {sample}.")
        if condition == 1:
            selected.append(index)
            labels.append(label)
    if not selected:
        raise ValueError("No inner-speech epochs found.")
    epochs = epochs[selected]
    channels = [
        name for name, kind in zip(epochs.ch_names, epochs.get_channel_types())
        if kind == "eeg" and not name.upper().startswith("EXG")
    ]
    if not channels:
        raise ValueError("No EEG channels found.")
    if not np.isclose(epochs.info["sfreq"], SFREQ) or not np.isclose(epochs.tmin, -0.5):
        raise ValueError("Expected official cue-aligned epochs at 256 Hz starting at -0.5 s.")
    end = RELAX[1] if include_relax else ACTION[1]
    if epochs.times[-1] < end - 1 / SFREQ:
        raise ValueError("Epochs do not cover the requested analysis windows.")

    def window(bounds):
        start, stop = (round((edge - epochs.tmin) * SFREQ) for edge in bounds)
        return detrend(epochs.get_data(picks=channels, tmin=epochs.times[start], tmax=epochs.times[stop - 1]))

    relax = window(RELAX) if include_relax else None
    return window(ACTION), relax, np.asarray(labels), channels
