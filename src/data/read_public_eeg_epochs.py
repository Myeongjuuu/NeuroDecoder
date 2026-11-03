"""Load condition-filtered, cue-aligned OpenNeuro EEG epochs."""

import mne
import numpy as np
from scipy.signal import detrend


DIRECTIONS = ("up", "down", "right", "left")
SFREQ = 256
# Times are relative to the cue marker in the official FIF.
ACTION = (0.5, 3.0)


def load_session(eeg_path, events_path, trusted_legacy_events):
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
    start, stop = (round((edge - epochs.tmin) * SFREQ) for edge in ACTION)
    if stop > len(epochs.times):
        raise ValueError("Epochs do not cover the action window.")
    action = epochs.get_data(picks=channels)[..., start:stop]
    return detrend(action), np.asarray(labels), channels


def load_rest_baseline(path, channels, window_samples):
    """Split one resting recording into nonoverlapping action-length windows."""
    epochs = mne.read_epochs(path, preload=False, verbose="error")
    if len(epochs) != 1 or not np.isclose(epochs.info["sfreq"], SFREQ):
        raise ValueError("Expected one resting baseline epoch recorded at 256 Hz.")
    if not set(channels).issubset(epochs.ch_names):
        raise ValueError("Baseline is missing channels present in the task EEG.")
    signal = epochs.get_data(picks=channels)[0]
    count = signal.shape[-1] // window_samples
    if count < 2:
        raise ValueError("Baseline must contain at least two full action-length windows.")
    windows = signal[:, :count * window_samples].reshape(len(channels), count, window_samples)
    return detrend(windows.transpose(1, 0, 2))
