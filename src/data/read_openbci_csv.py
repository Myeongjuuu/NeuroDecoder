"""Read paired word markers from the legacy OpenBCI CSV layout."""

import numpy as np


WORD_CODES = tuple(range(2, 12))


def read_trials(path):
    table = np.genfromtxt(path, delimiter=",", skip_header=1, ndmin=2)
    if table.shape[0] < 32:
        raise ValueError(f"{path}: expected 32 rows with EEG at 1–16 and markers at 31.")
    # The original pandas export includes an index column called Unnamed: 0.
    with path.open(encoding="utf-8-sig") as stream:
        first_header = stream.readline().split(",", 1)[0].strip().strip('"')
    if first_header in ("", "Unnamed: 0"):
        table = table[:, 1:]
    eeg, markers = table[1:17], table[31]
    starts = np.flatnonzero(np.isin(markers, WORD_CODES) & np.r_[True, markers[1:] != markers[:-1]])
    if len(starts) == 0 or len(starts) % 2:
        raise ValueError(f"{path}: expected an even number of word marker pulses.")

    trials, codes = [], []
    for start, end in starts.reshape(-1, 2):
        code = int(markers[start])
        if code != markers[end]:
            raise ValueError(f"{path}: mismatched word markers at samples {start} and {end}.")
        segment = eeg[:, start:end + 1]
        if not np.isfinite(segment).all():
            raise ValueError(f"{path}: nonfinite EEG within word marker pair at sample {start}.")
        trials.append(segment)
        codes.append(code)
    return trials, np.asarray(codes)
