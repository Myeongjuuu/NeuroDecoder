"""Evaluate a sparse ridge ESN on paired word trials from OpenBCI recordings."""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.signal import detrend, welch

from neurodecoder.analysis.bandpass_filters import BANDS, filter_band
from neurodecoder.data.read_openbci_csv import WORD_CODES, read_trials
from neurodecoder.models.ridge_esn import RidgeESN


def feature_sequence(eeg, sfreq, method):
    bands = [band for band in BANDS if band[1] < sfreq / 2]
    if method == "filter-bank":
        return np.concatenate([filter_band(eeg, band, sfreq) for band in bands], axis=0)
    window = round(sfreq)
    hop = max(1, round(sfreq / 4))
    if eeg.shape[-1] < window:
        raise ValueError("PSD features require word trials lasting at least one second.")
    frames = np.lib.stride_tricks.sliding_window_view(eeg, window, axis=-1)[:, ::hop]
    freq, power = welch(frames, fs=sfreq, nperseg=window)
    features = [
        power[..., (freq >= low) & (freq < high)].mean(axis=-1)
        for low, high in bands
    ]
    return np.concatenate(features, axis=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path, help="Legacy rows-by-samples OpenBCI training CSV.")
    parser.add_argument("--test-csv", type=Path, help="Separate recording for evaluation.")
    parser.add_argument("--holdout-trials", type=int, default=10, help="Final trials held out when using one CSV.")
    parser.add_argument("--sfreq", required=True, type=float, help="Recording sampling rate in Hz.")
    parser.add_argument("--features", choices=("filter-bank", "psd"), default="filter-bank")
    parser.add_argument("--units", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True, type=Path, help="Prediction CSV path.")
    args = parser.parse_args(argv)
    if args.units < 1 or not np.isfinite(args.sfreq) or args.sfreq <= 4:
        parser.error("Use a positive reservoir size and a sampling rate above 4 Hz.")
    if args.test_csv and args.test_csv.resolve() == args.csv.resolve():
        parser.error("Training and test recordings must be separate.")

    train, train_labels = read_trials(args.csv)
    if args.test_csv:
        test, test_labels = read_trials(args.test_csv)
    else:
        holdout = args.holdout_trials
        if not 1 <= holdout < len(train):
            parser.error("The holdout must leave at least one training trial.")
        train, test = train[:-holdout], train[-holdout:]
        train_labels, test_labels = train_labels[:-holdout], train_labels[-holdout:]
    classes = np.asarray(WORD_CODES)
    if not np.array_equal(np.unique(train_labels), classes):
        parser.error("Training must contain all ten word codes 2–11.")

    train = [detrend(trial) for trial in train]
    lower = np.min([trial.min(axis=-1) for trial in train], axis=0)[:, None]
    upper = np.max([trial.max(axis=-1) for trial in train], axis=0)[:, None]
    span = upper - lower
    span[span == 0] = 1

    def features(trial):
        return feature_sequence((trial - lower) / span, args.sfreq, args.features)

    train_sequences = [features(trial) for trial in train]
    esn = RidgeESN(train_sequences[0].shape[0], args.units, args.seed)
    esn.fit(train_sequences, train_labels, classes)
    predictions = np.asarray([esn.predict(features(detrend(trial))) for trial in test])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("held_out_trial", "true_code", "predicted_code"))
        writer.writerows((index, actual, predicted)
                         for index, (actual, predicted) in enumerate(zip(test_labels, predictions), 1))
    print(f"{len(train)} train / {len(test)} held out; accuracy {np.mean(test_labels == predictions):.3f}")


if __name__ == "__main__":
    main()
