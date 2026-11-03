"""Reproduce directional Fisher maps and optionally evaluate a reconstructed ESN."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import sosfiltfilt

from neurodecoder.analysis.bandpass_filters import BANDS, band_filter
from neurodecoder.data.read_public_eeg_epochs import DIRECTIONS, SFREQ, load_rest_baseline, load_session
from neurodecoder.models.online_readout_esn import EchoStateNetwork


def normalize(action, baseline, calibration):
    """Fit one range per channel using calibration action EEG and the resting baseline."""
    lower = np.minimum(action[:calibration].min((0, 2)), baseline.min((0, 2)))
    upper = np.maximum(action[:calibration].max((0, 2)), baseline.max((0, 2)))
    span = upper - lower
    span[span == 0] = 1
    return (action - lower[None, :, None]) / span[None, :, None], (
        baseline - lower[None, :, None]
    ) / span[None, :, None]


def fisher_maps(action, baseline, labels):
    """Compare action band power against the shared resting baseline."""
    scores = np.empty((len(DIRECTIONS), action.shape[1], len(BANDS)))
    if np.any(np.bincount(labels, minlength=len(DIRECTIONS)) < 2):
        raise ValueError("Fisher analysis requires at least two trials per direction.")
    for band_index, band in enumerate(BANDS):
        sos = band_filter(band, SFREQ)
        action_power = (sosfiltfilt(sos, action) ** 2).mean(axis=-1)
        baseline_power = (sosfiltfilt(sos, baseline) ** 2).mean(axis=-1)
        baseline_mean = baseline_power.mean(axis=0)
        baseline_variance = baseline_power.var(axis=0)
        for label in range(len(DIRECTIONS)):
            task = action_power[labels == label]
            variance = task.var(axis=0) + baseline_variance
            gap = (task.mean(axis=0) - baseline_mean) ** 2
            scores[label, :, band_index] = gap / np.maximum(variance, 1e-12)
    return scores


def save_fisher(output, scores, channels):
    with (output / "fisher_scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("direction", "channel", "low_hz", "high_hz", "fisher_ratio"))
        for label, direction in enumerate(DIRECTIONS):
            for channel_index, channel in enumerate(channels):
                for band_index, (low, high) in enumerate(BANDS):
                    writer.writerow((direction, channel, low, high, scores[label, channel_index, band_index]))

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for axis, direction, values in zip(axes.flat, DIRECTIONS, scores):
        edges = [BANDS[0][0], *(high for _, high in BANDS)]
        plot = axis.pcolormesh(edges, np.arange(len(channels) + 1), values,
                               vmin=0, vmax=max(scores.max(), 1e-12))
        axis.set(title=direction, xlabel="Filter bank (Hz)", ylabel="EEG channel index")
    figure.colorbar(plot, ax=axes, label="Fisher ratio")
    figure.savefig(output / "fisher_maps.png", dpi=150)
    plt.close(figure)


def evaluate_esn(output, action, labels, scores, channels, calibration, top_k, seed):
    ranking = np.argsort(-scores.mean(axis=0).ravel(), kind="stable")[:top_k]
    channel_index, band_index = np.unravel_index(ranking, scores.shape[1:])
    traces = np.stack([
        sosfiltfilt(band_filter(BANDS[band], SFREQ), action[:, channel])
        for channel, band in zip(channel_index, band_index)
    ], axis=1)
    model = EchoStateNetwork(len(ranking), len(DIRECTIONS), SFREQ // 4, seed)
    confusion = np.zeros((len(DIRECTIONS), len(DIRECTIONS)), dtype=int)
    with (output / "esn_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("inner_speech_trial", "true_direction", "prediction"))
        for index, (sequence, label) in enumerate(zip(traces, labels)):
            state = model.encode(sequence)
            readout = state @ model.readout
            prediction = int(readout.argmax())
            if index >= calibration:
                confusion[label, prediction] += 1
                writer.writerow((index + 1, DIRECTIONS[label], DIRECTIONS[prediction]))
            model.update(state, label, readout)
    print(f"Chronological ESN replay: {calibration} calibration, {confusion.sum()} scored trials")
    print(f"Accuracy: {confusion.trace() / confusion.sum():.3f}")
    print("Confusion matrix (true rows, predicted columns; up, down, right, left):")
    print(confusion)
    print("Selected channel/band pairs:", [(channels[c], BANDS[b]) for c, b in zip(channel_index, band_index)])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", required=True, type=Path, help="Official cue-aligned EEG FIF")
    parser.add_argument("--events", required=True, type=Path, help="Headerless numeric CSV or official legacy DAT")
    parser.add_argument("--trusted-legacy-events", action="store_true",
                        help="Allow pickle loading for the trusted original OpenNeuro DAT")
    parser.add_argument("--baseline", required=True, type=Path, help="Resting baseline FIF from the same session")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--esn", action="store_true", help="Also run chronological ESN replay")
    parser.add_argument("--calibration-trials", type=int, help="Initial trial count for ESN evaluation; defaults to half")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be positive.")
    if args.calibration_trials is not None and not args.esn:
        parser.error("--calibration-trials requires --esn.")
    action, labels, channels = load_session(args.epochs, args.events, args.trusted_legacy_events)
    baseline = load_rest_baseline(args.baseline, channels, action.shape[-1])
    calibration = len(labels)
    if args.esn:
        calibration = args.calibration_trials if args.calibration_trials is not None else len(labels) // 2
        if not 1 <= calibration < len(labels):
            parser.error("Calibration must leave at least one trial for scoring.")
    action, baseline = normalize(action, baseline, calibration)
    scores = fisher_maps(action[:calibration], baseline, labels[:calibration])
    args.output.mkdir(parents=True, exist_ok=True)
    save_fisher(args.output, scores, channels)
    print(f"Fisher maps: {calibration} of {len(labels)} inner-speech trials, {len(channels)} EEG channels")
    print(f"Rest reference: {len(baseline)} windows of {baseline.shape[-1] / SFREQ:g} s from one recording")
    if args.esn:
        evaluate_esn(args.output, action, labels, scores, channels, calibration, args.top_k, args.seed)



if __name__ == "__main__":
    main()
