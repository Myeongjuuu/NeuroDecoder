"""Select an ESN on training sessions and evaluate a separate session."""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, solve
from scipy.signal import welch

from neurodecoder.analysis.bandpass_filters import filter_band
from neurodecoder.data.read_public_eeg_epochs import DIRECTIONS, SFREQ, load_session


RESERVOIRS = ((32, 0.3, 0.6), (64, 0.5, 0.9), (128, 0.7, 0.9))
POWER_BANDS = ((4, 8), (8, 13), (13, 20), (20, 30), (30, 45))


def sequences(action, feature):
    # Start after the first half-second of the action to limit cue carryover.
    eeg = action[..., SFREQ // 2:]
    eeg = eeg - eeg.mean(axis=1, keepdims=True)
    starts = range(0, eeg.shape[-1] - SFREQ + 1, SFREQ // 2)
    if feature == "band-power":
        frames = np.stack([eeg[..., i:i + SFREQ] for i in starts], axis=1)
        frequencies, spectrum = welch(frames, fs=SFREQ, nperseg=SFREQ)
        power = np.stack([spectrum[..., (frequencies >= lo) & (frequencies < hi)].mean(-1)
                          for lo, hi in POWER_BANDS], axis=-1)
        log_power = np.log(np.maximum(power, np.finfo(float).tiny))
        centered = log_power - log_power.mean(axis=2, keepdims=True)
        return centered.reshape(*centered.shape[:2], -1)

    channels = eeg.shape[1]
    upper = np.triu_indices(channels)
    identity = np.eye(channels) / channels
    bands = []
    for band in ((4, 13), (8, 30), (13, 30), (30, 45)):
        filtered = filter_band(eeg, band, SFREQ)
        frames = np.stack([filtered[..., i:i + SFREQ] for i in starts], axis=1)
        cov = frames @ frames.swapaxes(-1, -2)
        trace = np.trace(cov, axis1=-2, axis2=-1)
        cov = 0.9 * cov / np.maximum(trace[..., None, None], np.finfo(float).tiny) + 0.1 * identity
        logs = []
        for matrix in cov.reshape(-1, channels, channels):
            values, vectors = eigh(matrix)
            logs.append(((vectors * np.log(values)) @ vectors.T)[upper])
        bands.append(np.asarray(logs).reshape(len(eeg), len(starts), -1))
    return np.concatenate(bands, axis=-1)


def fit_projection(features):
    # Use an eigendecomposition in sample space because there are few trials.
    flat = features.reshape(-1, features.shape[-1])
    mean = flat.mean(axis=0)
    centered = flat - mean
    first = max(0, len(flat) - 24)
    values, vectors = eigh(centered @ centered.T, subset_by_index=(first, len(flat) - 1))
    usable = values > max(values[-1], 1) * 1e-10
    if not np.any(usable):
        raise ValueError("Calibration features have no measurable variation.")
    singular = np.sqrt(values[usable])
    components = (centered.T @ vectors[:, usable]) / singular
    return mean, components, singular / np.sqrt(len(flat) - 1)


def project(features, projection):
    mean, components, scale = projection
    return (features - mean) @ components / scale


def reservoir_states(data, config):
    units, leak, radius = config
    rng = np.random.default_rng(42)
    inputs = rng.normal(0, 0.8 / np.sqrt(data.shape[-1]), size=(data.shape[-1], units))
    recurrent = rng.normal(size=(units, units))
    recurrent *= radius / np.max(np.abs(np.linalg.eigvals(recurrent)))
    state = np.zeros((len(data), units))
    accumulated = np.zeros_like(state)
    for sample in data.swapaxes(0, 1):
        state = (1 - leak) * state + leak * np.tanh(sample @ inputs + state @ recurrent)
        accumulated += state
    return accumulated / data.shape[1]


def fit_readout(states, labels, penalty=0.1):
    states = np.column_stack((np.ones(len(states)), states))
    gram = states.T @ states + penalty * np.eye(states.shape[1])
    return solve(gram, states.T @ np.eye(len(DIRECTIONS))[labels], assume_a="pos")


def predict(states, readout):
    design = np.column_stack((np.ones(len(states)), states))
    return (design @ readout).argmax(axis=1)


def balanced_accuracy(labels, predicted):
    return np.mean([np.mean(predicted[labels == label] == label) for label in range(len(DIRECTIONS))])


def select_settings(features, labels, groups):
    best = None
    for name, data in features.items():
        scores = [[] for _ in RESERVOIRS]
        for session in np.unique(groups):
            train = groups != session
            projection = fit_projection(data[train])
            projected = project(data, projection)
            for index, config in enumerate(RESERVOIRS):
                states = reservoir_states(projected, config)
                guesses = predict(states[~train], fit_readout(states[train], labels[train]))
                scores[index].append(balanced_accuracy(labels[~train], guesses))
        for config, fold_scores in zip(RESERVOIRS, scores):
            score = float(np.mean(fold_scores))
            if best is None or score > best[0]:
                best = (score, name, config)
    return best


def select_readout_penalty(features, labels, groups, config):
    penalties = (0.1, 0.01, 1.0, 10.0)
    scores = np.zeros(len(penalties))
    for session in np.unique(groups):
        fit = groups != session
        projection = fit_projection(features[fit])
        states = reservoir_states(project(features, projection), config)
        for index, penalty in enumerate(penalties):
            readout = fit_readout(states[fit], labels[fit], penalty)
            guesses = predict(states[~fit], readout)
            scores[index] += balanced_accuracy(labels[~fit], guesses)
    return penalties[int(scores.argmax())]


def windows(signal):
    return np.lib.stride_tricks.sliding_window_view(signal, SFREQ, axis=-1)[..., ::SFREQ // 2, :]


def log_power(signal):
    power = (windows(signal) ** 2).mean(axis=-1)
    return np.log(np.maximum(power, np.finfo(float).tiny)).swapaxes(1, 2)


def csp_bands(action):
    signal = action[..., SFREQ // 2:]
    signal = signal - signal.mean(axis=1, keepdims=True)
    return [filter_band(signal, band, SFREQ) for band in ((4, 8), (8, 13), (13, 30), (30, 45))]


def csp_input(train, labels, apply):
    features = []
    for data, evaluation in zip(train, apply):
        covariance = data @ data.swapaxes(-1, -2)
        trace = np.trace(covariance, axis1=-1, axis2=-2)
        covariance /= np.maximum(trace[:, None, None], np.finfo(float).tiny)
        shrink = 0.15 * np.eye(data.shape[1]) / data.shape[1]
        spatial_filters = []
        for label in range(len(DIRECTIONS)):
            selected = labels == label
            within = 0.85 * covariance[selected].mean(axis=0) + shrink
            other = 0.85 * covariance[~selected].mean(axis=0) + shrink
            _, vectors = eigh(within, within + other)
            spatial_filters.extend(vectors[:, [0, 1, -2, -1]].T)
        features.append(log_power(np.asarray(spatial_filters) @ evaluation))
    return np.concatenate(features, axis=-1)


def rank_directions(train, labels):
    values = train.mean(axis=1)
    overall = values.mean(axis=0)
    between = np.zeros(values.shape[1])
    within = np.zeros(values.shape[1])
    for label in range(len(DIRECTIONS)):
        group = values[labels == label]
        center = group.mean(axis=0)
        between += len(group) * (center - overall) ** 2
        within += ((group - center) ** 2).sum(axis=0)
    return np.argsort(-between / np.maximum(within, 1e-12), kind="stable")


def score_test(train, labels, test, config):
    projection = fit_projection(train)
    fit_states = reservoir_states(project(train, projection), config)
    test_states = reservoir_states(project(test, projection), config)
    return predict(test_states, fit_readout(fit_states, labels))


def compare_csp(action, labels, groups, test):
    config = (64, 0.5, 0.9)
    candidates = (8, 16)
    training_bands = csp_bands(action)
    cv_scores = {count: [] for count in candidates}
    for session in np.unique(groups):
        train = groups != session
        features = csp_input([band[train] for band in training_bands], labels[train], training_bands)
        order = rank_directions(features[train], labels[train])
        for count in candidates:
            selected = features[..., order[:count]]
            guesses = score_test(selected[train], labels[train], selected[~train], config)
            cv_scores[count].append(balanced_accuracy(labels[~train], guesses))
    best_count = max(candidates, key=lambda count: np.mean(cv_scores[count]))
    combined_bands = [np.concatenate((fit_band, test_band))
                      for fit_band, test_band in zip(training_bands, csp_bands(test))]
    combined = csp_input(training_bands, labels, combined_bands)
    fit_features, test_features = combined[:len(labels)], combined[len(labels):]
    selected = rank_directions(fit_features, labels)[:best_count]
    return {
        "filter-bank-csp": score_test(fit_features, labels, test_features, config),
        f"filter-bank-csp-fisher-top-{best_count}": score_test(
            fit_features[..., selected], labels, test_features[..., selected], config
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", nargs=2, type=Path, action="append", required=True, metavar=("FIF", "EVENTS"))
    parser.add_argument("--test", nargs=2, type=Path, required=True, metavar=("FIF", "EVENTS"))
    parser.add_argument("--trusted-legacy-events", action="store_true")
    parser.add_argument("--tuned-power", action="store_true",
                        help="Use band power and select the ridge readout penalty on training sessions.")
    parser.add_argument("--compare", action="store_true",
                        help="Save predictions from the existing ESN and both CSP variants.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if len(args.train) < 2:
        parser.error("Supply at least two training sessions.")
    pairs = [*args.train, args.test]
    for column in range(2):
        for i, pair in enumerate(pairs):
            if any(pair[column].samefile(earlier[column]) for earlier in pairs[:i]):
                parser.error("Use distinct files for each session.")

    actions, targets, groups = [], [], []
    expected_channels = None

    def read(pair):
        action, labels, channels = load_session(*pair, args.trusted_legacy_events)
        if expected_channels is not None and channels != expected_channels:
            parser.error("Sessions must use the same EEG channels in the same order.")
        if np.unique(labels).size != len(DIRECTIONS):
            parser.error("Every session must contain all four directions.")
        return action, labels, channels

    for index, pair in enumerate(args.train):
        action, labels, expected_channels = read(pair)
        actions.append(action)
        targets.append(labels)
        groups.extend([index] * len(labels))

    combined = np.concatenate(actions)
    labels = np.concatenate(targets)
    groups = np.asarray(groups)
    test, true, _ = read(args.test)

    def decode(tuned_power):
        candidates = ("band-power",) if tuned_power else ("band-power", "spatial-covariance")
        training = {name: sequences(combined, name) for name in candidates}
        score, name, config = select_settings(training, labels, groups)
        features = training[name]
        projection = fit_projection(features)
        states = reservoir_states(project(features, projection), config)
        penalty = select_readout_penalty(features, labels, groups, config) if tuned_power else 0.1
        readout = fit_readout(states, labels, penalty)
        test_states = reservoir_states(project(sequences(test, name), projection), config)
        return predict(test_states, readout), score, name, config, penalty

    predicted, score, name, config, penalty = decode(args.tuned_power)
    if args.compare:
        predictions = {"public-esn": decode(False)[0] if args.tuned_power else predicted}
        if args.tuned_power:
            predictions["tuned-power-esn"] = predicted
        predictions.update(compare_csp(combined, labels, groups, test))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("method", "inner_speech_trial", "true_direction", "prediction"))
            for method, guesses in predictions.items():
                writer.writerows((method, index, DIRECTIONS[actual], DIRECTIONS[guess])
                                 for index, (actual, guess) in enumerate(zip(true, guesses), 1))
        print(f"Comparison predictions saved to {args.output}")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("inner_speech_trial", "true_direction", "prediction"))
        writer.writerows((index, DIRECTIONS[actual], DIRECTIONS[guess])
                         for index, (actual, guess) in enumerate(zip(true, predicted), 1))
    print(f"Selected {name}, reservoir (units, leak, radius) {config}; session CV {score:.3f}")
    if args.tuned_power:
        print(f"Selected ridge readout penalty: {penalty:g}")
    print(f"Held-out accuracy: {np.sum(true == predicted)}/{len(true)} ({np.mean(true == predicted):.1%})")
    print(f"Held-out balanced accuracy: {balanced_accuracy(true, predicted):.1%}")
