"""Fixed leaky reservoir with a recursive least-squares readout."""

import numpy as np


class EchoStateNetwork:
    """Fixed leaky reservoir with a recursive least-squares readout."""

    def __init__(self, inputs, classes, washout, seed):
        self.washout = washout
        self.targets = np.eye(classes)
        rng = np.random.default_rng(seed)
        units = 32
        self.input_weights = rng.normal(0, 0.5 / np.sqrt(inputs), (units, inputs))
        recurrent = rng.normal(size=(units, units))
        self.recurrent = 0.9 * recurrent / max(abs(np.linalg.eigvals(recurrent)))
        self.readout = np.zeros((units + 1, classes))
        self.inverse_covariance = 10 * np.eye(units + 1)

    def encode(self, sequence):
        state = np.zeros(self.recurrent.shape[0])
        mean = np.zeros_like(state)
        washout = self.washout
        for index, sample in enumerate(sequence.T):
            state = 0.7 * state + 0.3 * np.tanh(self.input_weights @ sample + self.recurrent @ state)
            if index >= washout:
                mean += state
        return np.r_[1, mean / (sequence.shape[1] - washout)]

    def update(self, state, label, scores):
        projection = self.inverse_covariance @ state
        gain = projection / (1 + state @ projection)
        target = self.targets[label]
        self.readout += np.outer(gain, target - scores)
        self.inverse_covariance -= np.outer(gain, projection)
