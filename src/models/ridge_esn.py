"""Sparse leaky reservoir with a sample-level ridge readout."""

import numpy as np
from scipy.linalg import solve
from scipy.sparse import csr_matrix


class RidgeESN:
    def __init__(self, features, units, seed, density=0.05, leak=0.3):
        rng = np.random.default_rng(seed)
        self.leak = leak
        self.input_weights = rng.uniform(-0.8, 0.8, (units, features + 1))
        self.input_weights *= rng.random(self.input_weights.shape) < density
        recurrent = rng.uniform(-1, 1, (units, units))
        recurrent *= rng.random(recurrent.shape) < density
        radius = np.max(np.abs(np.linalg.eigvals(recurrent)))
        if radius:
            recurrent *= 0.8 / radius
        self.recurrent = csr_matrix(recurrent)

    def states(self, sequence):
        inputs = np.column_stack((np.ones(sequence.shape[1]), sequence.T))
        drive = inputs @ self.input_weights.T
        states = np.empty_like(drive)
        state = np.zeros(self.recurrent.shape[0])
        for index, sample in enumerate(drive):
            state = (1 - self.leak) * state + self.leak * np.tanh(sample + self.recurrent @ state)
            states[index] = state
        return np.column_stack((inputs, states))

    def fit(self, sequences, labels, classes):
        width = 1 + sequences[0].shape[0] + self.recurrent.shape[0]
        gram = 1e-3 * np.eye(width)
        targets = np.zeros((width, len(classes)))
        label_index = {label: index for index, label in enumerate(classes)}
        for sequence, label in zip(sequences, labels):
            states = self.states(sequence)
            gram += states.T @ states
            targets[:, label_index[label]] += states.sum(axis=0)
        self.readout = solve(gram, targets, assume_a="pos")
        self.classes = classes

    def predict(self, sequence):
        scores = self.states(sequence).mean(axis=0) @ self.readout
        return self.classes[scores.argmax()]
