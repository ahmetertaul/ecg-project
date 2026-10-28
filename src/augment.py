"""Data augmentation for heartbeat signals (applied to training batches only)."""

import numpy as np
from tensorflow import keras


def effective_length(X):
    """Length of each beat without the trailing zero padding."""
    nonzero = X[:, ::-1] != 0
    return X.shape[1] - nonzero.argmax(axis=1)


def add_noise(X, rng, std=0.02):
    """Add Gaussian noise to the beat part of each signal. The zero padding is left as it is."""
    mask = np.arange(X.shape[1]) < effective_length(X)[:, None]
    return X + rng.normal(0, std, X.shape).astype(X.dtype) * mask


def scale_amplitude(X, rng, low=0.9, high=1.1):
    """Multiply each signal by a random factor."""
    factors = rng.uniform(low, high, size=(len(X), 1)).astype(X.dtype)
    return X * factors


def time_shift(X, rng, max_shift=5):
    """Shift each beat by up to max_shift samples inside its own length.

    The empty side is filled with the edge value, and the trailing zero padding stays where it was,
    so a shifted beat still looks like the original data.
    """
    n, length = X.shape
    lengths = effective_length(X)[:, None]
    shifts = rng.integers(-max_shift, max_shift + 1, size=(n, 1))
    idx = np.arange(length)[None, :]
    src = np.clip(idx - shifts, 0, lengths - 1)
    out = np.take_along_axis(X, src, axis=1)
    return np.where(idx < lengths, out, 0).astype(X.dtype)


def augment_batch(X, rng, p=0.5, noise_std=0.02, scale=(0.9, 1.1), max_shift=5):
    """Apply each augmentation independently with probability p to each beat. max_shift=0 disables the time shift."""
    X = X.copy()
    transforms = [lambda x: scale_amplitude(x, rng, *scale), lambda x: add_noise(x, rng, noise_std)]
    if max_shift > 0:
        transforms.insert(0, lambda x: time_shift(x, rng, max_shift))
    for fn in transforms:
        chosen = rng.random(len(X)) < p
        if chosen.any():
            X[chosen] = fn(X[chosen])
    return X


class AugmentedBatches(keras.utils.PyDataset):
    """Shuffled training batches with fresh random augmentation in every epoch."""

    def __init__(self, X, y, batch_size=128, seed=42, **augment_kwargs):
        super().__init__()
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment_kwargs = augment_kwargs
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(len(X))

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, i):
        idx = self.order[i * self.batch_size:(i + 1) * self.batch_size]
        return augment_batch(self.X[idx], self.rng, **self.augment_kwargs), self.y[idx]

    def on_epoch_end(self):
        self.order = self.rng.permutation(len(self.X))
