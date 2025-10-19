import numpy as np

class EEGScaler:
    """Standardize EEG data per electrode (channel)."""
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray):
        # X: (samples, freq, electrodes, 1)
        self.mean_ = X.mean(axis=(0, 1), keepdims=True)
        self.scale_ = X.std(axis=(0, 1), keepdims=True) + 1e-8
        return self

    def transform(self, X: np.ndarray):
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray):
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: np.ndarray):
        return X_scaled * self.scale_ + self.mean_