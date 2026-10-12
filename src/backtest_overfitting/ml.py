"""A small machine learning strategy: logistic regression trained by gradient descent.

Features are the last LAGS daily returns. The model predicts the
probability that tomorrow's return is positive; the strategy is long when
that probability is above one half. Written from scratch in numpy so every
step is visible: the gradient of the log-loss tells each weight which way
to move to reduce the error.
"""

import numpy as np

LAGS = 5


def features(returns: np.ndarray, lags: int = LAGS) -> tuple[np.ndarray, np.ndarray]:
    """Row t holds returns t-lags..t-1; the label is whether return t was positive."""
    X = np.column_stack([returns[lags - k - 1 : len(returns) - k - 1] for k in range(lags)])
    y = (returns[lags:] > 0).astype(float)
    return X, y


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class LogisticRegression:
    def __init__(self, learning_rate: float = 0.1, epochs: int = 2000):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        # Standardise using training data only, so the test set leaks nothing.
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-12
        Xs = (X - self.mean_) / self.std_
        self.w_ = np.zeros(Xs.shape[1])
        self.b_ = 0.0
        n = len(y)
        for _ in range(self.epochs):
            error = _sigmoid(Xs @ self.w_ + self.b_) - y  # derivative of log-loss
            self.w_ -= self.learning_rate * (Xs.T @ error) / n
            self.b_ -= self.learning_rate * error.mean()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(((X - self.mean_) / self.std_) @ self.w_ + self.b_)


def fit_and_position(train: np.ndarray, test: np.ndarray, lags: int = LAGS):
    """Fit on `train`; return (model, accuracy on train, train positions, test positions).

    Positions line up with train[lags:] and with all of `test`. The test
    features use the last `lags` training days as history, so the first test
    day is predictable, and the model is never refit on test data.
    """
    X_tr, y_tr = features(train, lags)
    model = LogisticRegression().fit(X_tr, y_tr)
    in_pos = (model.predict_proba(X_tr) > 0.5).astype(float)
    X_te, _ = features(np.concatenate([train[-lags:], test]), lags)
    out_pos = (model.predict_proba(X_te) > 0.5).astype(float)
    accuracy = float(np.mean(in_pos == y_tr))
    return model, accuracy, in_pos, out_pos
