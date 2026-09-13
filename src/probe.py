"""Linear probes, concept subspaces (INLP-style), random control subspaces, and linear erasure."""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


class Probe:
    """Logistic regression on standardized activations; direction() maps the weights back to raw space."""

    def __init__(self, X, y, C: float = 1.0, max_iter: int = 2000):
        X = np.asarray(X, dtype=np.float64)
        self.mean = X.mean(0)
        self.scale = X.std(0) + 1e-6
        self.clf = LogisticRegression(C=C, max_iter=max_iter)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.clf.fit((X - self.mean) / self.scale, y)

    def score(self, X, y) -> float:
        return float(self.clf.score((np.asarray(X, dtype=np.float64) - self.mean) / self.scale, y))

    def direction(self) -> np.ndarray:
        # logit = w_std . (x - m) / s  =>  w_raw = w_std / s
        w = self.clf.coef_[0] / self.scale
        return w / np.linalg.norm(w)


def inlp_basis(X, y, rank: int, C: float = 1.0, max_iter: int = 2000) -> np.ndarray:
    """Orthonormal [d, rank] basis: fit a probe, project its direction out, refit, repeat."""
    Xc = np.asarray(X, dtype=np.float64).copy()
    basis: list[np.ndarray] = []
    for _ in range(rank):
        v = Probe(Xc, y, C, max_iter).direction()
        for u in basis:
            v = v - (v @ u) * u
        v = v / np.linalg.norm(v)
        basis.append(v)
        Xc = Xc - np.outer(Xc @ v, v)
    return np.stack(basis, axis=1).astype(np.float32)


def random_basis(d: int, rank: int, rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((d, rank)))
    return q.astype(np.float32)


def erase(X, V, mu) -> np.ndarray:
    """Set the component of X inside span(V) to that of mu: X - ((X - mu) V) V^T."""
    X = np.asarray(X, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    return X - ((X - mu) @ V) @ V.T
