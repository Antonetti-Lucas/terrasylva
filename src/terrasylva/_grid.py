"""Shared input checks and grid helpers."""
from __future__ import annotations

import numpy as np


def as_square(heightmap: np.ndarray) -> np.ndarray:
    """Validated float64 copy-or-view of a square, finite 2D array."""
    h = np.ascontiguousarray(heightmap, dtype=np.float64)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("heightmap must be a square 2D array")
    if h.size == 0 or not np.isfinite(h).all():
        raise ValueError("heightmap must be non-empty and finite")
    return h


def gradient(h: np.ndarray, wrap: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(d/dy, d/dx), with central differences across the edges if ``wrap``."""
    if not wrap:
        return tuple(np.gradient(h))
    return tuple(g[1:-1, 1:-1] for g in np.gradient(np.pad(h, 1, mode="wrap")))


def area_matrix(n: int, r: int) -> np.ndarray:
    """(r, n) matrix averaging ``n`` cells into ``r`` by exact area overlap."""
    s = n / r
    lo = np.arange(r)[:, None] * s
    hi = lo + s
    j = np.arange(n)[None, :]
    overlap = np.clip(np.minimum(hi, j + 1) - np.maximum(lo, j), 0.0, None)
    return overlap / s


def linear_matrix(n: int, r: int, wrap: bool) -> np.ndarray:
    """(n, r) matrix interpolating ``r`` cell centres linearly onto ``n`` cells."""
    q = (np.arange(n) + 0.5) * (r / n) - 0.5
    i0 = np.floor(q).astype(np.int64)
    f = q - i0
    m = np.zeros((n, r))
    rows = np.arange(n)
    for idx, weight in ((i0, 1.0 - f), (i0 + 1, f)):
        idx = idx % r if wrap else np.clip(idx, 0, r - 1)
        np.add.at(m, (rows, idx), weight)
    return m


def blur_matrix(n: int, sigma: float, wrap: bool) -> np.ndarray:
    """(n, n) Gaussian blur with rows summing to 1."""
    d = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    if wrap:
        d = np.minimum(d, n - d)
    m = np.exp(-0.5 * (d / max(sigma, 1e-6)) ** 2)
    return m / m.sum(axis=1, keepdims=True)
