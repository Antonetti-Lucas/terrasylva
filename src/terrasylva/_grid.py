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
