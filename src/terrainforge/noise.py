"""Coherent noise: Perlin gradient noise with fBm, ridged and domain-warp variants.

All generators are deterministic for a given seed and return float32 arrays
normalised to [0, 1].
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange


def _permutation(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = rng.permutation(256).astype(np.int32)
    return np.concatenate([p, p])


@njit(cache=True, inline="always")
def _fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


@njit(cache=True, inline="always")
def _grad(h, x, y):
    # 8 unit-ish gradient directions
    h = h & 7
    if h == 0:
        return x + y
    elif h == 1:
        return -x + y
    elif h == 2:
        return x - y
    elif h == 3:
        return -x - y
    elif h == 4:
        return x
    elif h == 5:
        return -x
    elif h == 6:
        return y
    return -y


@njit(cache=True)
def perlin2(x, y, perm):
    """Single sample of 2D Perlin noise, roughly in [-1, 1]."""
    xi = int(np.floor(x))
    yi = int(np.floor(y))
    xf = x - xi
    yf = y - yi
    xi &= 255
    yi &= 255
    u = _fade(xf)
    v = _fade(yf)
    aa = perm[perm[xi] + yi]
    ab = perm[perm[xi] + yi + 1]
    ba = perm[perm[xi + 1] + yi]
    bb = perm[perm[xi + 1] + yi + 1]
    x1 = _grad(aa, xf, yf) * (1 - u) + _grad(ba, xf - 1, yf) * u
    x2 = _grad(ab, xf, yf - 1) * (1 - u) + _grad(bb, xf - 1, yf - 1) * u
    return x1 * (1 - v) + x2 * v


@njit(cache=True)
def _fbm_sample(x, y, perm, octaves, lacunarity, gain, ridged):
    amp = 1.0
    freq = 1.0
    total = 0.0
    norm = 0.0
    weight = 1.0
    for _ in range(octaves):
        n = perlin2(x * freq, y * freq, perm)
        if ridged:
            n = 1.0 - abs(n)
            n = n * n * weight
            weight = min(max(n * 2.0, 0.0), 1.0)
        total += n * amp
        norm += amp
        amp *= gain
        freq *= lacunarity
    return total / norm


@njit(cache=True, parallel=True)
def _fbm_grid(size, scale, perm, octaves, lacunarity, gain, ridged,
              warp, warp_scale, ox, oy):
    out = np.empty((size, size), dtype=np.float32)
    inv = scale / size
    for j in prange(size):
        for i in range(size):
            x = i * inv + ox
            y = j * inv + oy
            if warp > 0.0:
                wx = perlin2(x * warp_scale + 17.3, y * warp_scale + 3.1, perm)
                wy = perlin2(x * warp_scale - 9.7, y * warp_scale + 41.9, perm)
                x += wx * warp
                y += wy * warp
            out[j, i] = _fbm_sample(x, y, perm, octaves, lacunarity, gain, ridged)
    return out


def normalize(a: np.ndarray) -> np.ndarray:
    """Rescale to [0, 1]. A constant array becomes all zeros."""
    a = a.astype(np.float32, copy=False)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def fbm(size: int, *, seed: int = 0, scale: float = 4.0, octaves: int = 8,
        lacunarity: float = 2.0, gain: float = 0.5, ridged: bool = False,
        warp: float = 0.0, warp_scale: float = 0.4,
        offset: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Fractal Brownian motion heightfield of shape (size, size) in [0, 1].

    Args:
        size: grid resolution (square).
        seed: RNG seed for the permutation table.
        scale: number of base-frequency cells across the map.
        octaves: number of noise layers.
        lacunarity: frequency multiplier per octave.
        gain: amplitude multiplier per octave.
        ridged: use ridged multifractal (sharp mountain crests).
        warp: domain-warp strength in noise cells (0 disables).
        warp_scale: frequency of the warp field.
        offset: shift in noise space, useful for tiling chunks.
    """
    if size < 2:
        raise ValueError("size must be >= 2")
    if octaves < 1:
        raise ValueError("octaves must be >= 1")
    perm = _permutation(seed)
    h = _fbm_grid(size, float(scale), perm, int(octaves), float(lacunarity),
                  float(gain), bool(ridged), float(warp), float(warp_scale),
                  float(offset[0]), float(offset[1]))
    return normalize(h)
