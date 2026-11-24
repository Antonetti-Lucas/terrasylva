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


@njit(cache=True, nogil=True, inline="always")
def _fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


@njit(cache=True, nogil=True, inline="always")
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


@njit(cache=True, nogil=True)
def perlin2(x, y, perm, period=0):
    """Single sample of 2D Perlin noise, roughly in [-1, 1].

    Repeats every ``period`` units in x and y if it is positive.
    """
    xi = int(np.floor(x))
    yi = int(np.floor(y))
    xf = x - xi
    yf = y - yi
    xj = xi + 1
    yj = yi + 1
    if period > 0:
        xi %= period
        yi %= period
        xj %= period
        yj %= period
    xi &= 255
    yi &= 255
    xj &= 255
    yj &= 255
    u = _fade(xf)
    v = _fade(yf)
    aa = perm[perm[xi] + yi]
    ab = perm[perm[xi] + yj]
    ba = perm[perm[xj] + yi]
    bb = perm[perm[xj] + yj]
    x1 = _grad(aa, xf, yf) * (1 - u) + _grad(ba, xf - 1, yf) * u
    x2 = _grad(ab, xf, yf - 1) * (1 - u) + _grad(bb, xf - 1, yf - 1) * u
    return x1 * (1 - v) + x2 * v


@njit(cache=True, nogil=True)
def _fbm_sample(x, y, perm, octaves, lacunarity, gain, ridged, period):
    amp = 1.0
    freq = 1.0
    total = 0.0
    norm = 0.0
    weight = 1.0
    for o in range(octaves):
        # shift each octave so their lattices don't line up into straight features
        n = perlin2(x * freq + 19.37 * o, y * freq + 7.73 * o, perm, period)
        if ridged:
            n = 1.0 - abs(n)
            n = n * n * weight
            weight = min(max(n * 2.0, 0.0), 1.0)
        total += n * amp
        norm += amp
        amp *= gain
        freq *= lacunarity
        period = int(period * lacunarity)
    return total / norm


@njit(cache=True, nogil=True, parallel=True)
def _fbm_grid(size, scale, perm, octaves, lacunarity, gain, ridged,
              warp, warp_scale, ox, oy, period, warp_period):
    out = np.empty((size, size), dtype=np.float32)
    inv = scale / size
    for j in prange(size):
        for i in range(size):
            x = i * inv + ox
            y = j * inv + oy
            if warp > 0.0:
                wx = perlin2(x * warp_scale + 17.3, y * warp_scale + 3.1, perm, warp_period)
                wy = perlin2(x * warp_scale - 9.7, y * warp_scale + 41.9, perm, warp_period)
                x += wx * warp
                y += wy * warp
            out[j, i] = _fbm_sample(x, y, perm, octaves, lacunarity, gain, ridged,
                                    period)
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
        offset: tuple[float, float] = (0.0, 0.0),
        wrap: bool = False) -> np.ndarray:
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
        offset: shift in noise space. Each map is normalized on its own, so
            shifted maps don't share border heights.
        wrap: make the result loop in both directions. ``scale`` and
            ``lacunarity`` must then be whole numbers, and ``warp_scale`` is
            rounded so the warp field loops too.
    """
    if size < 2:
        raise ValueError("size must be >= 2")
    if octaves < 1:
        raise ValueError("octaves must be >= 1")
    period = warp_period = 0
    if wrap:
        if scale < 1 or scale != int(scale) or lacunarity < 1 or lacunarity != int(lacunarity):
            raise ValueError("wrap needs scale and lacunarity to be whole numbers >= 1")
        period = int(scale)
        warp_period = max(1, round(scale * warp_scale))
        warp_scale = warp_period / scale
    perm = _permutation(seed)
    h = _fbm_grid(size, float(scale), perm, int(octaves), float(lacunarity),
                  float(gain), bool(ridged), float(warp), float(warp_scale),
                  float(offset[0]), float(offset[1]), period, warp_period)
    return normalize(h)
