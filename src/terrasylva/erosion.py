"""Hydraulic (particle-based) and thermal (talus) erosion.

Heights are expected in [0, 1]. Internally the map is scaled to ``relief``
cell-units so that slopes are physically meaningful regardless of resolution.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from ._grid import as_square


# --------------------------------------------------------------------------
# Hydraulic erosion: simulate individual raindrops carrying sediment.
# --------------------------------------------------------------------------

@njit(cache=True, nogil=True)
def _make_brush(radius):
    n = (2 * radius + 1) ** 2
    bx = np.empty(n, dtype=np.int32)
    by = np.empty(n, dtype=np.int32)
    bw = np.empty(n, dtype=np.float64)
    k = 0
    for y in range(-radius, radius + 1):
        for x in range(-radius, radius + 1):
            d = np.sqrt(float(x * x + y * y))
            if d <= radius:
                bx[k] = x
                by[k] = y
                bw[k] = radius - d + 1e-3
                k += 1
    return bx[:k], by[:k], bw[:k]


@njit(cache=True, nogil=True)
def _hydraulic(h, n_drops, seed, inertia, capacity, min_capacity, deposition,
               erosion, evaporation, gravity, max_steps, radius,
               init_speed, init_water, wrap):
    np.random.seed(seed)
    size = h.shape[0]
    bx, by, bw = _make_brush(radius)
    margin = radius + 1
    span = size - 2 - 2 * margin
    if wrap:
        margin = 0
        span = size
    for _ in range(n_drops):
        px = margin + np.random.random() * span
        py = margin + np.random.random() * span
        dx = 0.0
        dy = 0.0
        speed = init_speed
        water = init_water
        sediment = 0.0
        for _step in range(max_steps):
            xi = int(px)
            yi = int(py)
            xj = (xi + 1) % size
            yj = (yi + 1) % size
            fx = px - xi
            fy = py - yi

            h00 = h[yi, xi]
            h10 = h[yi, xj]
            h01 = h[yj, xi]
            h11 = h[yj, xj]
            gx = (h10 - h00) * (1 - fy) + (h11 - h01) * fy
            gy = (h01 - h00) * (1 - fx) + (h11 - h10) * fx
            height = (h00 * (1 - fx) * (1 - fy) + h10 * fx * (1 - fy)
                      + h01 * (1 - fx) * fy + h11 * fx * fy)

            dx = dx * inertia - gx * (1 - inertia)
            dy = dy * inertia - gy * (1 - inertia)
            norm = np.sqrt(dx * dx + dy * dy)
            if norm < 1e-9:
                # flat spot: pick a random direction so the drop keeps moving
                ang = np.random.random() * 2.0 * np.pi
                dx = np.cos(ang)
                dy = np.sin(ang)
            else:
                dx /= norm
                dy /= norm

            nx = px + dx
            ny = py + dy
            # Stop before the border margin: erosion there would be an
            # unbounded sink (the brush is clipped), so keep edges untouched.
            if wrap:
                nx %= size
                ny %= size
                # a tiny negative value rounds up to exactly ``size``
                if nx >= size:
                    nx = 0.0
                if ny >= size:
                    ny = 0.0
            elif nx < margin or nx >= size - 1 - margin or ny < margin or ny >= size - 1 - margin:
                break

            nxi = int(nx)
            nyi = int(ny)
            nxj = (nxi + 1) % size
            nyj = (nyi + 1) % size
            nfx = nx - nxi
            nfy = ny - nyi
            new_height = (h[nyi, nxi] * (1 - nfx) * (1 - nfy)
                          + h[nyi, nxj] * nfx * (1 - nfy)
                          + h[nyj, nxi] * (1 - nfx) * nfy
                          + h[nyj, nxj] * nfx * nfy)
            dh = new_height - height

            cap = max(-dh * speed * water * capacity, min_capacity)

            if sediment > cap or dh > 0:
                # deposit: fill a pit first if moving uphill, else drop excess
                if dh > 0:
                    amount = min(dh, sediment)
                else:
                    amount = (sediment - cap) * deposition
                sediment -= amount
                h[yi, xi] += amount * (1 - fx) * (1 - fy)
                h[yi, xj] += amount * fx * (1 - fy)
                h[yj, xi] += amount * (1 - fx) * fy
                h[yj, xj] += amount * fx * fy
            else:
                # erode over a brush so we don't carve single-pixel trenches
                amount = min((cap - sediment) * erosion, -dh)
                wsum = 0.0
                for k in range(bx.shape[0]):
                    cx = xi + bx[k]
                    cy = yi + by[k]
                    if wrap or (0 <= cx < size and 0 <= cy < size):
                        wsum += bw[k]
                for k in range(bx.shape[0]):
                    cx = xi + bx[k]
                    cy = yi + by[k]
                    if wrap:
                        cx %= size
                        cy %= size
                    if 0 <= cx < size and 0 <= cy < size:
                        h[cy, cx] -= amount * bw[k] / wsum
                sediment += amount

            # downhill (dh < 0) gains speed, uphill loses it
            speed = np.sqrt(max(speed * speed - dh * gravity, 0.0))
            water *= (1 - evaporation)
            px = nx
            py = ny
            if water < 1e-3:
                break
        # Droplet is done: drop whatever it still carries so mass is conserved
        # (otherwise eroded material just vanishes and pits grow without bound).
        xi = int(px)
        yi = int(py)
        xj = (xi + 1) % size
        yj = (yi + 1) % size
        fx = px - xi
        fy = py - yi
        h[yi, xi] += sediment * (1 - fx) * (1 - fy)
        h[yi, xj] += sediment * fx * (1 - fy)
        h[yj, xi] += sediment * (1 - fx) * fy
        h[yj, xj] += sediment * fx * fy


def hydraulic_erosion(heightmap: np.ndarray, *, drops: int = 200_000, seed: int = 0,
                      relief: float | None = None, inertia: float = 0.05,
                      capacity: float = 4.0, min_capacity: float = 0.01,
                      deposition: float = 0.3, erosion: float = 0.3,
                      evaporation: float = 0.02, gravity: float = 4.0,
                      max_steps: int = 64, radius: int = 3,
                      init_speed: float = 1.0, init_water: float = 1.0,
                      wrap: bool = False) -> np.ndarray:
    """Erode a heightmap with simulated water droplets.

    Args:
        heightmap: 2D float array in [0, 1].
        drops: number of simulated droplets (more = more eroded).
        relief: height of the map's full [0, 1] range in cell units.
            Defaults to ``0.1 * size``.
        wrap: droplets cross the edges, for maps that loop in both directions.
            Without it, a border margin is left untouched.
        Remaining arguments tune the droplet physics; defaults give
        natural-looking gullies on fBm terrain.

    Returns:
        A new eroded array (the input is not modified), same shape, float32.
    """
    h = as_square(heightmap)
    if h.shape[0] < 8:
        raise ValueError("heightmap must be at least 8x8")
    if not 0 <= inertia < 1:
        raise ValueError("inertia must be in [0, 1)")
    size = h.shape[0]
    relief = 0.1 * size if relief is None else float(relief)
    h = h * relief
    _hydraulic(h, int(drops), int(seed), float(inertia), float(capacity),
               float(min_capacity), float(deposition), float(erosion),
               float(evaporation), float(gravity), int(max_steps), int(radius),
               float(init_speed), float(init_water), bool(wrap))
    return (h / relief).astype(np.float32)


# --------------------------------------------------------------------------
# Thermal erosion: material slides down slopes steeper than the talus angle.
# --------------------------------------------------------------------------

@njit(cache=True, nogil=True)
def _thermal(h, iterations, talus, rate, wrap):
    size = h.shape[0]
    delta = np.zeros_like(h)
    ox = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
    oy = np.array([-1, -1, -1, 0, 0, 1, 1, 1])
    dist = np.array([1.41421356, 1.0, 1.41421356, 1.0, 1.0, 1.41421356, 1.0, 1.41421356])
    excess = np.zeros(8)
    lo = 0 if wrap else 1
    for _ in range(iterations):
        delta[:] = 0.0
        for y in range(lo, size - lo):
            for x in range(lo, size - lo):
                total = 0.0
                biggest = 0.0
                for k in range(8):
                    e = h[y, x] - h[(y + oy[k]) % size, (x + ox[k]) % size] - talus * dist[k]
                    if e > 0:
                        excess[k] = e
                        total += e
                        if e > biggest:
                            biggest = e
                    else:
                        excess[k] = 0.0
                if total > 0:
                    moved = rate * 0.5 * biggest
                    delta[y, x] -= moved
                    for k in range(8):
                        if excess[k] > 0:
                            delta[(y + oy[k]) % size, (x + ox[k]) % size] += moved * excess[k] / total
        h += delta


def thermal_erosion(heightmap: np.ndarray, *, iterations: int = 30,
                    talus: float = 0.6, rate: float = 0.5,
                    relief: float | None = None, wrap: bool = False) -> np.ndarray:
    """Slump over-steep slopes so cliffs relax toward the angle of repose.

    Args:
        heightmap: 2D float array in [0, 1].
        iterations: relaxation passes.
        talus: max stable height difference per cell, in cell units.
            ``0.6`` is about a 31 degree slope.
        rate: fraction of excess moved per pass, in (0, 1].
        relief, wrap: see :func:`hydraulic_erosion`.
    """
    h = as_square(heightmap)
    if not 0 < rate <= 1:
        raise ValueError("rate must be in (0, 1]")
    size = h.shape[0]
    relief = 0.1 * size if relief is None else float(relief)
    h = h * relief
    _thermal(h, int(iterations), float(talus), float(rate), bool(wrap))
    return (h / relief).astype(np.float32)
