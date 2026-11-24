"""Lake detection, flow routing and river extraction.

Pipeline: least-cost priority flood -> water level, lakes and drainage tree ->
flow accumulation (upstream cell count) -> river mask / valley carving.
"""
from __future__ import annotations

import heapq  # noqa: F401  (numba resolves heapq natively)
from dataclasses import dataclass

import numpy as np
from numba import njit

from ._grid import as_square

_OX = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int64)
_OY = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int64)
_DIST = np.array([1.41421356, 1.0, 1.41421356, 1.0, 1.0, 1.41421356, 1.0, 1.41421356])


@dataclass
class Lakes:
    """Standing water found in closed basins."""

    mask: np.ndarray    # bool
    depth: np.ndarray   # float32 water depth in heightmap units, 0 outside lakes


def _outlet_mask(h, outlets, wrap):
    """Cells where water leaves: ``outlets``, plus the border unless wrapping."""
    n = h.shape[0]
    mask = np.zeros((n, n), dtype=np.bool_)
    if outlets is not None:
        outlets = np.asarray(outlets)
        if outlets.shape != h.shape or outlets.dtype != bool:
            raise ValueError("outlets must be a bool array with the heightmap's shape")
        mask |= outlets
    if not wrap:
        mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = True
    elif not mask.any():
        # water has to end somewhere: use the lowest point of the world
        mask.ravel()[h.argmin()] = True
    return mask


@njit(cache=True)
def _tie_break(idx):
    """Deterministic integer hash, so exact flats don't drain in index order."""
    v = (idx ^ (idx >> 15)) & 0x7FFFFFFF
    v = (v * 0x2C1B3C6D) & 0x7FFFFFFF
    v ^= v >> 12
    v = (v * 0x297A2D39) & 0x7FFFFFFF
    return v ^ (v >> 15)


@njit(cache=True)
def _flood(h, outlets, wrap, ox, oy):
    """Flood upwards from the outlets, always visiting the lowest reachable cell.

    Cells are keyed by their *original* height, so once a spill point is passed
    the search dives to the basin floor and climbs back out along its valleys.
    The running maximum of visited heights is the water level.

    Returns (filled, parent, order): water level per cell, the neighbour each
    cell was reached from (-1 at outlets), and the visiting order.
    """
    n = h.shape[0]
    filled = h.copy()
    parent = np.full(n * n, -1, dtype=np.int64)
    order = np.empty(n * n, dtype=np.int64)
    closed = np.zeros((n, n), dtype=np.bool_)
    heap = [(0.0, np.int64(0), np.int64(0))]
    heapq.heappop(heap)
    for y in range(n):
        for x in range(n):
            if outlets[y, x]:
                closed[y, x] = True
                idx = np.int64(y * n + x)
                heapq.heappush(heap, (h[y, x], _tie_break(idx), idx))
    level = -np.inf
    count = 0
    while len(heap) > 0:
        z, _, idx = heapq.heappop(heap)
        y = idx // n
        x = idx % n
        level = max(level, z)
        filled[y, x] = level
        order[count] = idx
        count += 1
        for k in range(8):
            ny = y + oy[k]
            nx = x + ox[k]
            if wrap:
                ny %= n
                nx %= n
            if 0 <= ny < n and 0 <= nx < n and not closed[ny, nx]:
                closed[ny, nx] = True
                nidx = np.int64(ny * n + nx)
                parent[nidx] = idx
                heapq.heappush(heap, (h[ny, nx], _tie_break(nidx), nidx))
    return filled, parent, order


@njit(cache=True)
def _receivers(h, filled, parent, outlets, ox, oy, dist):
    """Downstream cell per cell, -1 at outlets.

    Dry cells follow steepest descent on the water surface. Submerged and flat
    cells follow the flood tree to their spill point. Every step either lowers
    the surface or moves up the tree, so there can be no cycles.
    """
    n = h.shape[0]
    recv = parent.copy()
    for y in range(n):
        for x in range(n):
            if outlets[y, x] or filled[y, x] > h[y, x]:
                continue
            best = 0.0
            for k in range(8):
                # without wrapping the border is all outlets, so % is a no-op
                ny = (y + oy[k]) % n
                nx = (x + ox[k]) % n
                s = (filled[y, x] - filled[ny, nx]) / dist[k]
                if s > best:
                    best = s
                    recv[y * n + x] = ny * n + nx
    return recv


@njit(cache=True)
def _accumulate(recv, order, rain):
    # a receiver is always visited by the flood before its donors
    acc = rain.copy().ravel()
    for i in range(order.shape[0] - 1, -1, -1):
        r = recv[order[i]]
        if r >= 0:
            acc[r] += acc[order[i]]
    return acc


@njit(cache=True)
def _keep_lakes(depth, filled, min_depth, min_area, wrap, ox, oy):
    """8-connected submerged regions, minus those too shallow or too small."""
    n = depth.shape[0]
    mask = np.zeros((n, n), dtype=np.bool_)
    seen = np.zeros((n, n), dtype=np.bool_)
    stack = np.empty(n * n, dtype=np.int64)
    for start in range(n * n):
        sy = start // n
        sx = start % n
        if seen[sy, sx] or depth[sy, sx] <= 0.0:
            continue
        # cells stay in stack[0:tail] so the region can be written back
        stack[0] = start
        seen[sy, sx] = True
        head = 0
        tail = 1
        deepest = 0.0
        while head < tail:
            y = stack[head] // n
            x = stack[head] % n
            head += 1
            deepest = max(deepest, depth[y, x])
            for k in range(8):
                ny = y + oy[k]
                nx = x + ox[k]
                if wrap:
                    ny %= n
                    nx %= n
                if (0 <= ny < n and 0 <= nx < n and not seen[ny, nx]
                        and depth[ny, nx] > 0.0 and filled[ny, nx] == filled[y, x]):
                    seen[ny, nx] = True
                    stack[tail] = ny * n + nx
                    tail += 1
        if deepest >= min_depth and tail >= min_area:
            for i in range(tail):
                mask[stack[i] // n, stack[i] % n] = True
    return mask


def fill_depressions(heightmap: np.ndarray, *, outlets: np.ndarray | None = None,
                     wrap: bool = False) -> np.ndarray:
    """Heightmap with every closed basin raised to its spill level (float64)."""
    h = as_square(heightmap)
    return _flood(h, _outlet_mask(h, outlets, wrap), bool(wrap), _OX, _OY)[0]


def detect_lakes(heightmap: np.ndarray, *, min_depth: float = 0.005,
                 min_area: int | None = None, outlets: np.ndarray | None = None,
                 wrap: bool = False) -> Lakes:
    """Find lakes: closed basins filled to the level at which they spill.

    Args:
        heightmap: square height array.
        min_depth: drop lakes whose deepest point is below this (heightmap units).
        min_area: drop lakes with fewer cells; default 0.02% of the map.
        outlets, wrap: see :func:`flow_accumulation`. A basin open to an outlet
            never holds water.
    """
    if min_depth < 0:
        raise ValueError("min_depth must be >= 0")
    if min_area is not None and min_area < 0:
        raise ValueError("min_area must be >= 0")
    h = as_square(heightmap)
    area = max(int(h.size * 0.0002), 4) if min_area is None else int(min_area)
    filled = fill_depressions(h, outlets=outlets, wrap=wrap)
    depth = filled - h
    mask = _keep_lakes(depth, filled, float(min_depth), area, bool(wrap), _OX, _OY)
    return Lakes(mask=mask, depth=np.where(mask, depth, 0.0).astype(np.float32))


def ocean_mask(heightmap: np.ndarray, sea_level: float, *, min_area: float = 0.01,
               wrap: bool = False) -> np.ndarray:
    """Open water: connected cells below ``sea_level`` covering enough area.

    ``min_area`` is a fraction of the map. Smaller basins are left out, so they
    can become lakes. If none is large enough, the largest one is the ocean.
    """
    if not np.isfinite(sea_level):
        raise ValueError("sea_level must be finite")
    if not 0 <= min_area <= 1:
        raise ValueError("min_area must be in [0, 1]")
    h = as_square(heightmap)
    depth = np.maximum(sea_level - h, 0.0)
    if not depth.any():
        return np.zeros(h.shape, dtype=bool)
    same = np.zeros_like(h)
    cells = max(int(min_area * h.size), 1)
    while True:
        mask = _keep_lakes(depth, same, 0.0, cells, bool(wrap), _OX, _OY)
        if mask.any():
            return mask
        cells = max(int(cells * 0.9), 1)


def flow_accumulation(heightmap: np.ndarray, rainfall: np.ndarray | float = 1.0, *,
                      outlets: np.ndarray | None = None, wrap: bool = False) -> np.ndarray:
    """Number of upstream cells (weighted by rainfall) draining through each cell.

    Water leaves at the bool mask ``outlets`` (e.g. an ocean) and, unless ``wrap``,
    at the border. A looping world without outlets drains to its lowest cell.
    Flow entering a closed basin
    follows the submerged valley floor to the basin's spill point, so rivers
    always reach an outlet instead of dying in sinks.
    """
    h = as_square(heightmap)
    rain = np.broadcast_to(np.asarray(rainfall, dtype=np.float64), h.shape).copy()
    out = _outlet_mask(h, outlets, wrap)
    filled, parent, order = _flood(h, out, bool(wrap), _OX, _OY)
    recv = _receivers(h, filled, parent, out, _OX, _OY, _DIST)
    return _accumulate(recv, order, rain).reshape(h.shape).astype(np.float32)


def river_mask(accumulation: np.ndarray, threshold: float | None = None,
               lakes: np.ndarray | None = None, *, wrap: bool = False) -> np.ndarray:
    """Boolean mask of cells whose drainage area exceeds ``threshold``.

    Default threshold is 0.4% of the cell count, which gives a handful of
    major rivers plus tributaries on a typical map. The 1-cell border (the
    sea outlet) is never marked unless ``wrap``. Cells in the optional ``lakes`` mask are left
    out, so rivers stop at the shore and resume at the outlet.
    """
    if threshold is None:
        threshold = max(accumulation.size * 0.004, 2.0)
    mask = accumulation >= threshold
    if not wrap:
        mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
    if lakes is not None:
        mask &= ~lakes
    return mask


def carve_rivers(heightmap: np.ndarray, accumulation: np.ndarray, *,
                 depth: float = 0.02, threshold: float | None = None,
                 lakes: np.ndarray | None = None, wrap: bool = False) -> np.ndarray:
    """Lower the terrain along rivers, deeper where flow is larger.

    ``depth`` is the carve depth for the largest river, in heightmap units.
    Beds inside the optional ``lakes`` mask are left alone.
    """
    thr = max(accumulation.size * 0.004, 2.0) if threshold is None else threshold
    a = np.maximum(accumulation, 1.0)
    strength = np.clip(np.log(a / thr) / np.log(a.max() / thr + 1e-9), 0.0, 1.0)
    strength = np.where(accumulation >= thr, strength, 0.0).astype(np.float32)
    # soften so valleys, not 1-px slits, are produced
    k = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0
    if wrap:
        strength = np.pad(strength, 2, mode="wrap")
    for axis in (0, 1):
        strength = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), axis, strength)
    if wrap:
        strength = strength[2:-2, 2:-2]
    if lakes is not None:
        strength = np.where(lakes, 0.0, strength)
    return (heightmap - depth * strength).astype(np.float32)
