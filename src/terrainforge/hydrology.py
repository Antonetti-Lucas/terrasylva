"""Lake detection, flow routing and river extraction.

Pipeline: least-cost priority flood -> water level, lakes and drainage tree ->
flow accumulation (upstream cell count) -> river mask / valley carving.
"""
from __future__ import annotations

import heapq  # noqa: F401  (numba resolves heapq natively)
from dataclasses import dataclass

import numpy as np
from numba import njit

_OX = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int64)
_OY = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int64)
_DIST = np.array([1.41421356, 1.0, 1.41421356, 1.0, 1.0, 1.41421356, 1.0, 1.41421356])


@dataclass
class Lakes:
    """Standing water found in closed basins."""

    mask: np.ndarray    # bool
    depth: np.ndarray   # float32 water depth in heightmap units, 0 outside lakes


def _as_square(heightmap: np.ndarray) -> np.ndarray:
    h = np.ascontiguousarray(heightmap, dtype=np.float64)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("heightmap must be a square 2D array")
    if h.size == 0 or not np.isfinite(h).all():
        raise ValueError("heightmap must be non-empty and finite")
    return h


@njit(cache=True)
def _tie_break(idx):
    """Deterministic integer hash, so exact flats don't drain in index order."""
    v = (idx ^ (idx >> 15)) & 0x7FFFFFFF
    v = (v * 0x2C1B3C6D) & 0x7FFFFFFF
    v ^= v >> 12
    v = (v * 0x297A2D39) & 0x7FFFFFFF
    return v ^ (v >> 15)


@njit(cache=True)
def _flood(h, ox, oy):
    """Flood inwards from the border, always visiting the lowest reachable cell.

    Cells are keyed by their *original* height, so once a spill point is passed
    the search dives to the basin floor and climbs back out along its valleys.
    The running maximum of visited heights is the water level.

    Returns (filled, parent, order): water level per cell, the neighbour each
    cell was reached from (-1 on the border), and the visiting order.
    """
    n = h.shape[0]
    filled = h.copy()
    parent = np.full(n * n, -1, dtype=np.int64)
    order = np.empty(n * n, dtype=np.int64)
    closed = np.zeros((n, n), dtype=np.bool_)
    heap = [(0.0, np.int64(0), np.int64(0))]
    heapq.heappop(heap)
    for i in range(n):
        for (y, x) in ((0, i), (n - 1, i), (i, 0), (i, n - 1)):
            if not closed[y, x]:
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
            if 0 <= ny < n and 0 <= nx < n and not closed[ny, nx]:
                closed[ny, nx] = True
                nidx = np.int64(ny * n + nx)
                parent[nidx] = idx
                heapq.heappush(heap, (h[ny, nx], _tie_break(nidx), nidx))
    return filled, parent, order


@njit(cache=True)
def _receivers(h, filled, parent, ox, oy, dist):
    """Downstream cell per cell, -1 on the border.

    Dry cells follow steepest descent on the water surface. Submerged and flat
    cells follow the flood tree to their spill point. Every step either lowers
    the surface or moves up the tree, so there can be no cycles.
    """
    n = h.shape[0]
    recv = parent.copy()
    for y in range(1, n - 1):
        for x in range(1, n - 1):
            if filled[y, x] > h[y, x]:
                continue
            best = 0.0
            for k in range(8):
                s = (filled[y, x] - filled[y + oy[k], x + ox[k]]) / dist[k]
                if s > best:
                    best = s
                    recv[y * n + x] = (y + oy[k]) * n + (x + ox[k])
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
def _keep_lakes(depth, filled, min_depth, min_area, ox, oy):
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
                if (0 <= ny < n and 0 <= nx < n and not seen[ny, nx]
                        and depth[ny, nx] > 0.0 and filled[ny, nx] == filled[y, x]):
                    seen[ny, nx] = True
                    stack[tail] = ny * n + nx
                    tail += 1
        if deepest >= min_depth and tail >= min_area:
            for i in range(tail):
                mask[stack[i] // n, stack[i] % n] = True
    return mask


def fill_depressions(heightmap: np.ndarray) -> np.ndarray:
    """Heightmap with every closed basin raised to its spill level (float64)."""
    return _flood(_as_square(heightmap), _OX, _OY)[0]


def detect_lakes(heightmap: np.ndarray, *, min_depth: float = 0.005,
                 min_area: int | None = None) -> Lakes:
    """Find lakes: closed basins filled to the level at which they spill.

    Args:
        heightmap: square height array. Border cells are outlets, so a basin
            open to the border never holds water.
        min_depth: drop lakes whose deepest point is below this (heightmap units).
        min_area: drop lakes with fewer cells; default 0.02% of the map.
    """
    if min_depth < 0:
        raise ValueError("min_depth must be >= 0")
    if min_area is not None and min_area < 0:
        raise ValueError("min_area must be >= 0")
    h = _as_square(heightmap)
    area = max(int(h.size * 0.0002), 4) if min_area is None else int(min_area)
    filled = _flood(h, _OX, _OY)[0]
    depth = filled - h
    mask = _keep_lakes(depth, filled, float(min_depth), area, _OX, _OY)
    return Lakes(mask=mask, depth=np.where(mask, depth, 0.0).astype(np.float32))


def flow_accumulation(heightmap: np.ndarray, rainfall: np.ndarray | float = 1.0) -> np.ndarray:
    """Number of upstream cells (weighted by rainfall) draining through each cell.

    Border cells are treated as outlets to the sea. Flow entering a closed basin
    follows the submerged valley floor to the basin's spill point, so rivers
    always reach the border instead of dying in sinks.
    """
    h = _as_square(heightmap)
    rain = np.broadcast_to(np.asarray(rainfall, dtype=np.float64), h.shape).copy()
    filled, parent, order = _flood(h, _OX, _OY)
    recv = _receivers(h, filled, parent, _OX, _OY, _DIST)
    return _accumulate(recv, order, rain).reshape(h.shape).astype(np.float32)


def river_mask(accumulation: np.ndarray, threshold: float | None = None,
               lakes: np.ndarray | None = None) -> np.ndarray:
    """Boolean mask of cells whose drainage area exceeds ``threshold``.

    Default threshold is 0.4% of the cell count, which gives a handful of
    major rivers plus tributaries on a typical map. The 1-cell border (the
    sea outlet) is never marked. Cells in the optional ``lakes`` mask are left
    out, so rivers stop at the shore and resume at the outlet.
    """
    if threshold is None:
        threshold = max(accumulation.size * 0.004, 2.0)
    mask = accumulation >= threshold
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
    if lakes is not None:
        mask &= ~lakes
    return mask


def carve_rivers(heightmap: np.ndarray, accumulation: np.ndarray, *,
                 depth: float = 0.02, threshold: float | None = None,
                 lakes: np.ndarray | None = None) -> np.ndarray:
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
    for axis in (0, 1):
        strength = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), axis, strength)
    if lakes is not None:
        strength = np.where(lakes, 0.0, strength)
    return (heightmap - depth * strength).astype(np.float32)
