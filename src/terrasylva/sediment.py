"""Where material and water end up: valley floors, deltas, sea floors, closed basins."""
from __future__ import annotations

import heapq  # noqa: F401  (numba resolves heapq natively)
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numba import njit

from ._grid import blur_matrix
from .biomes import _distance
from .hydrology import _OX, _OY, _tie_break, fill_depressions
from .noise import fbm

if TYPE_CHECKING:  # world.py imports this module, so only import it for type checks
    from .world import World


# ------------------------------------------------------------ valley floors
@njit(cache=True, nogil=True)
def _valley_floors(h, recv, order, share, room, start, lift, gentle, grade):
    """Fill the bottoms of valleys to a nearly flat floor, downstream first.

    ``share`` is each cell's drainage. Rivers (``share >= start``) are raised by
    up to ``lift * share**0.4``, reached gradually from the outlet by ``gentle`` per
    cell and never thicker than ``room`` (so rivers on plains don't end up on dykes); ground that drains into a raised river and lies
    below its floor is filled too. Floors always climb away from the outlet, by
    ``grade`` per cell at least, so no pits form.

    Returns (new heights, bool mask of filled cells).
    """
    n = h.shape[0]
    g = h.copy().ravel()
    filled = np.zeros(n * n, dtype=np.bool_)
    thick = np.zeros(n * n)
    for i in range(order.shape[0]):
        c = order[i]
        q = recv[c]
        if q < 0:
            continue
        dist = 1.0 if (c // n == q // n or c % n == q % n) else 1.41421356
        if share[c] >= start:
            # sediment gets thicker away from the outlet, towards a limit set by the river
            thick[c] = min(lift * share[c] ** 0.4, room[c], thick[q] + gentle * dist)
            level = max(g[c] + thick[c], g[q] + grade * dist)
            if level > g[c]:
                g[c] = level
                filled[c] = True
        elif filled[q]:
            level = g[q] + grade * dist
            if level > g[c]:
                g[c] = level
                filled[c] = True
    return g.reshape(n, n), filled.reshape(n, n)


# ------------------------------------------------------------------- deltas
@njit(cache=True, nogil=True)
def _grow_delta(ocean, cost, start, cells, wrap, ox, oy):
    """Ocean cells reached from ``start``, cheapest first; at most ``cells`` of them."""
    n = ocean.shape[0]
    out = np.empty(cells, dtype=np.int64)
    seen = np.zeros(n * n, dtype=np.bool_)
    heap = [(0.0, np.int64(0))]
    heapq.heappop(heap)
    heapq.heappush(heap, (0.0, np.int64(start)))
    seen[start] = True
    count = 0
    while len(heap) > 0 and count < cells:
        z, c = heapq.heappop(heap)
        y = c // n
        x = c % n
        if ocean[y, x]:
            out[count] = c
            count += 1
        for k in range(8):
            ny = y + oy[k]
            nx = x + ox[k]
            if wrap:
                ny %= n
                nx %= n
            elif not (0 <= ny < n and 0 <= nx < n):
                continue
            q = np.int64(ny * n + nx)
            if not seen[q] and ocean[ny, nx]:
                seen[q] = True
                d = 1.0 if (ny == y or nx == x) else 1.41421356
                heapq.heappush(heap, (z + d * cost[ny, nx], q))
    return out[:count]


def build_deltas(height: np.ndarray, ocean: np.ndarray, sea: float, mouths: np.ndarray,
                 cells: np.ndarray, cost: np.ndarray, wrap: bool) -> tuple[np.ndarray, np.ndarray]:
    """(height, ocean) with each river mouth extended by about ``cells`` of new land.

    Material first builds land just above the sea, then a shallow apron around it.
    """
    h, o = height.copy(), ocean.copy()
    for mouth, count in zip(mouths, cells):
        count = int(count)
        reach = _grow_delta(o, cost, int(mouth), 3 * count, bool(wrap), _OX, _OY)
        land, apron = reach[:count], reach[count:]
        # higher near the mouth, so the river keeps a way out
        h.ravel()[land] = sea + 0.004 * (1.0 - 0.9 * np.arange(len(land)) / max(len(land), 1))
        o.ravel()[land] = False
        fade = np.arange(1, len(apron) + 1) / max(len(apron), 1)
        deep = h.ravel()[apron]
        h.ravel()[apron] = np.minimum(np.maximum(deep, deep + (sea - 1e-3 - deep) * (1.0 - fade)),
                                      sea - 1e-3)
    return h, o


# ---------------------------------------------------------------- sea floor
@dataclass(frozen=True)
class SeaFloor:
    """Reshape the sea floor into shelf, slope and abyssal plain.

    Run it after :class:`Ocean`. ``shelf`` is the width of the shallow shelf and
    ``slope`` of the drop behind it, as fractions of a region. ``shelf_depth`` is
    how deep the shelf's edge lies, as a fraction of the deepest sea. ``rough``
    keeps that share of the original floor.
    """

    shelf: float = 0.04
    slope: float = 0.08
    shelf_depth: float = 0.12
    rough: float = 0.35
    name: str = "sea floor"

    def __call__(self, world: World) -> None:
        if self.shelf < 0 or self.slope <= 0 or not 0 <= self.shelf_depth <= 1 \
                or not 0 <= self.rough <= 1:
            raise ValueError("shelf >= 0, slope > 0, shelf_depth and rough in [0, 1] expected")
        if "ocean" not in world or world.sea_level is None:
            raise ValueError("SeaFloor needs an ocean; run Ocean first")
        ocean, sea, n = world["ocean"], float(world.sea_level), world.size
        if not ocean.any() or ocean.all():
            return
        away = _distance(~ocean, bool(world.wrap)) / n * world.extent
        # grid distances have creases along straight lines
        blur = blur_matrix(n, 0.02 * n / world.extent, world.wrap)
        near = np.clip(away / 0.03, 0.0, 1.0)
        away = away * (1.0 - near) + near * np.minimum(blur @ away @ blur.T, away * 1.5)
        vary = fbm(n, seed=world.seed_for(self.name), scale=3.0 * world.extent, octaves=4,
                   warp=0.4, wrap=world.wrap).astype(np.float64)
        away = away * (0.2 + 1.6 * vary)
        shelf = self.shelf_depth * np.clip(away / max(self.shelf, 1e-9), 0.0, 1.0)
        t = np.clip((away - self.shelf) / self.slope, 0.0, 1.0)
        profile = shelf + (1.0 - self.shelf_depth) * 0.85 * t * t * (3.0 - 2.0 * t)
        h = np.asarray(world["height"], dtype=np.float64)
        depth = (sea - h) / max(sea, 1e-9)
        # original detail, faded out at the coast so that the coastline stays put
        depth = profile * (1.0 - self.rough * t) + self.rough * t * np.clip(depth, 0.0, 1.0)
        floor = np.minimum(sea * (1.0 - np.clip(depth, 0.0, 1.0)), sea - 1e-4)
        new = np.where(ocean, np.maximum(floor, 0.0), h).astype(np.float32)
        world["height"] = new
        world["ocean_depth"] = np.where(ocean, np.maximum(sea - new, 0.0), 0.0).astype(np.float32)


# ------------------------------------------------------------ closed basins
@dataclass(frozen=True)
class Basins:
    """Create ``basin``: closed depressions whose floor lies below the sea.

    ``count`` basins for every region across the world are placed as far inland as possible (none on
    land too narrow for one). ``depth`` is how far they reach below the sea as a
    fraction of the height above it, ``width`` their size as a fraction of a
    region. The layer is the depth of the floor, 0 elsewhere. :class:`Fluvial`
    shapes the land around it and :class:`Rivers` ends rivers in a lake that
    evaporates as much as flows in.
    """

    count: int = 1
    depth: float = 0.2
    width: float = 0.05
    name: str = "basins"

    def __call__(self, world: World) -> None:
        if self.count < 0 or not 0 < self.depth <= 1 or self.width <= 0:
            raise ValueError("count >= 0, depth in (0, 1] and width > 0 expected")
        n, extent = world.size, world.extent
        out = np.zeros((n, n))
        ocean = world.layers.get("ocean")
        sea = 0.0 if world.sea_level is None or ocean is None else float(world.sea_level)
        far = ocean.copy() if ocean is not None else np.zeros((n, n), dtype=bool)
        if not world.wrap:
            far[0] = far[-1] = far[:, 0] = far[:, -1] = True
        rng = np.random.default_rng(world.seed_for(self.name))
        width = self.width / extent
        noise = fbm(n, seed=world.seed_for(self.name + "/shape"), scale=6.0 * extent, octaves=4,
                    gain=0.65, warp=0.4, wrap=world.wrap).astype(np.float64) - 0.5
        inland = _distance(far, bool(world.wrap)) / n if far.any() else np.ones((n, n))
        score = np.minimum(inland, 0.3 / extent) * extent + 0.25 * noise
        score[inland < 2.2 * width] = -np.inf
        if "uplift" in world:
            # rifts open beside ranges, not on top of them
            score = np.where(np.isfinite(score), score - 0.3 * world["uplift"], score)
        grid = np.arange(n) / n
        for _ in range(int(self.count) * extent):
            at = int(np.argmax(score))
            if not np.isfinite(score.ravel()[at]):
                break
            cy, cx = (at // n) / n, (at % n) / n
            dy, dx = grid[:, None] - cy, grid[None, :] - cx
            if world.wrap:
                dy, dx = dy - np.round(dy), dx - np.round(dx)
            a = rng.random() * np.pi      # rifts are longer than wide
            u, v = np.cos(a) * dy + np.sin(a) * dx, -np.sin(a) * dy + np.cos(a) * dx
            r = np.hypot(u / 1.6, v / 0.7) / width + 2.4 * noise
            size = 0.6 + 0.4 * rng.random()
            bowl = np.where(np.hypot(dy, dx) < 2.2 * width, np.clip(1.0 - r, 0.0, 1.0), 0.0)
            out = np.maximum(out, size * bowl ** 0.5)
            score[np.hypot(dy, dx) < 5.0 * width] = -np.inf
        # the floor can't reach below the bottom of the height range
        limit = min(self.depth, 0.9 * sea / max(1.0 - sea, 1e-9)) if sea > 0 else 0.0
        world["basin"] = (out * limit).astype(np.float32)


@njit(cache=True, nogil=True)
def _fill_lake(h, start, cells, inside, wrap, ox, oy):
    """Up to ``cells`` of the lowest connected ``inside`` cells around ``start``.

    Returns (cells, water level).
    """
    n = h.shape[0]
    out = np.empty(cells, dtype=np.int64)
    seen = np.zeros(n * n, dtype=np.bool_)
    heap = [(0.0, np.int64(0), np.int64(0))]
    heapq.heappop(heap)
    s = np.int64(start)
    heapq.heappush(heap, (h[s // n, s % n], _tie_break(s), s))
    seen[s] = True
    level = h[s // n, s % n]
    count = 0
    while len(heap) > 0 and count < cells:
        z, _, c = heapq.heappop(heap)
        y = c // n
        x = c % n
        level = max(level, z)
        out[count] = c
        count += 1
        for k in range(8):
            ny = y + oy[k]
            nx = x + ox[k]
            if wrap:
                ny %= n
                nx %= n
            elif not (0 <= ny < n and 0 <= nx < n):
                continue
            q = np.int64(ny * n + nx)
            if not seen[q] and inside[ny, nx]:
                seen[q] = True
                heapq.heappush(heap, (h[ny, nx], _tie_break(q), q))
    return out[:count], level


@njit(cache=True, nogil=True)
def _label(mask, wrap, ox, oy):
    """8-connected components: (labels from 1, count)."""
    n = mask.shape[0]
    lab = np.zeros((n, n), dtype=np.int64)
    stack = np.empty(n * n, dtype=np.int64)
    count = 0
    for s in range(n * n):
        if not mask[s // n, s % n] or lab[s // n, s % n]:
            continue
        count += 1
        lab[s // n, s % n] = count
        stack[0] = s
        top = 1
        while top:
            top -= 1
            y = stack[top] // n
            x = stack[top] % n
            for k in range(8):
                ny = y + oy[k]
                nx = x + ox[k]
                if wrap:
                    ny %= n
                    nx %= n
                elif not (0 <= ny < n and 0 <= nx < n):
                    continue
                if mask[ny, nx] and not lab[ny, nx]:
                    lab[ny, nx] = count
                    stack[top] = ny * n + nx
                    top += 1
    return lab, count


def terminal_lakes(height: np.ndarray, flow: np.ndarray, sinks: np.ndarray,
                   ocean: np.ndarray | None, *, evaporation: float = 25.0,
                   wrap: bool = False) -> np.ndarray:
    """Water depth of lakes without an outflow, one per connected part of ``sinks``.

    A lake grows until it evaporates what flows in: ``inflow / evaporation`` cells,
    with evaporation in units of mean rainfall, but never past its spill point.
    """
    if not evaporation > 0:
        raise ValueError("evaporation must be > 0")
    h = np.ascontiguousarray(height, dtype=np.float64)
    depth = np.zeros(h.shape)
    # ground that would be under water if the basin filled to its spill point
    closed = fill_depressions(h, outlets=ocean, wrap=wrap) > h
    lab, count = _label(np.ascontiguousarray(sinks), bool(wrap), _OX, _OY)
    for i in range(1, count + 1):
        part = lab == i
        want = int(float(flow[part].sum()) / evaporation)
        if want < 1:
            continue
        spots = np.flatnonzero(part)
        start = spots[np.argmin(h.ravel()[spots])]
        if not closed.ravel()[start]:
            continue
        cells, level = _fill_lake(h, int(start), min(want, h.size), closed, bool(wrap), _OX, _OY)
        depth.ravel()[cells] = np.maximum(depth.ravel()[cells], level - h.ravel()[cells])
    return depth.astype(np.float32)
