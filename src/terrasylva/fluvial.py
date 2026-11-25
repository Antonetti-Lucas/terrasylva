"""Mountain ranges from plates, and landscapes carved by rivers.

:class:`Ranges` says where the ground rises. :class:`Fluvial` lets it rise while
rivers cut down, using the stream-power law ``dh/dt = U - K A^m S`` plus hillslope
diffusion. Valleys, ridges and drainage basins then come out of the same process
instead of rivers being drawn on top of noise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numba import njit

from ._grid import area_matrix, blur_matrix, linear_matrix
from .biomes import _distance
from .erosion import thermal_erosion
from .hydrology import _DIST, _OX, _OY, _accumulate, _flood, _outlet_mask, _receivers
from .noise import fbm, normalize
from .sediment import _label, _valley_floors, build_deltas

if TYPE_CHECKING:  # world.py imports this module, so only import it for type checks
    from .world import World

# coarsest grid the simulation starts from
_COARSEST = 96


def _smoothstep(x: np.ndarray) -> np.ndarray:
    t = np.clip(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ------------------------------------------------------------------ plates
def plate_collision(size: int, *, seed: int, plates: int, width: float,
                    wrap: bool, extent: int = 1) -> np.ndarray:
    """[0, 1] intensity of mountain building along converging plate borders.

    Plates are warped Voronoi cells with random velocities. Borders where two
    plates move towards each other fold up; others stay flat. ``width`` is the
    width of a range as a fraction of the map.
    """
    rng = np.random.default_rng(seed)
    centres = rng.random((plates, 2))
    angle = rng.random(plates) * 2.0 * np.pi
    velocity = np.stack([np.cos(angle), np.sin(angle)], axis=1)
    noise = [fbm(size, seed=int(s), scale=4.0 * extent, octaves=5, wrap=wrap).astype(np.float64) - 0.5
             for s in rng.integers(0, 2**31, 3)]
    grid = (np.arange(size) + 0.5) / size
    bend = 0.22 / extent
    y = grid[:, None] + bend * noise[0]
    x = grid[None, :] + bend * noise[1]
    first = np.full((size, size), np.inf)
    second = np.full((size, size), np.inf)
    owner = np.zeros((size, size), dtype=np.int64)
    other = np.zeros((size, size), dtype=np.int64)
    for i, (cy, cx) in enumerate(centres):
        dy, dx = y - cy, x - cx
        if wrap:
            dy, dx = dy - np.round(dy), dx - np.round(dx)
        d = np.hypot(dy, dx)
        closest = d < first
        runner_up = ~closest & (d < second)
        other = np.where(closest, owner, np.where(runner_up, i, other))
        second = np.where(closest, first, np.where(runner_up, d, second))
        owner = np.where(closest, i, owner)
        first = np.where(closest, d, first)
    gap = centres[other] - centres[owner]
    if wrap:
        gap -= np.round(gap)
    gap /= np.maximum(np.linalg.norm(gap, axis=-1, keepdims=True), 1e-9)
    closing = ((velocity[owner] - velocity[other]) * gap).sum(axis=-1)
    distance = 0.5 * (second - first)
    # uneven along the border, so ranges have passes and end gradually
    strength = np.clip(closing, 0.0, None) * (0.55 + 0.9 * (noise[2] + 0.5))
    belt = strength * np.exp(-(distance / max(width, 1e-6)) ** 2)
    # plate ownership jumps where three plates meet; soften that
    blur = blur_matrix(size, 0.2 * width * size, wrap)
    belt = blur @ belt @ blur.T
    top = belt.max()
    return (belt / top if top > 0 else belt).astype(np.float32)


@dataclass(frozen=True)
class Ranges:
    """Create ``uplift``: how fast the ground rises, in [0, 1].

    Mountain ranges follow the borders of colliding plates; ``hills`` adds
    rolling uplands elsewhere. If the world has a height, ranges also raise it
    by ``raise_land``, so that they tend to become land, and ``coast`` roughens it
    with fine noise, for bays and islands.
    """

    plates: int = 9
    width: float = 0.07
    hills: float = 0.2
    raise_land: float = 0.25
    coast: float = 0.12
    name: str = "ranges"

    def __call__(self, world: World) -> None:
        if self.plates < 2:
            raise ValueError("plates must be >= 2")
        if self.width <= 0 or self.hills < 0 or self.raise_land < 0 or self.coast < 0:
            raise ValueError("width must be > 0, hills, raise_land and coast >= 0")
        n = world.size
        belt = plate_collision(n, seed=world.seed_for(self.name), plates=int(self.plates) * world.extent,
                               width=float(self.width) / world.extent, wrap=world.wrap, extent=world.extent)
        rolling = fbm(n, seed=world.seed_for(self.name + "/hills"), scale=3.0 * world.extent, octaves=5,
                      warp=0.4, wrap=world.wrap)
        rough = fbm(n, seed=world.seed_for(self.name + "/rough"), scale=8.0 * world.extent, octaves=4,
                    warp=0.4, wrap=world.wrap)
        hills = self.hills * _smoothstep((rolling - 0.4) / 0.45)
        up = 0.015 + hills + belt ** 1.3 * (0.6 + 0.8 * rough)
        world["uplift"] = np.clip(up / max(float(up.max()), 1e-9), 0.0, 1.0).astype(np.float32)
        if "height" in world:
            fine = fbm(n, seed=world.seed_for(self.name + "/coast"), scale=7.0 * world.extent, octaves=8,
                       gain=0.6, warp=0.4, wrap=world.wrap)
            world["height"] = normalize(world["height"] + self.raise_land * belt
                                        + self.coast * (fine - 0.5))


# ------------------------------------------------------------ stream power
@njit(cache=True, nogil=True)
def _drainage(h, filled, recv, order, rain, spread, ox, oy, dist):
    """Drainage with flow split over every lower neighbour, by slope**spread.

    Splitting stops channels from locking onto the eight grid directions.
    """
    n = h.shape[0]
    acc = rain.copy().ravel()
    weight = np.empty(8)
    for i in range(order.shape[0] - 1, -1, -1):
        c = order[i]
        if recv[c] < 0:
            continue
        y = c // n
        x = c % n
        total = 0.0
        if filled[y, x] <= h[y, x]:
            for k in range(8):
                s = (filled[y, x] - filled[(y + oy[k]) % n, (x + ox[k]) % n]) / dist[k]
                weight[k] = s ** spread if s > 0.0 else 0.0
                total += weight[k]
        if total > 0.0:
            for k in range(8):
                if weight[k] > 0.0:
                    q = ((y + oy[k]) % n) * n + (x + ox[k]) % n
                    acc[q] += acc[c] * weight[k] / total
        else:
            acc[recv[c]] += acc[c]
    return acc


@njit(cache=True, nogil=True)
def _incise(h, filled, recv, order, acc, lift, hardness, k_dt, m, spread, ox, oy, dist):
    """One implicit step, downstream cells first, so any time step is stable."""
    n = h.shape[0]
    out = h.copy()
    weight = np.empty(8)
    for i in range(order.shape[0]):
        c = order[i]
        q = recv[c]
        if q < 0:
            continue
        y = c // n
        x = c % n
        raised = h[y, x] + lift[y, x]
        power = k_dt * acc[c] ** m / hardness[y, x]
        total = 0.0
        if filled[y, x] <= h[y, x]:
            for k in range(8):
                s = (filled[y, x] - filled[(y + oy[k]) % n, (x + ox[k]) % n]) / dist[k]
                weight[k] = s ** spread if s > 0.0 else 0.0
                total += weight[k]
        top = raised
        bottom = 1.0
        if total > 0.0:
            for k in range(8):
                if weight[k] > 0.0:
                    low = out[(y + oy[k]) % n, (x + ox[k]) % n]
                    if low < raised:
                        f = power * weight[k] / (total * dist[k])
                        top += f * low
                        bottom += f
        else:
            yq = q // n
            xq = q % n
            low = out[yq, xq]
            if low < raised:
                straight = yq == y or xq == x
                f = power if straight else power / 1.41421356
                top += f * low
                bottom += f
        out[y, x] = top / bottom
    return out


@njit(cache=True, nogil=True)
def _limit_slope(h, recv, order, rise):
    """Cap the climb to each cell from the one it drains to (``rise`` per cell)."""
    n = h.shape[0]
    g = h.copy().ravel()
    for i in range(order.shape[0]):
        c = order[i]
        q = recv[c]
        if q >= 0:
            straight = c // n == q // n or c % n == q % n
            top = g[q] + (rise if straight else rise * 1.41421356)
            if g[c] > top:
                g[c] = top
    return g.reshape(n, n)


def _diffuse(h: np.ndarray, rate: float, fixed: np.ndarray, wrap: bool) -> np.ndarray:
    """One step of hillslope creep. ``fixed`` cells keep their height."""
    p = np.pad(h, 1, mode="wrap" if wrap else "edge")
    lap = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4.0 * h
    diag = p[:-2, :-2] + p[2:, 2:] + p[2:, :-2] + p[:-2, 2:] - 4.0 * h
    return np.where(fixed, h, h + rate * (lap * 2.0 / 3.0 + diag / 6.0))


def _levels(size: int) -> list[int]:
    out = [size]
    while out[0] // 2 >= _COARSEST:
        out.insert(0, out[0] // 2)
    return out


def stream_power(uplift: np.ndarray, *, outlets: np.ndarray | None = None, seed: int = 0,
                 steps: int = 60, concavity: float = 0.45, smoothing: float = 0.1,
                 hardness: np.ndarray | None = None, rain: np.ndarray | None = None,
                 wrap: bool = False, spread: float = 1.5, deposition: float = 0.0,
                 base: np.ndarray | None = None) -> np.ndarray:
    """Height above the outlets of a landscape in balance between uplift and rivers.

    Args:
        uplift: square array of rising speed, >= 0.
        outlets: bool mask where water leaves; held at height 0. Without
            ``wrap`` the border is added.
        steps: time steps on the coarsest grid; finer grids use fewer.
        concavity: drainage exponent ``m``; rivers flatten downstream faster as
            it grows. Real rivers are mostly between 0.4 and 0.5.
        smoothing: hillslope creep per step, in [0, 0.25].
        hardness: resistance of the rock, around 1, > 0.
        rain: rainfall per cell, around 1.
        spread: how strongly flow concentrates on the steepest way down.
        deposition: how readily rivers drop the sediment they carry. Above 0,
            debris from fast-rising ground builds up slow-rising ground downstream.
        base: height of the outlets, 0 or below; default 0.

    Returns:
        float64 heights in arbitrary units, not below the lowest outlet.
    """
    u = np.ascontiguousarray(uplift, dtype=np.float64)
    n = u.shape[0] if u.ndim == 2 else 0
    if u.ndim != 2 or n != u.shape[1] or n < 8:
        raise ValueError("uplift must be a square 2D array of at least 8x8")
    if not np.isfinite(u).all() or u.min() < 0:
        raise ValueError("uplift must be finite and >= 0")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if not 0 <= smoothing <= 0.25:
        raise ValueError("smoothing must be in [0, 0.25]")
    if not 0 < concavity < 1 or not spread > 0:
        raise ValueError("concavity must be in (0, 1) and spread > 0")
    if not 0 <= deposition <= 2:
        raise ValueError("deposition must be in [0, 2]")
    low = np.zeros_like(u) if base is None else np.ascontiguousarray(base, dtype=np.float64)
    if low.shape != u.shape or not np.isfinite(low).all() or low.max() > 0:
        raise ValueError("base must be finite, <= 0 and match uplift")
    extra = []
    for label, a in (("hardness", hardness), ("rain", rain)):
        a = np.ones_like(u) if a is None else np.ascontiguousarray(a, dtype=np.float64)
        if a.shape != u.shape or not np.isfinite(a).all():
            raise ValueError(f"{label} must be finite and match uplift")
        extra.append(a)
    hard, wet = extra
    if hard.min() <= 0 or wet.min() < 0:
        raise ValueError("hardness must be > 0 and rain >= 0")
    sinks = _outlet_mask(u, outlets, wrap)

    rng = np.random.default_rng(seed)
    h = None
    for level, size in enumerate(_levels(n)):
        down = area_matrix(n, size)
        shrink = (lambda a: a) if size == n else (lambda a: down @ a @ down.T)
        lift = shrink(u)
        hardness_c, rain_c = shrink(hard), shrink(wet)
        sink = sinks if size == n else shrink(sinks.astype(np.float64)) > 0.5
        if size != n and base is not None:
            # narrow basins must not vanish on coarse grids
            sink = sink | (shrink(low) < 0.3 * min(float(low.min()), -1e-12))
        sink = _outlet_mask(lift, sink, wrap)
        if h is None:
            h = rng.random((size, size)) * 1e-3
        else:
            up = linear_matrix(size, h.shape[0], wrap)
            h = up @ h @ up.T
            # break the symmetry of interpolated slopes, or valleys follow the grid
            h += rng.random((size, size)) * 2e-2 * h
        bottom = low if size == n else shrink(low)
        bottom = np.where(sink, bottom, 0.0)
        deepest = float(bottom.min())
        h = np.ascontiguousarray(np.where(sink, bottom, h))
        load = None
        count = max(int(round(steps * 0.45 ** level)), 4)
        cell = 1.0 / size
        for _ in range(count):
            filled, parent, order = _flood(h, sink, bool(wrap), _OX, _OY)
            recv = _receivers(h, filled, parent, sink, _OX, _OY, _DIST)
            acc = _drainage(h, filled, recv, order, rain_c, float(spread),
                            _OX, _OY, _DIST) * cell * cell
            rise = lift * 0.05
            if load is not None:
                # deposition is limited by how fast the land upstream wears down
                rise = rise + deposition * load
            before = h
            h = _incise(h, filled, recv, order, acc, rise, hardness_c, 0.4 / cell,
                        float(concavity), float(spread), _OX, _OY, _DIST)
            h = np.ascontiguousarray(np.maximum(_diffuse(h, smoothing, sink, wrap), deepest))
            if deposition > 0:
                eroded = np.where(sink, 0.0, np.maximum(before + lift * 0.05 - h, 0.0))
                carried = _drainage(before, filled, recv, order, np.ascontiguousarray(eroded),
                                    float(spread), _OX, _OY, _DIST) * cell * cell
                load = (carried / acc).reshape(size, size)
    return h


@dataclass(frozen=True)
class Fluvial:
    """Replace land heights by a landscape that rivers carved out of ``uplift``.

    Needs ``uplift`` (see :class:`Ranges`). With an ``ocean`` layer, sea floor and
    sea level stay as they are and rivers end at the coast; run :class:`Ocean`
    first. ``steps`` is how far the landscape develops, ``smoothing`` rounds
    the hillslopes, ``rounding`` is how many passes of weathering wear down sharp
    crests, ``hardness`` is how much the resistance of the rock varies
    (soft areas become valleys), and ``lowlands`` raises plains relative to
    peaks. Ground steeper than ``max_slope`` (height per distance) is worn back.

    Sediment: ``deposition`` builds plains below mountains, ``valleys`` gives rivers
    flat floors that widen downstream, and ``deltas`` extends the coast at river
    mouths (which updates ``ocean``). 0 turns each off. A ``basin`` layer (see
    :class:`Basins`) becomes dry land below the sea, and its lowest parts the
    bool layer ``sinks``, where water ends.
    """

    steps: int = 60
    concavity: float = 0.32
    smoothing: float = 0.1
    hardness: float = 0.4
    lowlands: float = 0.3
    max_slope: float = 0.8
    rounding: int = 2
    deposition: float = 0.4
    valleys: float = 2.0
    deltas: float = 1.0
    name: str = "fluvial"

    def __call__(self, world: World) -> None:
        if "uplift" not in world:
            raise ValueError("Fluvial needs the 'uplift' layer; run Ranges first")
        if not 0 <= self.hardness <= 1 or not 0 <= self.lowlands <= 0.9 or self.max_slope <= 0 \
                or self.rounding < 0 or self.valleys < 0 or self.deltas < 0:
            raise ValueError("hardness must be in [0, 1], lowlands in [0, 0.9], max_slope > 0, "
                             "rounding, valleys and deltas >= 0")
        n = world.size
        ocean = world.layers.get("ocean")
        sea = 0.0 if world.sea_level is None or ocean is None else float(world.sea_level)
        rock = fbm(n, seed=world.seed_for(self.name + "/rock"), scale=10.0 * world.extent, octaves=5, warp=0.5,
                   wrap=world.wrap).astype(np.float64)
        hard = np.exp(2.0 * self.hardness * (2.0 * rock - 1.0))
        basin = world.layers.get("basin")
        basin = np.zeros((n, n)) if basin is None else np.asarray(basin, dtype=np.float64)
        if ocean is not None:
            basin = np.where(ocean, 0.0, basin)
        hollow = basin > 0
        ends = ocean
        if hollow.any():
            ends = hollow if ocean is None else ocean | hollow
        h = stream_power(world["uplift"], outlets=ends, deposition=self.deposition, seed=world.seed_for(self.name),
                         steps=self.steps, concavity=self.concavity, smoothing=self.smoothing,
                         hardness=hard, wrap=world.wrap)
        land = np.ones((n, n), dtype=bool) if ocean is None else ~ocean
        if not land.any():
            return
        # the few highest cells must not flatten everything else
        top = max(float(np.quantile(h[land], 0.9995)), 1e-12)
        t = h / top
        # basins are worked out as outlets at sea level; their floor is lowered below
        t = np.where(t > 0.9, 0.9 + 0.1 * np.tanh((t - 0.9) / 0.1), t)
        t = t ** (1.0 - self.lowlands)
        # land starts just above the sea, so the coast stays where it is
        step = 0.002
        land_height = sea + (1.0 - sea) * (step + (1.0 - step) * np.clip(t, 0.0, 1.0))
        old = np.clip(np.asarray(world["height"], dtype=np.float64), 0.0, 1.0) \
            if "height" in world else np.zeros((n, n))
        new = np.where(land, land_height, old)
        if hollow.any():
            new = np.where(hollow, sea - (1.0 - sea) * basin, new)
            # the surroundings sink towards the basin, so land around it lies below the sea too
            reach = 0.06 / world.extent
            near = np.exp(-_distance(hollow, bool(world.wrap)) / (reach * n))
            pull = np.where(land & ~hollow, (1.0 - sea) * float(basin.max()) * 0.8 * near, 0.0)
            if ocean is not None and ocean.any():
                coast = _distance(ocean, bool(world.wrap)) / (reach * n)
                pull = pull * _smoothstep(coast / 2.0 - 0.5)
            new = new - pull
            new = np.maximum(new, sea - (1.0 - sea) * float(basin.max()))
            # water ends in the deepest spot of each basin
            lab, count = _label(np.ascontiguousarray(hollow), bool(world.wrap), _OX, _OY)
            spots = np.zeros((n, n), dtype=bool)
            for i in range(1, count + 1):
                part = lab == i
                if part.sum() >= 9:
                    spots |= part & (basin >= 0.97 * basin[part].max())
            world["sinks"] = spots if spots.any() else hollow
        ends = ocean
        if hollow.any():
            ends = world["sinks"] if ocean is None else ocean | world["sinks"]
        sinks = _outlet_mask(new, ends, world.wrap)
        filled, parent, order = _flood(new, sinks, bool(world.wrap), _OX, _OY)
        recv = _receivers(new, filled, parent, sinks, _OX, _OY, _DIST)
        # cliffs start from the sea surface, not from the sea floor
        base = np.where(ocean, sea, new) if ocean is not None else new
        new = _limit_slope(base, recv, order, float(self.max_slope) / world.relief)
        new = thermal_erosion(new, iterations=10, talus=self.max_slope,
                              relief=world.relief, wrap=world.wrap)
        # weathering rounds crests but doesn't fill valleys
        for _ in range(int(self.rounding)):
            new = np.minimum(new, _diffuse(new, 0.15, ~land, world.wrap))
        # slides must not move the coast
        floor = np.nextafter(np.float32(sea + 1e-4), np.float32(1.0))
        floor = np.where(new < sea, new, floor) if hollow.any() else floor
        new = np.where(land, np.maximum(new, floor), old)
        if self.valleys > 0 or self.deltas > 0:
            new = self._sediment(world, new, ends, sea)
        world["height"] = new.astype(np.float32)

    def _sediment(self, world: World, h: np.ndarray, ends, sea: float) -> np.ndarray:
        n = world.size
        h = np.ascontiguousarray(h, dtype=np.float64)
        ocean = world.layers.get("ocean")
        out = _outlet_mask(h, ends, world.wrap)
        filled, parent, order = _flood(h, out, bool(world.wrap), _OX, _OY)
        recv = _receivers(h, filled, parent, out, _OX, _OY, _DIST)
        # drainage as a share of a region, so that it doesn't depend on resolution
        share = _accumulate(recv, order, np.ones((n, n))) * (world.extent / n) ** 2
        if self.valleys > 0:
            surface = np.where(ocean, sea, h) if ocean is not None else h
            unit = 1.0 - sea
            blur = blur_matrix(n, 0.012 * n / world.extent, world.wrap)
            room = np.maximum(blur @ surface @ blur.T - surface, 0.0)
            # basins below the sea must not silt up
            room = np.where(h < sea, 0.0, room).ravel()
            new, _ = _valley_floors(np.ascontiguousarray(surface), recv, order, share,
                                    np.ascontiguousarray(room), 4e-4,
                                       0.16 * self.valleys * unit, 0.03 / world.relief,
                                       0.0008 / world.relief)
            h = np.where(out, h, new)
        if self.deltas > 0 and ocean is not None and ocean.any():
            down = np.where(recv >= 0, recv, 0)
            mouth = (recv >= 0) & ocean.ravel()[down] & ~ocean.ravel() & (share > 8e-3)
            mouths = np.flatnonzero(mouth)
            mouths = mouths[np.argsort(-share[mouths], kind="stable")]
            size = 0.06 * self.deltas * share[mouths] * (n / world.extent) ** 2
            keep = size >= 4
            rough = fbm(n, seed=world.seed_for(self.name + "/delta"), scale=40.0 * world.extent,
                        octaves=3, wrap=world.wrap).astype(np.float64)
            # deep water swallows sediment, shallows fill up
            cost = 0.15 + 8.0 * np.maximum(rough - 0.35, 0.0) + 12.0 * np.maximum(sea - h, 0.0)
            h, wet = build_deltas(h, ocean, sea, mouths[keep], size[keep], cost, world.wrap)
            world["ocean"] = wet
            world["ocean_depth"] = np.where(wet, np.maximum(sea - h, 0.0), 0.0).astype(np.float32)
        return h
