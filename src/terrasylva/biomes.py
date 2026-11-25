"""Biomes: a configurable table turned into smooth weights, a biome map and a splat map.

Each :class:`Biome` says where it lives in terms of temperature, moisture and,
optionally, slope and height. The table is an ordered list of rules: a cell
belongs to the first biome whose ranges contain it, edges are soft, and a cell no
rule covers goes to the nearest biome, so an incomplete table still maps every
cell.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

import numpy as np
from numba import njit

from .noise import fbm
from .render import slope as slope_of

if TYPE_CHECKING:  # world.py imports this module, so only import it for type checks
    from .world import World

# one "unit" of distance outside a range, per property
_UNIT = {"temperature": 10.0, "moisture": 0.2, "slope": 15.0, "height": 0.15,
         "shore": 0.01}
_SPLAT = ("rock", "grass", "sand", "snow")


def _range(value, label: str):
    if value is None:
        return None
    try:
        lo, hi = value
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be None or a (low, high) pair, got {value!r}") from None
    lo = None if lo is None else float(lo)
    hi = None if hi is None else float(hi)
    for v in (lo, hi):
        if v is not None and not np.isfinite(v):
            raise ValueError(f"{label} bounds must be finite or None")
    if lo is not None and hi is not None and lo > hi:
        raise ValueError(f"{label} has low > high: {value!r}")
    return (lo, hi)


@dataclass(frozen=True)
class Biome:
    """One row of the biome table.

    Ranges are ``(low, high)`` pairs with ``None`` for an open end, or ``None``
    for "anywhere":

    * ``temperature``: deg C.
    * ``moisture``: [0, 1], as in the ``moisture`` layer plus the bonus beside
      rivers and lakes.
    * ``slope``: degrees.
    * ``height``: 0 at sea level, 1 at the highest point, down to -1 at the
      deepest sea floor.
    * ``shore``: distance to the ocean as a fraction of a region's width, 0 for the first two cells of land; 1 without an ocean.

    ``water`` marks biomes of the open ocean: they are only considered for ocean
    cells, and every other biome only for land (if the table has none of one
    kind, the other kind takes both). ``splat`` is how much of each ground
    texture (rock, grass, sand, snow) the biome uses; it is normalised.
    """

    name: str
    color: tuple[int, int, int] = (128, 128, 128)
    temperature: tuple[float | None, float | None] | None = None
    moisture: tuple[float | None, float | None] | None = None
    slope: tuple[float | None, float | None] | None = None
    height: tuple[float | None, float | None] | None = None
    shore: tuple[float | None, float | None] | None = None
    water: bool = False
    splat: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0)

    def __post_init__(self):
        # recipes come back from JSON with lists instead of tuples
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("a biome needs a non-empty name")
        try:
            color = tuple(int(c) for c in self.color)
        except (TypeError, ValueError):
            raise ValueError(f"biome {self.name!r}: color must be three integers") from None
        if len(color) != 3 or any(not 0 <= c <= 255 for c in color):
            raise ValueError(f"biome {self.name!r}: color must be three values in 0..255")
        splat = tuple(float(s) for s in self.splat)
        if len(splat) != 4 or min(splat) < 0 or not sum(splat) > 0 or not np.isfinite(splat).all():
            raise ValueError(f"biome {self.name!r}: splat needs four weights >= 0, not all 0")
        set_ = object.__setattr__
        set_(self, "color", color)
        set_(self, "splat", splat)
        set_(self, "water", bool(self.water))
        for key in _UNIT:
            set_(self, key, _range(getattr(self, key), f"biome {self.name!r}: {key}"))


def default_biome_table() -> tuple[Biome, ...]:
    """Ocean, beach, desert, savanna, grassland, forest, rainforest, taiga, tundra, snow, rock.

    Order is priority: steep ground is rock before anything else, then snow,
    and so on down to the catch-alls at the end.
    """
    B = Biome
    return (
        B("ocean", (38, 84, 138), water=True, splat=(0, 0, 1, 0)),
        B("rock", (128, 120, 112), slope=(38, None), splat=(1, 0, 0, 0)),
        B("snow", (244, 247, 250), temperature=(None, -8), splat=(0, 0, 0, 1)),
        B("beach", (222, 205, 150), height=(None, 0.04), shore=(None, 0.006), temperature=(2, None),
          splat=(0, 0, 1, 0)),
        B("desert", (224, 196, 128), temperature=(12, None), moisture=(None, 0.3),
          splat=(0.1, 0, 0.9, 0)),
        B("rainforest", (24, 104, 58), temperature=(18, None), moisture=(0.6, None),
          splat=(0, 1, 0, 0)),
        B("savanna", (190, 180, 90), temperature=(18, None), moisture=(None, 0.5),
          splat=(0, 0.75, 0.25, 0)),
        B("forest", (54, 120, 62), temperature=(5, None), moisture=(0.42, None),
          splat=(0, 1, 0, 0)),
        B("grassland", (134, 176, 88), temperature=(5, None), splat=(0, 1, 0, 0)),
        B("taiga", (72, 108, 90), temperature=(-8, 5), moisture=(0.4, None),
          splat=(0.1, 0.9, 0, 0)),
        B("tundra", (152, 158, 138), temperature=(-8, 5), splat=(0.45, 0.45, 0, 0.1)),
    )


def biome_legend(table: Sequence[Biome]) -> list[dict]:
    """Plain-data description of a table, for ``biomes.json``."""
    return [{"index": i, "name": b.name, "color": list(b.color), "water": b.water,
             "temperature": b.temperature, "moisture": b.moisture, "slope": b.slope,
             "height": b.height, "shore": b.shore, "splat": dict(zip(_SPLAT, b.splat))}
            for i, b in enumerate(table)]


# ----------------------------------------------------------------- helpers
@njit(cache=True, nogil=True)
def _distance(mask, wrap):
    """Chamfer distance in cells to the nearest True cell (about 1e9 if there is none)."""
    n = mask.shape[0]
    d = np.empty((n, n))
    for y in range(n):
        for x in range(n):
            d[y, x] = 0.0 if mask[y, x] else 1e9
    for _ in range(16 if wrap else 1):
        changed = False
        for backward in range(2):
            for a in range(n):
                for b in range(n):
                    y = a if backward == 0 else n - 1 - a
                    x = b if backward == 0 else n - 1 - b
                    for k in range(4):
                        # forward pass looks up and left, backward pass down and right
                        if k == 0:
                            oy, ox, w = -1, -1, 1.41421356
                        elif k == 1:
                            oy, ox, w = -1, 0, 1.0
                        elif k == 2:
                            oy, ox, w = -1, 1, 1.41421356
                        else:
                            oy, ox, w = 0, -1, 1.0
                        if backward:
                            oy, ox = -oy, -ox
                        ny = y + oy
                        nx = x + ox
                        if wrap:
                            ny %= n
                            nx %= n
                        elif not (0 <= ny < n and 0 <= nx < n):
                            continue
                        if d[ny, nx] + w < d[y, x]:
                            d[y, x] = d[ny, nx] + w
                            changed = True
        if not changed:
            break
    return d


def _signed_excess(values: np.ndarray, rng, unit: float) -> np.ndarray:
    """Distance outside ``rng`` in units (negative inside); -inf if it has no bounds."""
    out = np.full(values.shape, -np.inf, dtype=np.float32)
    lo, hi = rng
    if lo is not None:
        out = np.maximum(out, (lo - values) / unit)
    if hi is not None:
        out = np.maximum(out, (values - hi) / unit)
    return out


def _quantize_to_255(weights: np.ndarray) -> np.ndarray:
    """(…, k) fractions summing to 1 -> uint8 values summing to exactly 255."""
    scaled = weights.astype(np.float64) * 255.0
    base = np.floor(scaled)
    short = (255 - base.sum(axis=-1)).astype(np.int64)
    # largest remainders get the missing units
    rank = np.argsort(np.argsort(-(scaled - base), axis=-1, kind="stable"), axis=-1)
    base += rank < short[..., None]
    return base.astype(np.uint8)


@dataclass(frozen=True)
class Biomes:
    """Create ``biome``, ``biome_weights`` and ``splat`` from the climate layers.

    Needs ``temperature`` and ``moisture`` (see :class:`Climate`). Uses ``height``,
    ``ocean``, ``rivers`` and ``lakes`` when the world has them.

    * ``biome``: uint8 index into ``table``.
    * ``biome_weights``: float32 ``(size, size, len(table))``, summing to 1, smooth
      across biome borders.
    * ``splat``: uint8 RGBA of rock, grass, sand and snow, summing to exactly 255.

    Parameters:
        table: the biomes, in priority order (see :func:`default_biome_table`).
        softness: width of the transition between biomes, in units of 10 deg C,
            0.2 moisture, 15 degrees of slope or 0.15 of height.
        noise: how far seeded, looping noise nudges temperature and moisture
            before they are matched to the table (same units), so borders are not
            smooth contours. 0 turns it off.
        wet_bonus: moisture added right beside a river or lake, fading with distance.
        wet_reach: how far that bonus reaches, as a fraction of a region's width.
    """

    table: tuple = field(default_factory=default_biome_table)
    softness: float = 0.25
    noise: float = 0.35
    wet_bonus: float = 0.25
    wet_reach: float = 0.02
    name: str = "biomes"

    def __post_init__(self):
        table = tuple(b if isinstance(b, Biome) else Biome(**b) for b in self.table)
        object.__setattr__(self, "table", table)
        if not 1 <= len(table) <= 256:
            raise ValueError("the biome table needs between 1 and 256 biomes")
        names = [b.name for b in table]
        if len(set(names)) != len(names):
            raise ValueError(f"biome names must be unique, got {names}")
        for label in ("softness", "noise", "wet_bonus", "wet_reach"):
            if not np.isfinite(getattr(self, label)) or getattr(self, label) < 0:
                raise ValueError(f"{label} must be finite and >= 0")
        if self.softness == 0:
            raise ValueError("softness must be > 0")

    def __call__(self, world: World) -> None:
        for needed in ("temperature", "moisture"):
            if needed not in world:
                raise ValueError(f"Biomes needs the {needed!r} layer; run Climate first")
        n = world.size
        h = np.clip(np.asarray(world["height"], dtype=np.float64), 0.0, 1.0)
        sea = 0.0 if world.sea_level is None else float(world.sea_level)
        ocean = world.layers.get("ocean")
        ocean = np.zeros((n, n), dtype=bool) if ocean is None else np.asarray(ocean).astype(bool)

        temperature = np.asarray(world["temperature"], dtype=np.float32).copy()
        moisture = np.asarray(world["moisture"], dtype=np.float32).copy()
        if self.noise > 0:
            for arr, key in ((temperature, "temperature"), (moisture, "moisture")):
                field_ = fbm(n, seed=world.seed_for(f"{self.name}/{key}"), scale=6.0 * world.extent, octaves=2,
                             warp=0.2, wrap=world.wrap)
                arr += (2.0 * self.noise * _UNIT[key]) * (field_.astype(np.float32) - 0.5)
        fresh = np.zeros((n, n), dtype=bool)
        for key in ("rivers", "lakes"):
            if key in world:
                fresh |= np.asarray(world[key]).astype(bool)
        if self.wet_bonus > 0 and fresh.any():
            reach = max(self.wet_reach * n / world.extent, 1e-6)
            near = _distance(fresh, bool(world.wrap))
            moisture += (self.wet_bonus * np.exp(-near / reach)).astype(np.float32)

        above = (h - sea) / max(1.0 - sea, 1e-9)
        below = (h - sea) / max(sea, 1e-9)
        fields = {
            "temperature": temperature,
            "moisture": moisture,
            "slope": np.degrees(slope_of(h, relief=world.relief, wrap=world.wrap)).astype(np.float32),
            "height": np.where(h >= sea, above, below).astype(np.float32),
            "shore": np.clip((_distance(ocean, bool(world.wrap)) - 2.0) / n * world.extent, 0.0, 1.0).astype(np.float32)
            if ocean.any() else np.ones((n, n), dtype=np.float32),
        }
        weights = self._weights(fields, ocean)
        splat_rows = np.array([b.splat for b in self.table], dtype=np.float64)
        splat_rows /= splat_rows.sum(axis=1, keepdims=True)
        splat = _quantize_to_255(weights.astype(np.float64) @ splat_rows)
        world["biome_weights"] = weights
        world["biome"] = np.argmax(weights, axis=-1).astype(np.uint8)
        world["splat"] = splat

    def _weights(self, fields: dict, ocean: np.ndarray) -> np.ndarray:
        table = self.table
        n = ocean.shape[0]
        soft = self.softness
        has_water = any(b.water for b in table)
        has_land = any(not b.water for b in table)
        weights = np.zeros((n, n, len(table)), dtype=np.float32)
        left = np.ones((n, n), dtype=np.float32)
        allowed = []
        for i, b in enumerate(table):
            ok = np.where(ocean, b.water or not has_water, (not b.water) or not has_land)
            allowed.append(ok)
            member = np.ones((n, n), dtype=np.float32)
            for key, unit in _UNIT.items():
                rng = getattr(b, key)
                if rng is None:
                    continue
                d = _signed_excess(fields[key], rng, unit)
                t = np.clip(0.5 - d / soft, 0.0, 1.0)
                member *= t * t * (3.0 - 2.0 * t)
            member *= ok
            weights[..., i] = left * member
            left *= 1.0 - member
        # cells no rule fully covers: the nearest biomes share what is left
        idx = np.nonzero(left > 1e-4)
        if idx[0].size:
            sub = {k: v[idx] for k, v in fields.items()}
            score = np.full((idx[0].size, len(table)), np.inf, dtype=np.float32)
            for i, b in enumerate(table):
                total = np.zeros(idx[0].size, dtype=np.float32)
                for key, unit in _UNIT.items():
                    rng = getattr(b, key)
                    if rng is not None:
                        total += np.maximum(_signed_excess(sub[key], rng, unit), 0.0)
                score[:, i] = np.where(allowed[i][idx], total, np.inf)
            share = np.exp(-(score - score.min(axis=1, keepdims=True)) / soft)
            share /= share.sum(axis=1, keepdims=True)
            weights[idx] += left[idx][:, None] * share
        weights /= weights.sum(axis=-1, keepdims=True)
        return weights
