"""World state, swappable generation steps and JSON recipes.

A step is any callable taking a :class:`World` and updating its layers. Use the
built-in ones, plain functions, or a mix::

    world = World(size=512, seed=42, wrap=True)
    world = Pipeline([Noise(), Continents(), Hydraulic(), Ocean(), Rivers()]).run(world)
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Callable, Iterable

import numpy as np

from .biomes import Biomes
from .climate import Climate
from .erosion import hydraulic_erosion, thermal_erosion
from .fluvial import Fluvial, Ranges
from .hydrology import (carve_rivers, detect_lakes, flow_accumulation,
                        ocean_mask, river_mask)
from .noise import fbm, normalize


@dataclass
class World:
    """A square world: real-world units plus named ``(size, size)`` layers.

    Attributes:
        size: cells per side.
        seed: master seed; see :meth:`seed_for`.
        cell_size: metres per cell.
        height_range: metres represented by heights 0 and 1. Defaults to a
            relief of a tenth of a region's width.
        wrap: the world loops in both directions.
        extent: how many regions fit across the world. 1 is a single landscape;
            4 holds 4x4 regions' worth of ranges, valleys and weather, at a
            smaller scale. Raise ``size`` along with it to keep the detail.
        sea_level: height of the sea surface, or None. Set by :class:`Ocean`.
        layers: arrays by name, e.g. ``"height"`` (float32, nominally [0, 1]).
    """

    size: int = 512
    seed: int = 0
    cell_size: float = 1.0
    height_range: tuple[float, float] | None = None
    wrap: bool = False
    sea_level: float | None = None
    extent: int = 1
    layers: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        if self.size < 8:
            raise ValueError("size must be >= 8")
        if not self.cell_size > 0:
            raise ValueError("cell_size must be > 0")
        if self.extent < 1 or self.extent != int(self.extent):
            raise ValueError("extent must be a whole number >= 1")
        self.extent = int(self.extent)
        if self.height_range is None:
            self.height_range = (0.0, 0.1 * self.size * self.cell_size / self.extent)
        lo, hi = self.height_range
        if not hi > lo:
            raise ValueError("height_range must be (low, high) with high > low")
        self.height_range = (float(lo), float(hi))
        given, self.layers = self.layers, {}
        for name, layer in given.items():
            self[name] = layer

    @property
    def relief(self) -> float:
        """Full height range in cell units, as the erosion functions expect."""
        lo, hi = self.height_range
        return (hi - lo) / self.cell_size

    def seed_for(self, name: str) -> int:
        """Seed that depends only on the master seed and ``name``.

        Adding, removing or reordering steps never changes another step's seed.
        """
        digest = hashlib.sha256(f"{self.seed}:{name}".encode()).digest()
        return int.from_bytes(digest[:4], "little")

    def __getitem__(self, name: str) -> np.ndarray:
        if name not in self.layers:
            raise KeyError(f"world has no layer {name!r}; it has {sorted(self.layers)}")
        return self.layers[name]

    def __setitem__(self, name: str, layer: np.ndarray) -> None:
        layer = np.asarray(layer)
        if layer.shape[:2] != (self.size, self.size):
            raise ValueError(f"layer {name!r} has shape {layer.shape}, "
                             f"expected ({self.size}, {self.size})")
        self.layers[name] = layer

    def __contains__(self, name: str) -> bool:
        return name in self.layers

    def metres(self) -> np.ndarray:
        """Height layer in metres."""
        lo, hi = self.height_range
        return lo + self["height"] * (hi - lo)


def _blend(world: World, new: np.ndarray, mask: str | None) -> None:
    """Store eroded heights, limited by a mask and by an existing coast."""
    old = world["height"]
    ocean = world.layers.get("ocean")
    if ocean is not None and world.sea_level is not None:
        # the sea floor rests, and land may wear down to the sea but not below
        floor = np.minimum(old, np.nextafter(np.float32(world.sea_level), np.float32(2.0)))
        new = np.where(ocean, old, np.maximum(new, floor))
    if mask is not None:
        m = np.clip(world[mask].astype(np.float32), 0.0, 1.0)
        # exact where m is 0, so protected cells keep their value bit for bit
        new = np.where(m > 0, old + m * (new - old), old)
    world["height"] = new.astype(np.float32)


@dataclass(frozen=True)
class Noise:
    """Create ``height`` and ``base`` from smooth and ridged noise."""

    scale: float = 3.0
    octaves: int = 8
    ridged: float = 0.5
    warp: float = 0.35
    name: str = "noise"

    def __call__(self, world: World) -> None:
        kw = dict(scale=self.scale * world.extent, octaves=self.octaves, warp=self.warp, wrap=world.wrap)
        smooth = fbm(world.size, seed=world.seed_for(self.name + "/smooth"), **kw)
        ridge = fbm(world.size, seed=world.seed_for(self.name + "/ridged"), ridged=True, **kw)
        h = normalize((1 - self.ridged) * smooth + self.ridged * ridge)
        # shape the histogram: flatter lowlands, sharper peaks
        world["height"] = world["base"] = normalize(h ** 1.4)


@dataclass(frozen=True)
class Continents:
    """Reshape ``height`` into continents and sea basins.

    ``scale`` sets how many: 2 gives a few large continents, 5 an archipelago.
    ``strength`` is how much of the height range the continents take up.
    Mountains fade out towards the sea floor. Creates ``continents``.
    """

    scale: float = 2.0
    strength: float = 0.55
    warp: float = 0.4
    name: str = "continents"

    def __call__(self, world: World) -> None:
        if not 0 <= self.strength <= 1:
            raise ValueError("strength must be in [0, 1]")
        c = fbm(world.size, seed=world.seed_for(self.name), scale=self.scale, octaves=4,
                warp=self.warp, warp_scale=1.0, wrap=world.wrap)
        t = np.clip((c - 0.3) / 0.4, 0.0, 1.0)
        inland = t * t * (3 - 2 * t)
        detail = world["height"] * (0.25 + 0.75 * inland)
        world["continents"] = c
        world["height"] = normalize(self.strength * c + (1 - self.strength) * detail)


@dataclass(frozen=True)
class Ocean:
    """Set the sea level and create ``ocean`` and ``ocean_depth``.

    ``land`` is the fraction of cells above the sea. Basins that lie below sea
    level but aren't connected to open water, or smaller than ``min_area`` of a
    region, stay out of the ocean, so total
    land is slightly higher. ``level`` sets the height directly instead.
    """

    land: float = 0.6
    level: float | None = None
    min_area: float = 0.01
    name: str = "ocean"

    def __call__(self, world: World) -> None:
        h = np.clip(world["height"], 0.0, 1.0)
        if self.level is None:
            if not 0 < self.land < 1:
                raise ValueError("land must be between 0 and 1; leave the step out for no ocean")
            level = float(np.quantile(h, 1 - self.land))
        else:
            level = float(self.level)
        ocean = ocean_mask(h, level, min_area=self.min_area / world.extent ** 2, wrap=world.wrap)
        world.sea_level = level
        world["height"] = h
        world["ocean"] = ocean
        world["ocean_depth"] = np.where(ocean, level - h, 0.0).astype(np.float32)


@dataclass(frozen=True)
class Thermal:
    """Thermal erosion. ``mask`` names a [0, 1] layer limiting where it applies."""

    iterations: int = 20
    talus: float = 0.6
    mask: str | None = None
    name: str = "thermal"

    def __call__(self, world: World) -> None:
        _blend(world, thermal_erosion(world["height"], iterations=self.iterations,
                                      talus=self.talus, relief=world.relief,
                                      wrap=world.wrap), self.mask)


@dataclass(frozen=True)
class Hydraulic:
    """Hydraulic erosion. ``drops`` defaults to 0.3 per cell.

    ``mask`` names a [0, 1] layer limiting where it applies. Masked results are
    blended, so mass is only conserved when there is no mask. After :class:`Ocean`,
    the sea floor is left alone and whatever reaches it is lost. ``capacity`` is
    how much sediment water can carry: 4 carves deeply but leaves a grainy
    surface, about 0.5 only adds fine detail.
    """

    drops: int | None = None
    capacity: float = 4.0
    mask: str | None = None
    name: str = "hydraulic"

    def __call__(self, world: World) -> None:
        drops = int(0.3 * world.size ** 2) if self.drops is None else self.drops
        _blend(world, hydraulic_erosion(world["height"], drops=drops, capacity=self.capacity,
                                        seed=world.seed_for(self.name),
                                        relief=world.relief, wrap=world.wrap), self.mask)


@dataclass(frozen=True)
class Rivers:
    """Create ``flow``, ``rivers``, ``lakes`` and ``lake_depth``.

    Optionally carves ``height``, which is then clipped to [0, 1]. Water drains
    to the bool layer ``ocean`` if the world has one. ``threshold`` defaults to
    0.4% of the land cells, per region.

    If the world has a ``rainfall`` layer (see :class:`Climate`), ``flow`` counts
    rain instead of cells, so dry regions get fewer rivers. Rainfall averages 1
    over land, so the default threshold still means about the same. The layer is
    read as it is: after removing :class:`Climate` from a pipeline, delete it too.
    """

    carve: bool = True
    threshold: float | None = None
    name: str = "rivers"

    def __call__(self, world: World) -> None:
        ocean = world.layers.get("ocean")
        rain = world.layers.get("rainfall")
        if rain is None:
            rain = 1.0
        else:
            rain = np.asarray(rain, dtype=np.float64)
            if not (np.isfinite(rain).all() and (rain >= 0).all()):
                raise ValueError("rainfall must be finite and >= 0")
        kw = dict(outlets=ocean, wrap=world.wrap)
        land = world.size ** 2 if ocean is None else int((~ocean).sum())
        threshold = max(0.004 * land / world.extent ** 2, 2.0) if self.threshold is None else self.threshold
        before = h = np.clip(world["height"], 0.0, 1.0)
        if self.carve:
            acc = flow_accumulation(h, rain, **kw)
            still = detect_lakes(h, **kw).mask
            if ocean is not None:
                still = still | ocean
            h = carve_rivers(h, acc, threshold=threshold, lakes=still, wrap=world.wrap)
            h = np.clip(h, 0.0, 1.0)
            if world.sea_level is not None:
                # don't dig river beds below the sea they flow into
                h = np.maximum(h, np.minimum(before, world.sea_level)).astype(np.float32)
        acc = flow_accumulation(h, rain, **kw)
        lakes = detect_lakes(h, **kw)
        still = lakes.mask if ocean is None else lakes.mask | ocean
        world["height"] = h.astype(np.float32)
        world["flow"] = acc
        world["lakes"] = lakes.mask
        world["lake_depth"] = lakes.depth
        world["rivers"] = river_mask(acc, threshold, still, wrap=world.wrap)


def _step_name(step: Callable) -> str:
    return getattr(step, "name", None) or getattr(step, "__name__", type(step).__name__)


class Pipeline:
    """An ordered list of uniquely named steps."""

    def __init__(self, steps: Iterable[Callable[[World], None]]):
        self.steps = list(steps)
        self.names = [_step_name(s) for s in self.steps]
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"step names must be unique, got {self.names}")

    def run(self, world: World, start: str | None = None,
            stop: str | None = None) -> World:
        """Run steps ``start`` to ``stop`` (inclusive) and return a new World.

        To edit a result, change a layer and re-run from the step that should
        see the change, e.g. ``run(world, start="rivers")`` after sculpting.
        The input world's layers are shared, so steps must replace arrays
        instead of modifying them in place.
        """
        for n in (start, stop):
            if n is not None and n not in self.names:
                raise ValueError(f"no step {n!r}; steps are {self.names}")
        first = 0 if start is None else self.names.index(start)
        last = len(self.steps) if stop is None else self.names.index(stop) + 1
        if first >= last:
            raise ValueError("start must not come after stop")
        out = replace(world, layers=dict(world.layers))
        for step in self.steps[first:last]:
            step(out)
        return out


STEPS: dict[str, type] = {c.__name__: c for c in (Noise, Continents, Ranges, Thermal, Hydraulic, Ocean,
                                                Fluvial, Climate, Rivers, Biomes)}


def default_pipeline() -> Pipeline:
    """Continents and mountain ranges carved by rivers, with climate and biomes."""
    return Pipeline([Noise(), Continents(), Ranges(), Ocean(), Fluvial(),
                     Hydraulic(capacity=0.5), Climate(), Rivers(carve=False), Biomes()])


def classic_pipeline() -> Pipeline:
    """Faster: erosion only touches up the noise, and rivers are carved in afterwards."""
    return Pipeline([Noise(), Continents(), Thermal(), Hydraulic(), Ocean(), Climate(),
                     Rivers(), Biomes()])


def save_recipe(path: str, world: World, pipeline: Pipeline) -> None:
    """Write world settings and steps as JSON. Layers are not saved.

    Every step must be a dataclass instance; plain functions can't be stored.
    """
    steps = []
    for step, name in zip(pipeline.steps, pipeline.names):
        if not hasattr(type(step), "__dataclass_fields__"):
            raise ValueError(f"step {name!r} is not a dataclass, so it can't be saved")
        steps.append({"step": type(step).__name__, **asdict(step)})
    settings = {f.name: getattr(world, f.name) for f in fields(world) if f.name != "layers"}
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"world": settings, "steps": steps}, f, indent=2)
        f.write("\n")


def load_recipe(path: str, extra_steps: dict[str, type] | None = None) -> tuple[World, Pipeline]:
    """Read a recipe. ``extra_steps`` maps names to your own step classes."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    known = {**STEPS, **(extra_steps or {})}
    steps = []
    try:
        for entry in data["steps"]:
            args = dict(entry)
            kind = args.pop("step")
            enabled = args.pop("enabled", True)  # the editor can switch steps off
            if kind not in known:
                raise ValueError(f"unknown step {kind!r}; known steps are {sorted(known)}")
            step = known[kind](**args)
            if enabled:
                steps.append(step)
        return World(**data["world"]), Pipeline(steps)
    except (KeyError, TypeError) as e:
        raise ValueError(f"invalid recipe {path}: {e}") from e
