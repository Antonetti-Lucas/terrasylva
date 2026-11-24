"""One-call generation on top of the World/Pipeline core."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .world import Continents, Hydraulic, Noise, Ocean, Pipeline, Rivers, Thermal, World


@dataclass
class Terrain:
    heightmap: np.ndarray          # float32 [0, 1]
    base: np.ndarray               # heightmap before erosion
    flow: np.ndarray               # upstream cell counts
    rivers: np.ndarray             # bool mask (stops at lake shores)
    lakes: np.ndarray | None = None       # bool mask of standing water
    lake_depth: np.ndarray | None = None  # water depth, heightmap units
    ocean: np.ndarray | None = None       # bool mask, None without an ocean
    sea_level: float | None = None        # heightmap units

    @classmethod
    def from_world(cls, world: World) -> "Terrain":
        return cls(heightmap=world["height"], base=world.layers.get("base", world["height"]),
                   flow=world["flow"], rivers=world["rivers"], lakes=world["lakes"],
                   lake_depth=world["lake_depth"], ocean=world.layers.get("ocean"),
                   sea_level=world.sea_level)


def generate(size: int = 512, *, seed: int = 0, scale: float = 3.0, octaves: int = 8,
             ridged: float = 0.5, warp: float = 0.35, erosion_drops: int | None = None,
             thermal_iterations: int = 20, carve: bool = True,
             river_threshold: float | None = None, wrap: bool = False,
             land: float | None = None) -> Terrain:
    """Generate an eroded terrain.

    Args:
        size: resolution (square). Powers of two plus one or two work well.
        seed: deterministic seed.
        scale: base feature frequency across the map (an integer if ``wrap``).
        octaves: noise detail levels.
        ridged: blend between smooth fBm (0) and ridged mountains (1).
        warp: domain warping, which breaks up the grid-aligned look.
        erosion_drops: hydraulic droplets; default is ``0.3 * size**2``.
        thermal_iterations: talus relaxation passes (0 disables).
        carve: carve valleys along extracted rivers.
        river_threshold: drainage cells needed to count as a river.
        wrap: make the terrain loop in both directions.
        land: fraction of land; adds continents and an ocean. None for neither.
    """
    steps = [Noise(scale=scale, octaves=octaves, ridged=ridged, warp=warp)]
    if land is not None:
        steps.append(Continents())
    if thermal_iterations > 0:
        steps.append(Thermal(iterations=thermal_iterations))
    if erosion_drops is None or erosion_drops > 0:
        steps.append(Hydraulic(drops=erosion_drops))
    if land is not None:
        steps.append(Ocean(land=land))
    steps.append(Rivers(carve=carve, threshold=river_threshold))
    return Terrain.from_world(Pipeline(steps).run(World(size=size, seed=seed, wrap=wrap)))
