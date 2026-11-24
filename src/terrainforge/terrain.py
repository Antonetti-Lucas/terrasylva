"""High-level pipeline tying noise, erosion and hydrology together."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .erosion import hydraulic_erosion, thermal_erosion
from .hydrology import carve_rivers, detect_lakes, flow_accumulation, river_mask
from .noise import fbm, normalize


@dataclass
class Terrain:
    heightmap: np.ndarray          # float32 [0, 1]
    base: np.ndarray               # heightmap before erosion
    flow: np.ndarray               # upstream cell counts
    rivers: np.ndarray             # bool mask (stops at lake shores)
    lakes: np.ndarray | None = None       # bool mask of standing water
    lake_depth: np.ndarray | None = None  # water depth, heightmap units


def generate(size: int = 512, *, seed: int = 0, scale: float = 3.0, octaves: int = 8,
             ridged: float = 0.5, warp: float = 0.35, erosion_drops: int | None = None,
             thermal_iterations: int = 20, carve: bool = True,
             river_threshold: float | None = None) -> Terrain:
    """Generate an eroded terrain.

    Args:
        size: resolution (square). Powers of two plus one or two work well.
        seed: deterministic seed.
        scale: base feature frequency across the map.
        octaves: noise detail levels.
        ridged: blend between smooth fBm (0) and ridged mountains (1).
        warp: domain warping, which breaks up the grid-aligned look.
        erosion_drops: hydraulic droplets; default is ``0.3 * size**2``.
        thermal_iterations: talus relaxation passes (0 disables).
        carve: carve valleys along extracted rivers.
        river_threshold: drainage cells needed to count as a river.
    """
    smooth = fbm(size, seed=seed, scale=scale, octaves=octaves, warp=warp)
    ridge = fbm(size, seed=seed + 1, scale=scale, octaves=octaves, ridged=True, warp=warp)
    base = normalize((1 - ridged) * smooth + ridged * ridge)
    # shape the histogram: flatter lowlands, sharper peaks
    base = normalize(base ** 1.4)

    h = base
    if thermal_iterations > 0:
        h = thermal_erosion(h, iterations=thermal_iterations)
    drops = int(0.3 * size * size) if erosion_drops is None else erosion_drops
    if drops > 0:
        h = hydraulic_erosion(h, drops=drops, seed=seed)
    h = normalize(h)

    acc = flow_accumulation(h)
    if carve:
        h = normalize(carve_rivers(h, acc, threshold=river_threshold,
                                   lakes=detect_lakes(h).mask))
        acc = flow_accumulation(h)
    lakes = detect_lakes(h)
    rivers = river_mask(acc, river_threshold, lakes.mask)
    return Terrain(heightmap=h, base=base, flow=acc, rivers=rivers,
                   lakes=lakes.mask, lake_depth=lakes.depth)
