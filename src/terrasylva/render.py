"""Quick-look rendering: hillshade, slope and a hypsometric colour preview."""
from __future__ import annotations

import numpy as np
from PIL import Image

from ._grid import gradient


def slope(heightmap: np.ndarray, relief: float | None = None, wrap: bool = False) -> np.ndarray:
    """Slope in radians, assuming the [0, 1] range spans ``relief`` cells."""
    size = heightmap.shape[0]
    relief = 0.1 * size if relief is None else relief
    gy, gx = gradient(heightmap.astype(np.float64) * relief, wrap)
    return np.arctan(np.hypot(gx, gy))


def hillshade(heightmap: np.ndarray, azimuth: float = 315.0, altitude: float = 40.0,
              relief: float | None = None, wrap: bool = False) -> np.ndarray:
    """Lambertian hillshade in [0, 1]. Azimuth is degrees clockwise from north."""
    size = heightmap.shape[0]
    relief = 0.1 * size if relief is None else relief
    gy, gx = gradient(heightmap.astype(np.float64) * relief, wrap)
    az = np.radians(360.0 - azimuth + 90.0)
    alt = np.radians(altitude)
    slp = np.arctan(np.hypot(gx, gy))
    asp = np.arctan2(gy, -gx)
    shade = np.sin(alt) * np.cos(slp) + np.cos(alt) * np.sin(slp) * np.cos(az - asp)
    return np.clip(shade, 0.0, 1.0).astype(np.float32)


# (height above sea, rgb) stops: grass -> forest -> rock -> snow
_LAND = [
    (0.00, (128, 162, 92)),
    (0.30, (70, 110, 60)),
    (0.55, (120, 108, 96)),
    (0.78, (160, 156, 150)),
    (0.90, (245, 245, 248)),
    (1.00, (255, 255, 255)),
]
_BEACH = (201, 190, 140)
# (depth, rgb) stops, depth as a fraction of the height range
_WATER = [
    (0.00, (96, 160, 190)),
    (0.04, (66, 128, 172)),
    (0.20, (38, 84, 138)),
    (0.60, (22, 48, 92)),
]


def _ramp(values: np.ndarray, stops: list) -> np.ndarray:
    xs = [s[0] for s in stops]
    return np.stack([np.interp(values, xs, [s[1][c] for s in stops]) for c in range(3)], axis=-1)


def colorize(heightmap: np.ndarray, rivers: np.ndarray | None = None,
             sea_level: float | None = None,
             lakes: np.ndarray | None = None, relief: float | None = None,
             wrap: bool = False, *, ocean: np.ndarray | None = None,
             lake_depth: np.ndarray | None = None) -> np.ndarray:
    """Hypsometric colouring with hillshade, as uint8 RGB.

    Only actual water is blue: ``ocean``, ``lakes`` and ``rivers`` masks. With a
    ``sea_level`` but no ``ocean``, everything below that level is sea.
    """
    h = np.clip(np.asarray(heightmap, dtype=np.float64), 0.0, 1.0)
    sea = 0.0 if sea_level is None else float(sea_level)
    if ocean is None and sea_level is not None:
        ocean = h < sea
    above = np.clip((h - sea) / max(1.0 - sea, 1e-9), 0.0, 1.0)
    rgb = _ramp(above, _LAND)
    if sea_level is not None:
        beach = np.clip(1.0 - above / 0.025, 0.0, 1.0)[..., None]
        rgb = rgb * (1 - beach) + np.array(_BEACH) * beach
    shade = hillshade(heightmap, relief=relief, wrap=wrap)
    rgb = rgb * (0.35 + 0.75 * shade[..., None])
    # water surfaces are flat, so they get depth colour instead of hillshade
    if ocean is not None:
        rgb[ocean] = _ramp(np.maximum(sea - h, 0.0)[ocean], _WATER)
    if lakes is not None:
        depth = np.full(h.shape, 0.1) if lake_depth is None else lake_depth
        rgb[lakes] = _ramp(depth[lakes], _WATER)
    if rivers is not None:
        rgb[rivers] = rgb[rivers] * 0.3 + np.array([50, 110, 200]) * 0.7
    return np.clip(rgb, 0, 255).astype(np.uint8)


def save_preview(path: str, heightmap: np.ndarray, rivers: np.ndarray | None = None,
                 mode: str = "color", lakes: np.ndarray | None = None,
                 relief: float | None = None, wrap: bool = False,
                 sea_level: float | None = None, ocean: np.ndarray | None = None,
                 lake_depth: np.ndarray | None = None) -> None:
    """Write a PNG preview. ``mode`` is ``color``, ``hillshade`` or ``height``."""
    if mode == "color":
        img = Image.fromarray(colorize(heightmap, rivers, sea_level, lakes, relief, wrap,
                                       ocean=ocean, lake_depth=lake_depth))
    elif mode == "hillshade":
        img = Image.fromarray((hillshade(heightmap, relief=relief, wrap=wrap) * 255).astype(np.uint8))
    elif mode == "height":
        img = Image.fromarray((np.clip(heightmap, 0, 1) * 255).astype(np.uint8))
    else:
        raise ValueError(f"unknown mode {mode!r}")
    img.save(path)
