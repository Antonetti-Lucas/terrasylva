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


# (height, rgb) stops: sea -> beach -> grass -> forest -> rock -> snow
_RAMP = [
    (0.00, (38, 84, 124)),
    (0.18, (66, 122, 160)),
    (0.20, (201, 190, 140)),
    (0.26, (110, 150, 80)),
    (0.45, (70, 110, 60)),
    (0.65, (120, 108, 96)),
    (0.82, (160, 156, 150)),
    (0.92, (245, 245, 248)),
    (1.00, (255, 255, 255)),
]


def colorize(heightmap: np.ndarray, rivers: np.ndarray | None = None,
             sea_level: float | None = None,
             lakes: np.ndarray | None = None, relief: float | None = None,
             wrap: bool = False) -> np.ndarray:
    """Hypsometric colouring with hillshade, as uint8 RGB. Optional river/lake overlays."""
    h = np.clip(heightmap, 0.0, 1.0)
    xs = np.array([s[0] for s in _RAMP])
    cols = np.array([s[1] for s in _RAMP], dtype=np.float64)
    rgb = np.stack([np.interp(h, xs, cols[:, c]) for c in range(3)], axis=-1)
    shade = hillshade(heightmap, relief=relief, wrap=wrap)
    rgb = rgb * (0.35 + 0.75 * shade[..., None])
    if lakes is not None:
        rgb[lakes] = (45, 100, 180)  # flat water surface, so no hillshade
    if rivers is not None:
        rgb[rivers] = rgb[rivers] * 0.3 + np.array([50, 110, 200]) * 0.7
    return np.clip(rgb, 0, 255).astype(np.uint8)


def save_preview(path: str, heightmap: np.ndarray, rivers: np.ndarray | None = None,
                 mode: str = "color", lakes: np.ndarray | None = None,
                 relief: float | None = None, wrap: bool = False) -> None:
    """Write a PNG preview. ``mode`` is ``color``, ``hillshade`` or ``height``."""
    if mode == "color":
        img = Image.fromarray(colorize(heightmap, rivers, lakes=lakes, relief=relief, wrap=wrap))
    elif mode == "hillshade":
        img = Image.fromarray((hillshade(heightmap, relief=relief, wrap=wrap) * 255).astype(np.uint8))
    elif mode == "height":
        img = Image.fromarray((np.clip(heightmap, 0, 1) * 255).astype(np.uint8))
    else:
        raise ValueError(f"unknown mode {mode!r}")
    img.save(path)
