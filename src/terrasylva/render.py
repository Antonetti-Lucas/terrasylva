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


def biome_rgb(biome: np.ndarray, colors, weights: np.ndarray | None = None) -> np.ndarray:
    """Float RGB of a biome map. ``weights`` (size, size, N) blends colours smoothly."""
    colors = np.asarray(colors, dtype=np.float64)
    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError("biome_colors must have shape (N, 3)")
    if weights is not None:
        weights = np.asarray(weights)
        if weights.ndim != 3 or weights.shape[2] != len(colors):
            raise ValueError("biome_weights must have shape (size, size, N) to match biome_colors")
        return weights.astype(np.float64) @ colors
    biome = np.asarray(biome)
    if biome.size and int(biome.max()) >= len(colors):
        raise ValueError("biome has indices beyond biome_colors")
    return colors[biome]


def colorize(heightmap: np.ndarray, rivers: np.ndarray | None = None,
             sea_level: float | None = None,
             lakes: np.ndarray | None = None, relief: float | None = None,
             wrap: bool = False, *, ocean: np.ndarray | None = None,
             lake_depth: np.ndarray | None = None, biome: np.ndarray | None = None,
             biome_colors=None, biome_weights: np.ndarray | None = None) -> np.ndarray:
    """Hypsometric colouring with hillshade, as uint8 RGB.

    Only actual water is blue: ``ocean``, ``lakes`` and ``rivers`` masks. With a
    ``sea_level`` but no ``ocean``, everything below that level is sea.

    Pass ``biome`` (an index map) and ``biome_colors`` (an ``(N, 3)`` table) to
    colour the land by biome instead of by height; ``biome_weights`` blends the
    colours smoothly instead. Shading and water are drawn on top either way.
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
    if biome_colors is not None:
        if biome is None and biome_weights is None:
            raise ValueError("biome_colors needs biome or biome_weights")
        rgb = biome_rgb(biome, biome_colors, biome_weights)
    elif biome is not None or biome_weights is not None:
        raise ValueError("biome needs biome_colors")
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


# value stops for the climate previews; rainfall is drawn on a square-root scale
_TEMPERATURE = [(-25.0, (49, 54, 149)), (-8.0, (116, 173, 209)), (3.0, (235, 240, 242)),
                (15.0, (254, 224, 139)), (26.0, (244, 109, 67)), (36.0, (165, 0, 38))]
_RAINFALL = [(0.0, (232, 222, 176)), (0.45, (196, 214, 120)), (0.9, (92, 164, 96)),
             (1.35, (40, 120, 160)), (2.2, (24, 48, 128))]
_PALETTES = {"temperature": (_TEMPERATURE, lambda v: v),
             "rainfall": (_RAINFALL, lambda v: np.sqrt(np.maximum(v, 0.0))),
             "moisture": (_RAINFALL, lambda v: 2.2 * np.clip(v, 0.0, 1.0))}


def colorize_layer(values: np.ndarray, kind: str, heightmap: np.ndarray | None = None, *,
                   ocean: np.ndarray | None = None, relief: float | None = None,
                   wrap: bool = False) -> np.ndarray:
    """Colour a climate layer (``temperature``, ``rainfall`` or ``moisture``) as uint8 RGB.

    With a ``heightmap`` the land is hillshaded; ``ocean`` is dimmed to a dark sea colour.
    """
    if kind not in _PALETTES:
        raise ValueError(f"unknown layer kind {kind!r}; use one of {sorted(_PALETTES)}")
    stops, transform = _PALETTES[kind]
    rgb = _ramp(transform(np.asarray(values, dtype=np.float64)), stops)
    if heightmap is not None:
        shade = hillshade(heightmap, relief=relief, wrap=wrap)
        rgb = rgb * (0.6 + 0.5 * shade[..., None])
    if ocean is not None:
        rgb[ocean] = rgb[ocean] * 0.25 + np.array([24, 40, 72]) * 0.75
    return np.clip(rgb, 0, 255).astype(np.uint8)


def save_layer_preview(path: str, values: np.ndarray, kind: str,
                       heightmap: np.ndarray | None = None, **kw) -> None:
    """Write a PNG of a climate layer; see :func:`colorize_layer`."""
    Image.fromarray(colorize_layer(values, kind, heightmap, **kw)).save(path)


def save_preview(path: str, heightmap: np.ndarray, rivers: np.ndarray | None = None,
                 mode: str = "color", lakes: np.ndarray | None = None,
                 relief: float | None = None, wrap: bool = False,
                 sea_level: float | None = None, ocean: np.ndarray | None = None,
                 lake_depth: np.ndarray | None = None, biome: np.ndarray | None = None,
                 biome_colors=None, biome_weights: np.ndarray | None = None) -> None:
    """Write a PNG preview. ``mode`` is ``color``, ``hillshade`` or ``height``.

    ``biome``, ``biome_colors`` and ``biome_weights`` are passed to :func:`colorize`.
    """
    if mode == "color":
        img = Image.fromarray(colorize(heightmap, rivers, sea_level, lakes, relief, wrap,
                                       ocean=ocean, lake_depth=lake_depth, biome=biome,
                                       biome_colors=biome_colors, biome_weights=biome_weights))
    elif mode == "hillshade":
        img = Image.fromarray((hillshade(heightmap, relief=relief, wrap=wrap) * 255).astype(np.uint8))
    elif mode == "height":
        img = Image.fromarray((np.clip(heightmap, 0, 1) * 255).astype(np.uint8))
    else:
        raise ValueError(f"unknown mode {mode!r}")
    img.save(path)
