"""Turns a World into what the 3D view draws: colours, surface heights, matrices."""
from __future__ import annotations

import numpy as np

from ..biomes import Biomes
from ..render import colorize, colorize_layer, hillshade
from ..world import World

MODES = ("Landscape", "Biomes", "Temperature", "Rainfall", "Height")
_NEEDS = {"Biomes": "biome_weights", "Temperature": "temperature", "Rainfall": "rainfall"}


def available(world: World, mode: str) -> bool:
    return mode not in _NEEDS or _NEEDS[mode] in world


def colours(world: World, mode: str, steps=()) -> np.ndarray:
    """uint8 RGB for a display mode; falls back to Landscape if its layer is missing."""
    h = world["height"]
    get = world.layers.get
    if not available(world, mode):
        mode = "Landscape"
    look = dict(relief=world.relief, wrap=world.wrap, ocean=get("ocean"))
    if mode in ("Temperature", "Rainfall"):
        return colorize_layer(get(mode.lower()), mode.lower(), h, **look)
    if mode == "Height":
        shade = hillshade(h, relief=world.relief, wrap=world.wrap)
        grey = np.clip(h, 0, 1) * (0.5 + 0.5 * shade) * 255
        return np.repeat(grey[..., None], 3, axis=2).astype(np.uint8)
    extra = {}
    if mode == "Biomes":
        table = next((s.table for s in steps if isinstance(s, Biomes)
                      and len(s.table) == world["biome_weights"].shape[-1]), None)
        if table is not None:
            extra = dict(biome=world["biome"], biome_colors=[b.color for b in table],
                         biome_weights=world["biome_weights"])
    return colorize(h, get("rivers"), world.sea_level, get("lakes"),
                    lake_depth=get("lake_depth"), **look, **extra)


def surface(world: World) -> np.ndarray:
    """Height of the visible surface (terrain or water), as a fraction of world width."""
    top = np.array(world["height"], dtype=np.float32)
    if "lake_depth" in world:
        top += world["lake_depth"]
    if "ocean" in world and world.sea_level is not None:
        top[world["ocean"]] = world.sea_level
    return top * (world.relief / world.size)


def mesh(world: World, resolution: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """(vertices as x, y, z, u, v; triangle indices). Looping worlds close exactly."""
    top = surface(world)
    n = len(top)
    if world.wrap:
        top = np.pad(top, (0, 1), mode="wrap")
    last = len(top) - 1
    pick = np.unique(np.linspace(0, last, min(resolution, last) + 1).round().astype(int))
    y = top[np.ix_(pick, pick)]
    k = len(pick)
    span = n if world.wrap else n - 1
    p = pick / span
    z, x = np.meshgrid(p - 0.5, p - 0.5, indexing="ij")
    # texels sit at cell centres
    uv = (pick + 0.5) / n
    v, u = np.meshgrid(uv, uv, indexing="ij")
    verts = np.stack([x, y, z, u, v], axis=-1).astype(np.float32).reshape(-1, 5)
    i = np.arange(k * k, dtype=np.int32).reshape(k, k)
    a, b, c, d = i[:-1, :-1], i[:-1, 1:], i[1:, :-1], i[1:, 1:]
    tris = np.stack([a, c, b, b, c, d], axis=-1).reshape(-1)
    return verts, tris


# --------------------------------------------------------------- camera
def perspective(fov: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / np.tan(np.radians(fov) / 2)
    m = np.zeros((4, 4))
    m[0, 0], m[1, 1] = f / aspect, f
    m[2, 2], m[2, 3] = (far + near) / (near - far), 2 * far * near / (near - far)
    m[3, 2] = -1
    return m


class Camera:
    """Orbits a target point. Angles in degrees, distances in world widths."""

    def __init__(self):
        self.yaw, self.pitch, self.distance = 30.0, 40.0, 1.5
        self.target = np.array([0.0, 0.03, 0.0])

    def orbit(self, dx: float, dy: float) -> None:
        self.yaw = (self.yaw - dx) % 360
        self.pitch = float(np.clip(self.pitch + dy, 4, 89))

    def zoom(self, steps: float) -> None:
        self.distance = float(np.clip(self.distance * 0.88 ** steps, 0.08, 6))

    def pan(self, dx: float, dy: float) -> None:
        right, forward = self._axes()
        self.target += (-dx * right + dy * forward) * self.distance
        self.target[[0, 2]] = np.clip(self.target[[0, 2]], -1.5, 1.5)

    def _axes(self):
        a = np.radians(self.yaw)
        return np.array([np.cos(a), 0, -np.sin(a)]), np.array([-np.sin(a), 0, -np.cos(a)])

    def eye(self) -> np.ndarray:
        _, forward = self._axes()
        p = np.radians(self.pitch)
        return self.target + self.distance * (np.cos(p) * -forward + np.array([0, np.sin(p), 0]))

    def matrix(self, aspect: float) -> np.ndarray:
        eye = self.eye()
        f = self.target - eye
        f /= np.linalg.norm(f)
        right, _ = self._axes()
        up = np.cross(right, f)
        view = np.eye(4)
        view[0, :3], view[1, :3], view[2, :3] = right, up, -f
        view[:3, 3] = -view[:3, :3] @ eye
        return perspective(45, aspect, 0.01, 20) @ view
