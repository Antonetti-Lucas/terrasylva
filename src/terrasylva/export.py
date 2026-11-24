"""Export heightmaps and meshes to formats game engines understand."""
from __future__ import annotations

import numpy as np
from PIL import Image

from ._grid import gradient


def save_png16(path: str, heightmap: np.ndarray) -> None:
    """16-bit greyscale PNG (Unreal, Unity, Blender displacement)."""
    a = (np.clip(heightmap, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
    Image.fromarray(a).save(path)


def load_png16(path: str) -> np.ndarray:
    """Inverse of :func:`save_png16`; returns float32 in [0, 1]."""
    a = np.asarray(Image.open(path))
    if a.dtype != np.uint16:
        raise ValueError("expected a 16-bit PNG")
    return (a.astype(np.float32) / 65535.0)


def save_raw16(path: str, heightmap: np.ndarray, big_endian: bool = False) -> None:
    """Headerless 16-bit RAW (Unity terrain: little-endian on Windows, big on Mac)."""
    a = (np.clip(heightmap, 0.0, 1.0) * 65535.0 + 0.5).astype(">u2" if big_endian else "<u2")
    a.tofile(path)


def water_map(shape: tuple[int, int], *, ocean: np.ndarray | None = None,
              lakes: np.ndarray | None = None,
              rivers: np.ndarray | None = None) -> np.ndarray:
    """uint8 map of water: 0 land, 85 river, 170 lake, 255 ocean."""
    out = np.zeros(shape, dtype=np.uint8)
    for mask, value in ((rivers, 85), (lakes, 170), (ocean, 255)):
        if mask is not None:
            out[mask] = value
    return out


def save_water_map(path: str, shape: tuple[int, int], **masks) -> None:
    Image.fromarray(water_map(shape, **masks)).save(path)


def normal_map(heightmap: np.ndarray, strength: float | None = None, *,
               wrap: bool = False) -> np.ndarray:
    """Tangent-space normal map as uint8 RGB. ``strength`` is relief in cell units."""
    size = heightmap.shape[0]
    strength = 0.1 * size if strength is None else strength
    gy, gx = gradient(heightmap.astype(np.float64) * strength, wrap)
    n = np.stack([-gx, -gy, np.ones_like(gx)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return ((n * 0.5 + 0.5) * 255.0 + 0.5).astype(np.uint8)


def save_normal_map(path: str, heightmap: np.ndarray, strength: float | None = None, *,
                    wrap: bool = False) -> None:
    Image.fromarray(normal_map(heightmap, strength, wrap=wrap)).save(path)


def save_obj(path: str, heightmap: np.ndarray, *, max_resolution: int = 256,
             horizontal_scale: float = 1.0, vertical_scale: float = 0.25,
             wrap: bool = False) -> None:
    """Write a triangle mesh (.obj) with UVs and smooth normals.

    The map is block-averaged down to at most ``max_resolution`` vertices per
    side. ``vertical_scale`` is height as a fraction of the map width.
    With ``wrap`` the first row and column are repeated at the far edges, so
    copies placed side by side meet exactly.
    """
    h = np.asarray(heightmap, dtype=np.float64)
    n = h.shape[0]
    step = max(1, int(np.ceil(n / max_resolution)))
    while wrap and n % step:
        step += 1  # blocks must fit exactly or the tile would lose its edge
    if step > 1:
        m = (n // step) * step
        h = h[:m, :m].reshape(m // step, step, m // step, step).mean(axis=(1, 3))
    if wrap:
        h = np.pad(h, (0, 1), mode="wrap")
    r = h.shape[0]
    xs = np.linspace(0.0, horizontal_scale, r)
    X, Z = np.meshgrid(xs, xs)
    Y = h * vertical_scale * horizontal_scale
    if wrap:
        # one more ring on each side, so normals match across tiles too
        ring = np.pad(Y[:-1, :-1], ((1, 2), (1, 2)), mode="wrap")
        gz, gx = (g[1:-1, 1:-1] for g in np.gradient(ring, xs[1] - xs[0]))
    else:
        gz, gx = np.gradient(Y, xs, xs)
    nrm = np.stack([-gx, np.ones_like(gx), -gz], axis=-1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    u = np.linspace(0, 1, r)
    U, V = np.meshgrid(u, 1 - u)

    with open(path, "w") as f:
        f.write("# terrasylva heightfield mesh\n")
        np.savetxt(f, np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]), fmt="v %.6f %.6f %.6f")
        np.savetxt(f, np.column_stack([U.ravel(), V.ravel()]), fmt="vt %.6f %.6f")
        np.savetxt(f, nrm.reshape(-1, 3), fmt="vn %.5f %.5f %.5f")
        idx = np.arange(r * r).reshape(r, r) + 1
        a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
        c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
        tris = np.concatenate([np.column_stack([a, c, b]), np.column_stack([b, c, d])])
        faces = np.repeat(tris, 3, axis=1)  # v/vt/vn share an index
        np.savetxt(f, faces, fmt="f %d/%d/%d %d/%d/%d %d/%d/%d")
