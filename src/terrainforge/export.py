"""Export heightmaps and meshes to formats game engines understand."""
from __future__ import annotations

import numpy as np
from PIL import Image


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


def normal_map(heightmap: np.ndarray, strength: float | None = None) -> np.ndarray:
    """Tangent-space normal map as uint8 RGB. ``strength`` is relief in cell units."""
    size = heightmap.shape[0]
    strength = 0.1 * size if strength is None else strength
    gy, gx = np.gradient(heightmap.astype(np.float64) * strength)
    n = np.stack([-gx, -gy, np.ones_like(gx)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return ((n * 0.5 + 0.5) * 255.0 + 0.5).astype(np.uint8)


def save_normal_map(path: str, heightmap: np.ndarray, strength: float | None = None) -> None:
    Image.fromarray(normal_map(heightmap, strength)).save(path)


def save_obj(path: str, heightmap: np.ndarray, *, max_resolution: int = 256,
             horizontal_scale: float = 1.0, vertical_scale: float = 0.25) -> None:
    """Write a triangle mesh (.obj) with UVs and smooth normals.

    The map is block-averaged down to at most ``max_resolution`` vertices per
    side. ``vertical_scale`` is height as a fraction of the map width.
    """
    h = np.asarray(heightmap, dtype=np.float64)
    n = h.shape[0]
    step = max(1, int(np.ceil(n / max_resolution)))
    if step > 1:
        m = (n // step) * step
        h = h[:m, :m].reshape(m // step, step, m // step, step).mean(axis=(1, 3))
    r = h.shape[0]
    xs = np.linspace(0.0, horizontal_scale, r)
    X, Z = np.meshgrid(xs, xs)
    Y = h * vertical_scale * horizontal_scale
    gz, gx = np.gradient(Y, xs, xs)
    nrm = np.stack([-gx, np.ones_like(gx), -gz], axis=-1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    u = np.linspace(0, 1, r)
    U, V = np.meshgrid(u, 1 - u)

    with open(path, "w") as f:
        f.write("# terrainforge heightfield mesh\n")
        np.savetxt(f, np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]), fmt="v %.6f %.6f %.6f")
        np.savetxt(f, np.column_stack([U.ravel(), V.ravel()]), fmt="vt %.6f %.6f")
        np.savetxt(f, nrm.reshape(-1, 3), fmt="vn %.5f %.5f %.5f")
        idx = np.arange(r * r).reshape(r, r) + 1
        a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
        c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
        tris = np.concatenate([np.column_stack([a, c, b]), np.column_stack([b, c, d])])
        faces = np.repeat(tris, 3, axis=1)  # v/vt/vn share an index
        np.savetxt(f, faces, fmt="f %d/%d/%d %d/%d/%d %d/%d/%d")
