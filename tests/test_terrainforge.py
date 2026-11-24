import os

import numpy as np
import pytest

import terrainforge as tf
from terrainforge.cli import main as cli_main
from terrainforge.noise import fbm


# ---------------------------------------------------------------- noise
def test_fbm_range_shape_dtype():
    h = fbm(64, seed=1)
    assert h.shape == (64, 64) and h.dtype == np.float32
    assert h.min() == pytest.approx(0.0) and h.max() == pytest.approx(1.0)


def test_fbm_deterministic_and_seed_sensitive():
    assert np.array_equal(fbm(48, seed=5), fbm(48, seed=5))
    assert not np.array_equal(fbm(48, seed=5), fbm(48, seed=6))


def test_fbm_is_spatially_coherent():
    h = fbm(128, seed=2)
    neighbour_diff = np.abs(np.diff(h, axis=1)).mean()
    random_diff = np.abs(h - np.roll(h, 37, axis=1)).mean()
    assert neighbour_diff < 0.2 * random_diff


@pytest.mark.parametrize("kwargs", [{"size": 1}, {"size": 32, "octaves": 0}])
def test_fbm_rejects_bad_args(kwargs):
    with pytest.raises(ValueError):
        fbm(**kwargs)


# -------------------------------------------------------------- erosion
@pytest.fixture(scope="module")
def base():
    return tf.normalize(fbm(96, seed=3, scale=3, warp=0.35))


def test_hydraulic_conserves_mass_and_stays_bounded(base):
    out = tf.hydraulic_erosion(base, drops=8000, seed=1)
    relief = 0.1 * base.shape[0]
    assert abs(float((out - base).sum())) * relief < 1.0  # ~zero net mass change
    assert out.min() >= -0.05 and out.max() <= 1.05      # no runaway pits/spikes
    assert float(np.abs(out - base).sum()) > 0           # but it did change things


def test_hydraulic_does_not_touch_border_or_input(base):
    before = base.copy()
    out = tf.hydraulic_erosion(base, drops=3000, seed=1)
    assert np.array_equal(base, before)
    assert np.array_equal(out[0], base[0]) and np.array_equal(out[:, -1], base[:, -1])


def test_hydraulic_deterministic(base):
    a = tf.hydraulic_erosion(base, drops=2000, seed=9)
    b = tf.hydraulic_erosion(base, drops=2000, seed=9)
    assert np.array_equal(a, b)


def test_hydraulic_validates_input():
    with pytest.raises(ValueError):
        tf.hydraulic_erosion(np.zeros((16, 8)))
    with pytest.raises(ValueError):
        tf.hydraulic_erosion(np.zeros((4, 4)))
    with pytest.raises(ValueError):
        tf.hydraulic_erosion(np.zeros((16, 16)), inertia=1.0)


def test_thermal_reduces_steepness_and_conserves_mass():
    h = np.zeros((64, 64), dtype=np.float32)
    h[24:40, 24:40] = 1.0  # a tall plateau with vertical cliffs
    out = tf.thermal_erosion(h, iterations=200, talus=0.5)
    assert np.abs(np.gradient(out)).max() < np.abs(np.gradient(h)).max()
    assert float(out.sum()) == pytest.approx(float(h.sum()), rel=1e-4)


# ------------------------------------------------------------ hydrology
def test_flow_conserves_rain_and_routes_to_border():
    n = 64
    acc = tf.flow_accumulation(fbm(n, seed=4))
    border = np.zeros((n, n), bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    assert float(acc[border].sum()) == pytest.approx(n * n, rel=1e-5)


def test_flow_fills_pits():
    n = 33
    y, x = np.mgrid[:n, :n]
    bowl = ((x - 16) ** 2 + (y - 16) ** 2).astype(np.float64)
    bowl = 1.0 - bowl / bowl.max()  # peak at centre -> invert to make a pit
    pit = 1.0 - bowl                 # minimum at centre, high rim
    pit[0, :] = pit[-1, :] = pit[:, 0] = pit[:, -1] = 0.5  # rim drains outward
    acc = tf.flow_accumulation(pit)
    assert acc[16, 16] >= 1.0
    assert float(acc[[0, -1], :].sum() + acc[:, [0, -1]].sum()) > n * n * 0.9


def test_flow_on_tilted_plane_accumulates_downhill():
    n = 32
    plane = np.tile(np.linspace(1, 0, n), (n, 1))  # slopes toward +x
    acc = tf.flow_accumulation(plane)
    assert acc[16, -2] > acc[16, 2]


def test_river_mask_ignores_border_and_carve_lowers_terrain():
    h = fbm(64, seed=2)
    acc = tf.flow_accumulation(h)
    m = tf.river_mask(acc, threshold=5)
    assert not m[0].any() and not m[-1].any() and not m[:, 0].any() and not m[:, -1].any()
    carved = tf.carve_rivers(h, acc, depth=0.05, threshold=5)
    assert (carved <= h + 1e-7).all() and (carved < h).any()


def _naive_fill(h):
    """Reference depression fill (Planchon-Darboux), slow but obvious."""
    w = np.full_like(h, np.inf)
    w[0], w[-1], w[:, 0], w[:, -1] = h[0], h[-1], h[:, 0], h[:, -1]
    n = h.shape[0]
    while True:
        p = np.pad(w, 1, constant_values=np.inf)
        low = np.min([p[1 + dy:1 + dy + n, 1 + dx:1 + dx + n]
                      for dy in (-1, 0, 1) for dx in (-1, 0, 1)], axis=0)
        new = np.minimum(w, np.maximum(h, low))
        if np.array_equal(new, w):
            return w
        w = new


@pytest.mark.parametrize("h", [fbm(48, seed=1).astype(np.float64),
                               np.round(fbm(48, seed=2).astype(np.float64) * 12) / 12,
                               np.random.default_rng(0).random((48, 48)),
                               np.zeros((48, 48))],
                         ids=["fbm", "terraced", "white-noise", "flat"])
def test_routing_invariants(h):
    from terrainforge import hydrology as hy
    filled, parent, order = hy._flood(h, hy._OX, hy._OY)
    assert np.array_equal(filled, _naive_fill(h))
    assert np.array_equal(np.sort(order), np.arange(h.size))
    recv = hy._receivers(h, filled, parent, hy._OX, hy._OY, hy._DIST)
    src = np.flatnonzero(recv >= 0)
    # everything but the border drains, to a neighbour, never up the water surface
    border = np.ones(h.shape, bool)
    border[1:-1, 1:-1] = False
    assert np.array_equal(recv.reshape(h.shape) < 0, border)
    sy, sx = np.divmod(src, h.shape[0])
    ry, rx = np.divmod(recv[src], h.shape[0])
    assert (np.maximum(abs(sy - ry), abs(sx - rx)) == 1).all()
    assert (filled.ravel()[recv[src]] <= filled.ravel()[src]).all()
    # receivers are visited before donors, which rules out cycles
    rank = np.empty(h.size, dtype=int)
    rank[order] = np.arange(h.size)
    assert (rank[recv[src]] < rank[src]).all()
    acc = tf.flow_accumulation(h)
    assert float(acc[border].sum()) == pytest.approx(h.size)


def _two_basins():
    """Plateau at 0.8 with a deep basin (notch at 0.6) and a puddle."""
    h = np.full((40, 40), 0.8)
    h[10:20, 10:20] = 0.3
    h[15, 20:] = 0.6         # outlet channel from the basin to the border
    h[30:32, 30:32] = 0.79   # puddle
    return h


def test_lake_depth_and_spill_level():
    lakes = tf.detect_lakes(_two_basins(), min_depth=0.05)
    expected = np.zeros((40, 40), bool)
    expected[10:20, 10:20] = True
    assert np.array_equal(lakes.mask, expected)
    assert lakes.depth.dtype == np.float32
    assert lakes.depth[expected] == pytest.approx(0.3)
    assert (lakes.depth[~expected] == 0).all()


def test_lake_filters():
    h = _two_basins()
    assert tf.detect_lakes(h, min_depth=0.0, min_area=0).mask.sum() == 104
    assert tf.detect_lakes(h, min_depth=0.0, min_area=5).mask.sum() == 100
    assert not tf.detect_lakes(h, min_depth=0.0, min_area=101).mask.any()


def test_no_lakes_on_slope_or_open_basin():
    plane = np.tile(np.linspace(1, 0, 32), (32, 1))
    assert not tf.detect_lakes(plane, min_depth=0.0, min_area=0).mask.any()
    h = np.full((32, 32), 0.8)
    h[10:20, 10:] = 0.2  # open to the border
    assert not tf.detect_lakes(h, min_depth=0.0, min_area=0).mask.any()


def test_lake_drains_through_its_outlet_only():
    h = _two_basins()
    acc = tf.flow_accumulation(h)
    assert acc[15, 20] >= 100  # everything in the lake leaves by the notch
    ring = np.zeros_like(h, bool)
    ring[9:21, 9:21] = True
    ring[10:20, 10:20] = False
    ring[14:17, 20] = False
    assert acc[ring].max() < 100


def test_flow_in_lake_follows_submerged_valley():
    """A V-shaped lake bed must collect flow along its thalweg, not in stripes."""
    n = 41
    y, x = np.mgrid[:n, :n]
    h = 0.05 + 0.01 * abs(y - 20) + 0.001 * x  # valley rising toward +x
    h[:, 0] = h[0] = h[-1] = h[:, -1] = 1.0
    h[20, -1] = 0.5                            # spill point
    acc = tf.flow_accumulation(h)
    mid = acc[1:-1, 5:36]
    assert (mid.argmax(axis=0) == 19).all()
    assert (np.sort(mid, axis=0)[-2] < 0.5 * mid.max(axis=0)).all()


def test_flat_channels_merge_and_bend():
    """On an exact flat with one outlet, no channel runs straight for long."""
    h = np.zeros((64, 64))
    h[0] = h[-1] = h[:, 0] = h[:, -1] = 1.0
    h[32, -1] = 0.0
    river = tf.river_mask(tf.flow_accumulation(h), threshold=20)
    longest = 0
    diagonals = [np.diagonal(m, i) for m in (river, river[::-1]) for i in range(-63, 64)]
    for row in [*river, *river.T, *diagonals]:
        run = 0
        for v in row:
            run = run + 1 if v else 0
            longest = max(longest, run)
    assert river.sum() > 40 and longest < 15


def test_rivers_and_carving_skip_lakes():
    h = fbm(96, seed=4)
    lakes = tf.detect_lakes(h)
    assert lakes.mask.any()
    acc = tf.flow_accumulation(h)
    rivers = tf.river_mask(acc, 20, lakes.mask)
    assert not (rivers & lakes.mask).any()
    assert np.array_equal(rivers | lakes.mask, tf.river_mask(acc, 20) | lakes.mask)
    carved = tf.carve_rivers(h, acc, threshold=20, lakes=lakes.mask)
    assert np.array_equal(carved[lakes.mask], h[lakes.mask])


def test_hydrology_is_pure_and_deterministic():
    h = fbm(64, seed=4)
    before = h.copy()
    a, b = tf.detect_lakes(h), tf.detect_lakes(h)
    assert np.array_equal(a.mask, b.mask) and np.array_equal(a.depth, b.depth)
    assert np.array_equal(tf.flow_accumulation(h), tf.flow_accumulation(h))
    assert np.array_equal(h, before)


@pytest.mark.parametrize("bad", [np.zeros((4, 8)), np.zeros(16), np.zeros((0, 0)),
                                 np.full((8, 8), np.nan)])
def test_hydrology_validates_heightmap(bad):
    with pytest.raises(ValueError):
        tf.detect_lakes(bad)
    with pytest.raises(ValueError):
        tf.flow_accumulation(bad)


@pytest.mark.parametrize("kwargs", [{"min_depth": -1}, {"min_area": -1}])
def test_detect_lakes_validates_filters(kwargs):
    with pytest.raises(ValueError):
        tf.detect_lakes(np.zeros((8, 8)), **kwargs)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_hydrology_tiny_maps(n):
    h = np.zeros((n, n))
    assert tf.flow_accumulation(h).shape == (n, n)
    assert not tf.detect_lakes(h, min_depth=0, min_area=0).mask.any()


# --------------------------------------------------------------- export
def test_png16_roundtrip(tmp_path):
    h = fbm(40, seed=1)
    p = str(tmp_path / "h.png")
    tf.save_png16(p, h)
    assert np.abs(tf.load_png16(p) - h).max() <= 1 / 65535 + 1e-6


def test_raw16_size_and_endianness(tmp_path):
    h = np.linspace(0, 1, 16 * 16, dtype=np.float32).reshape(16, 16)
    p = str(tmp_path / "h.r16")
    tf.save_raw16(p, h)
    assert os.path.getsize(p) == 16 * 16 * 2
    assert np.fromfile(p, "<u2")[-1] == 65535
    tf.save_raw16(p, h, big_endian=True)
    assert np.fromfile(p, ">u2")[-1] == 65535


def test_normal_map_flat_is_up():
    n = tf.normal_map(np.full((16, 16), 0.5, dtype=np.float32))
    assert tuple(n[8, 8]) == (128, 128, 255)


def test_obj_topology(tmp_path):
    h = fbm(32, seed=1)
    p = str(tmp_path / "t.obj")
    tf.save_obj(p, h, max_resolution=32)
    lines = open(p).read().splitlines()
    assert sum(l.startswith("v ") for l in lines) == 32 * 32
    assert sum(l.startswith("f ") for l in lines) == 2 * 31 * 31
    max_idx = max(int(t.split("/")[0]) for l in lines if l.startswith("f ") for t in l.split()[1:])
    assert max_idx == 32 * 32


def test_obj_downsamples(tmp_path):
    p = str(tmp_path / "t.obj")
    tf.save_obj(p, fbm(64, seed=1), max_resolution=16)
    assert sum(l.startswith("v ") for l in open(p)) == 16 * 16


# ------------------------------------------------------------- pipeline
def test_generate_deterministic_and_valid():
    a = tf.generate(64, seed=7, erosion_drops=2000)
    b = tf.generate(64, seed=7, erosion_drops=2000)
    assert np.array_equal(a.heightmap, b.heightmap)
    assert a.heightmap.shape == (64, 64)
    assert 0.0 <= a.heightmap.min() and a.heightmap.max() <= 1.0
    assert a.rivers.dtype == bool and a.flow.shape == (64, 64)
    assert np.array_equal(a.lakes, b.lakes)
    assert np.array_equal(a.lakes, a.lake_depth > 0)
    assert not (a.lakes & a.rivers).any()


def test_cli_smoke(tmp_path):
    out = str(tmp_path / "o")
    assert cli_main(["generate", "--size", "64", "--drops", "1000", "-o", out]) == 0
    for name in ["heightmap.png", "heightmap.r16", "normal.png", "terrain.obj",
                 "preview.png", "preview_base.png"]:
        assert os.path.getsize(os.path.join(out, name)) > 0
