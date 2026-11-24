import json
import os
from dataclasses import dataclass

import numpy as np
import pytest

import terrasylva as ts
from terrasylva import hydrology as hy
from terrasylva.cli import main as cli_main

SHIFT = (17, 41)


def roll(a):
    return np.roll(a, SHIFT, axis=(0, 1))


def seam_ratio(h):
    """Height steps across the seams relative to steps inside the map."""
    seam = (np.abs(h[:, 0] - h[:, -1]).mean() + np.abs(h[0] - h[-1]).mean()) / 2
    inside = (np.abs(np.diff(h, axis=0)).mean() + np.abs(np.diff(h, axis=1)).mean()) / 2
    return seam / inside


@pytest.fixture(scope="module")
def loop():
    return ts.fbm(64, seed=3, scale=3, warp=0.35, wrap=True)


# ------------------------------------------------------------- wrapping
@pytest.mark.parametrize("ridged", [False, True])
def test_wrapped_noise_is_exactly_periodic(ridged):
    kw = dict(seed=5, scale=3, warp=0.35, ridged=ridged, wrap=True)
    a = ts.fbm(64, **kw)
    assert np.allclose(a, ts.fbm(64, offset=(3, -6), **kw), atol=2e-5)
    assert seam_ratio(a) < 1.6


def test_wrapped_noise_periodic_past_permutation_table():
    # 3 * 2**7 = 384 lattice cells in the last octave, more than the table's 256
    kw = dict(seed=1, scale=3, octaves=8, wrap=True)
    assert np.allclose(ts.fbm(512, **kw), ts.fbm(512, offset=(3, 0), **kw), atol=2e-5)


@pytest.mark.parametrize("kw", [{"scale": 2.5}, {"scale": 0}, {"lacunarity": 1.9}])
def test_wrapped_noise_rejects_unloopable_parameters(kw):
    with pytest.raises(ValueError):
        ts.fbm(32, wrap=True, **kw)


def test_wrapped_thermal_commutes_with_shifting_and_conserves_mass(loop):
    kw = dict(iterations=20, talus=0.1, wrap=True)
    out = ts.thermal_erosion(loop, **kw)
    assert not np.array_equal(out, loop)
    assert np.allclose(ts.thermal_erosion(roll(loop), **kw), roll(out), atol=1e-6)
    assert float(out.sum()) == pytest.approx(float(loop.sum()), rel=1e-5)


def test_wrapped_hydraulic_conserves_mass_and_reaches_the_edges(loop):
    out = ts.hydraulic_erosion(loop, drops=6000, seed=1, wrap=True)
    assert abs(float((out.astype(np.float64) - loop).sum())) < 1e-2
    assert out.min() > -0.2 and out.max() < 1.2
    change = np.abs(out - loop)
    edge = np.concatenate([change[0], change[-1], change[:, 0], change[:, -1]])
    # edges are ordinary terrain: eroded about as much as everything else
    assert 0.5 < edge.mean() / change.mean() < 2.0
    assert seam_ratio(out) < 1.6


def test_wrapped_hydraulic_survives_small_maps_and_border_crossings():
    n = 8
    ramp = np.tile(np.linspace(0, 1, n), (n, 1))  # a cliff right on the seam
    out = ts.hydraulic_erosion(ramp, drops=3000, seed=2, wrap=True)
    assert np.isfinite(out).all()
    assert float(out.sum()) == pytest.approx(float(ramp.sum()), abs=1e-3)


def test_wrapped_hydrology_commutes_with_shifting(loop):
    ocean = loop < np.quantile(loop, 0.3)
    kw = dict(wrap=True)
    a = ts.detect_lakes(loop, outlets=ocean, **kw)
    b = ts.detect_lakes(roll(loop), outlets=roll(ocean), **kw)
    assert np.array_equal(roll(a.mask), b.mask) and np.array_equal(roll(a.depth), b.depth)
    assert np.array_equal(roll(ts.fill_depressions(loop, outlets=ocean, **kw)),
                          ts.fill_depressions(roll(loop), outlets=roll(ocean), **kw))


@pytest.mark.parametrize("with_ocean", [True, False])
def test_wrapped_flow_all_reaches_outlets(loop, with_ocean):
    ocean = loop < np.quantile(loop, 0.3) if with_ocean else None
    out = hy._outlet_mask(loop, ocean, True)
    assert out.any()
    acc = ts.flow_accumulation(loop, outlets=ocean, wrap=True)
    assert float(acc[out].sum()) == pytest.approx(loop.size)


def test_wrapped_routing_is_acyclic_and_stays_adjacent(loop):
    h = loop.astype(np.float64)
    n = len(h)
    out = hy._outlet_mask(h, None, True)
    filled, parent, order = hy._flood(h, out, True, hy._OX, hy._OY)
    recv = hy._receivers(h, filled, parent, out, hy._OX, hy._OY, hy._DIST)
    src = np.flatnonzero(recv >= 0)
    assert np.array_equal(recv.reshape(n, n) < 0, out)
    d = np.abs(np.array(np.divmod(src, n)) - np.array(np.divmod(recv[src], n)))
    assert (np.minimum(d, n - d).max(axis=0) == 1).all()
    rank = np.empty(h.size, dtype=int)
    rank[order] = np.arange(h.size)
    assert (rank[recv[src]] < rank[src]).all()


def test_lake_across_the_seam_is_one_lake():
    h = np.full((32, 32), 0.8)
    h[10:20, [0, 1, 30, 31]] = 0.3   # 40 cells, split by the edge
    h[25, 15] = 0.0                  # outlet elsewhere
    out = np.zeros((32, 32), bool)
    out[25, 15] = True
    assert ts.detect_lakes(h, min_area=40, outlets=out, wrap=True).mask.sum() == 40


@pytest.mark.parametrize("bad", [np.zeros((8, 8)), np.zeros((4, 4), bool)])
def test_outlets_are_validated(bad):
    with pytest.raises(ValueError):
        ts.flow_accumulation(np.zeros((8, 8)), outlets=bad)


def test_wrapped_shading_and_carving_commute_with_shifting(loop):
    for f in (lambda h: ts.hillshade(h, wrap=True), lambda h: ts.slope(h, wrap=True),
              lambda h: ts.normal_map(h, wrap=True).astype(float)):
        assert np.allclose(f(roll(loop)), roll(f(loop)), atol=1e-5)
    acc = ts.flow_accumulation(loop, wrap=True)
    carved = ts.carve_rivers(loop, acc, threshold=20, wrap=True)
    assert np.allclose(ts.carve_rivers(roll(loop), roll(acc), threshold=20, wrap=True),
                       roll(carved), atol=1e-6)


@pytest.mark.parametrize("f", [ts.hydraulic_erosion, ts.thermal_erosion])
def test_erosion_rejects_nan(f):
    with pytest.raises(ValueError):
        f(np.full((16, 16), np.nan))


# ---------------------------------------------------------------- world
def test_world_units():
    w = ts.World(size=64, cell_size=2.0, height_range=(-100, 300))
    assert w.relief == 200
    w["height"] = np.full((64, 64), 0.25)
    assert w.metres() == pytest.approx(0.0)
    assert ts.World(size=64, cell_size=4.0).relief == pytest.approx(6.4)


@pytest.mark.parametrize("kw", [{"size": 4}, {"cell_size": 0}, {"height_range": (5, 5)},
                                {"size": 16, "layers": {"height": np.zeros((8, 8))}}])
def test_world_validates(kw):
    with pytest.raises(ValueError):
        ts.World(**kw)


def test_missing_layer_error_names_it():
    with pytest.raises(KeyError, match="height"):
        ts.World()["height"]


def test_seeds_are_independent_and_stable():
    seeds = {ts.World(seed=s).seed_for(n) for s in range(50) for n in ("a", "b", "c")}
    assert len(seeds) == 150
    assert ts.World(seed=42).seed_for("noise") == ts.World(seed=42, size=64).seed_for("noise")


def small(**kw):
    return ts.World(size=64, seed=7, **kw)


STEPS = [ts.Noise(), ts.Thermal(), ts.Hydraulic(drops=1500), ts.Rivers()]


@pytest.mark.parametrize("wrap", [False, True])
def test_pipeline_deterministic_and_input_untouched(wrap):
    w = small(wrap=wrap)
    a, b = ts.Pipeline(STEPS).run(w), ts.Pipeline(STEPS).run(w)
    assert not w.layers
    for name in a.layers:
        assert np.array_equal(a[name], b[name])
    assert 0 <= a["height"].min() and a["height"].max() <= 1
    assert not (a["rivers"] & a["lakes"]).any()


def test_removing_a_step_keeps_other_steps_random_streams():
    full = ts.Pipeline(STEPS).run(small())
    without = ts.Pipeline([STEPS[0], STEPS[2]]).run(small())
    assert np.array_equal(full["base"], without["base"])
    direct = ts.hydraulic_erosion(full["base"], drops=1500, relief=small().relief,
                                  seed=small().seed_for("hydraulic"))
    assert np.array_equal(without["height"], direct)


def test_resuming_equals_running_straight_through():
    p = ts.Pipeline(STEPS)
    half = p.run(small(), stop="thermal")
    resumed = p.run(half, start="hydraulic")
    straight = p.run(small())
    assert "rivers" not in half
    assert np.array_equal(resumed["height"], straight["height"])
    assert np.array_equal(resumed["rivers"], straight["rivers"])


def test_sculpted_height_survives_and_water_updates():
    p = ts.Pipeline([*STEPS[:3], ts.Rivers(carve=False)])
    w = p.run(small())
    h = w["height"].copy()
    h[20:30, 20:30] = 0.0
    w["height"] = h
    again = p.run(w, start="rivers")
    assert np.array_equal(again["height"], h)
    assert again["lakes"][22:28, 22:28].all()


def test_mask_protects_cells_exactly():
    w = ts.Pipeline([ts.Noise()]).run(small())
    keep = np.zeros((64, 64), dtype=np.float32)
    keep[:, :32] = 1
    w["soft"] = keep
    out = ts.Pipeline([ts.Hydraulic(drops=3000, mask="soft"),
                       ts.Thermal(talus=0.1, mask="soft")]).run(w)
    assert np.array_equal(out["height"][:, 32:], w["height"][:, 32:])
    assert not np.array_equal(out["height"][:, :32], w["height"][:, :32])


def test_custom_function_step_and_ocean_layer():
    def flood_low_ground(world):
        world["ocean"] = world["height"] < np.quantile(world["height"], 0.3)

    out = ts.Pipeline([ts.Noise(), flood_low_ground, ts.Rivers(threshold=20)]).run(small(wrap=True))
    assert out["ocean"].any() and out["rivers"].any()
    assert not (out["rivers"] & out["ocean"]).any()
    assert not (out["lakes"] & out["ocean"]).any()


def test_pipeline_validates():
    with pytest.raises(ValueError, match="unique"):
        ts.Pipeline([ts.Thermal(), ts.Thermal()])
    p = ts.Pipeline(STEPS)
    with pytest.raises(ValueError, match="no step"):
        p.run(small(), start="nope")
    with pytest.raises(ValueError, match="start"):
        p.run(small(), start="rivers", stop="noise")


# -------------------------------------------------------------- recipes
@dataclass(frozen=True)
class Raise:
    amount: float = 0.1
    name: str = "raise"

    def __call__(self, world):
        world["height"] = np.clip(world["height"] + self.amount, 0, 1)


def test_recipe_roundtrip_with_custom_step(tmp_path):
    path = str(tmp_path / "r.json")
    w = small(wrap=True, cell_size=2.0, height_range=(-50, 200))
    p = ts.Pipeline([ts.Noise(scale=2.0), Raise(0.2), ts.Rivers()])
    ts.save_recipe(path, w, p)
    w2, p2 = ts.load_recipe(path, {"Raise": Raise})
    assert w2 == w and p2.steps == p.steps
    assert np.array_equal(p.run(w)["height"], p2.run(w2)["height"])
    with pytest.raises(ValueError, match="unknown step"):
        ts.load_recipe(path)


def test_recipe_rejects_functions_and_bad_files(tmp_path):
    path = str(tmp_path / "r.json")
    with pytest.raises(ValueError, match="dataclass"):
        ts.save_recipe(path, small(), ts.Pipeline([lambda w: None]))
    for bad in ({}, {"world": {"nope": 1}, "steps": []},
                {"world": {}, "steps": [{"step": "Noise", "nope": 1}]}):
        with open(path, "w") as f:
            json.dump(bad, f)
        with pytest.raises(ValueError):
            ts.load_recipe(path)


def test_cli_recipe_reproduces_output(tmp_path):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    assert cli_main(["generate", "--size", "64", "--drops", "1000", "--wrap", "-o", a]) == 0
    assert cli_main(["run", os.path.join(a, "recipe.json"), "-o", b]) == 0
    for name in ("heightmap.r16", "recipe.json"):
        assert open(os.path.join(a, name), "rb").read() == open(os.path.join(b, name), "rb").read()


def test_generate_wrap_has_no_seam():
    t = ts.generate(128, seed=3, wrap=True, erosion_drops=6000)
    assert seam_ratio(t.heightmap) < 1.6
    assert seam_ratio(ts.generate(128, seed=3, erosion_drops=6000).heightmap) > 5
