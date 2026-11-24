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


# ---------------------------------------------------------------- ocean
OCEAN_STEPS = [ts.Noise(), ts.Continents(), ts.Thermal(), ts.Hydraulic(drops=1500),
               ts.Ocean(land=0.6), ts.Rivers(threshold=20)]


@pytest.fixture(scope="module", params=[False, True], ids=["bordered", "looping"])
def sea(request):
    return ts.Pipeline(OCEAN_STEPS).run(ts.World(size=96, seed=3, wrap=request.param))


def test_ocean_layers_agree(sea):
    h, ocean = sea["height"], sea["ocean"]
    assert (h[ocean] < sea.sea_level).all()
    assert np.allclose(sea["ocean_depth"][ocean], sea.sea_level - h[ocean], atol=1e-6)
    assert (sea["ocean_depth"][~ocean] == 0).all()
    assert not (ocean & sea["lakes"]).any() and not (ocean & sea["rivers"]).any()


def test_land_fraction_is_met(sea):
    assert 0.6 - 1e-3 <= (~sea["ocean"]).mean() < 0.7
    # everything that isn't land or ocean is an enclosed basin
    assert (sea["height"] < sea.sea_level).mean() == pytest.approx(0.4, abs=0.01)


def test_all_rain_reaches_ocean_or_border(sea):
    out = hy._outlet_mask(sea["height"], sea["ocean"], sea.wrap)
    assert float(sea["flow"][out].sum()) == pytest.approx(sea.size ** 2, rel=1e-5)


def test_carving_does_not_push_land_below_the_sea():
    p = ts.Pipeline(OCEAN_STEPS[:-1])
    w = p.run(ts.World(size=96, seed=3, wrap=True))
    carved = ts.Pipeline([ts.Rivers(threshold=10)]).run(w)
    was_land = w["height"] >= np.float32(w.sea_level)
    assert (carved["height"] < w["height"]).any()
    assert (carved["height"][was_land] >= np.float32(w.sea_level)).all()


def test_ocean_mask_separates_inland_basins_and_joins_across_the_seam():
    h = np.full((40, 40), 0.8)
    h[:, :6] = h[:, -6:] = 0.1       # open water on both sides of the seam
    h[18:21, 18:21] = 0.1            # small inland basin
    looped = ts.ocean_mask(h, 0.5, min_area=0.25, wrap=True)
    assert looped.sum() == 480 and not looped[19, 19]
    assert ts.ocean_mask(h, 0.5, min_area=0.0, wrap=True)[19, 19]
    # unconnected halves: neither reaches min_area, so the largest one remains
    h[:, 0] = 0.8
    assert ts.ocean_mask(h, 0.5, min_area=0.5).sum() == 240
    assert not ts.ocean_mask(h, 0.0).any()


def test_enclosed_basin_below_sea_becomes_a_lake():
    h = np.full((64, 64), 0.8, dtype=np.float32)
    h[:, :16] = 0.1
    h[30:36, 40:46] = 0.2            # below sea level but enclosed
    w = ts.World(size=64, wrap=True, layers={"height": h})
    w = ts.Pipeline([ts.Ocean(level=0.5), ts.Rivers(carve=False)]).run(w)
    assert w["lakes"][30:36, 40:46].all() and not w["ocean"][30:36, 40:46].any()
    assert w.sea_level == 0.5


def test_continents_loop_and_change_with_seed():
    p = ts.Pipeline([ts.Noise(), ts.Continents(scale=3)])
    a = p.run(ts.World(size=96, seed=1, wrap=True))
    b = p.run(ts.World(size=96, seed=2, wrap=True))
    assert seam_ratio(a["height"]) < 1.6 and seam_ratio(a["continents"]) < 1.6
    assert not np.array_equal(a["continents"], b["continents"])
    assert a["height"].min() == 0 and a["height"].max() == 1


@pytest.mark.parametrize("step", [ts.Ocean(land=0), ts.Ocean(land=1.0), ts.Ocean(min_area=2),
                                  ts.Ocean(level=float("nan")), ts.Continents(strength=2)])
def test_ocean_and_continents_validate(step):
    with pytest.raises(ValueError):
        ts.Pipeline([ts.Noise(), step]).run(small())


def test_generate_with_and_without_land():
    t = ts.generate(64, seed=1, erosion_drops=500, land=0.5, wrap=True)
    assert t.ocean.any() and 0 < t.sea_level < 1
    t = ts.generate(64, seed=1, erosion_drops=500)
    assert t.ocean is None and t.sea_level is None


# ---------------------------------------------------- previews and export
def is_blue(rgb):
    rgb = rgb.astype(int)
    return (rgb[..., 2] > rgb[..., 0] + 20) & (rgb[..., 2] > rgb[..., 1])


def test_only_water_is_blue(sea):
    img = ts.colorize(sea["height"], None, sea.sea_level, sea["lakes"], sea.relief, sea.wrap,
                      ocean=sea["ocean"], lake_depth=sea["lake_depth"])
    water = sea["ocean"] | sea["lakes"]
    assert np.array_equal(is_blue(img), water)
    assert not is_blue(ts.colorize(sea["height"])).any()


def test_sea_level_alone_floods_low_ground():
    h = ts.fbm(48, seed=1)
    assert np.array_equal(is_blue(ts.colorize(h, sea_level=0.4)), h < 0.4)


def test_water_map_values(sea):
    m = ts.water_map(sea["height"].shape, **{k: sea[k] for k in ("ocean", "lakes", "rivers")})
    assert m.dtype == np.uint8
    for value, layer in ((255, "ocean"), (170, "lakes"), (85, "rivers")):
        assert np.array_equal(m == value, sea[layer])


@pytest.mark.parametrize("size, res", [(32, 32), (64, 16), (48, 20)])
def test_wrapped_obj_tiles_meet_exactly(tmp_path, size, res):
    path = str(tmp_path / "t.obj")
    h = ts.fbm(size, seed=2, wrap=True)
    ts.save_obj(path, h, max_resolution=res, wrap=True)
    rows = [l.split() for l in open(path)]
    read = lambda tag: np.array([r[1:] for r in rows if r[0] == tag], dtype=float)  # noqa: E731
    v, vn = read("v"), read("vn")
    n = int(round(np.sqrt(len(v))))
    assert n <= res + 1
    v, vn = v.reshape(n, n, 3), vn.reshape(n, n, 3)
    for a in (v[..., 1], vn[..., 0], vn[..., 1], vn[..., 2]):
        assert np.array_equal(a[:, 0], a[:, -1]) and np.array_equal(a[0], a[-1])
    assert v[..., 0].min() == 0 and v[..., 0].max() == 1
