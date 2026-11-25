import json
import os

import numpy as np
import pytest
from PIL import Image

import terrasylva as ts
from terrasylva import biomes as bm
from terrasylva.cli import main as cli_main

N = 48


def layers_world(temperature, moisture, height=None, ocean=None, sea=0.2, wrap=False,
                 height_range=None, **extra):
    """A world with hand-made climate layers, ready for the Biomes step."""
    n = np.shape(temperature)[0]
    w = ts.World(size=n, seed=4, wrap=wrap, sea_level=sea, height_range=height_range)
    w["height"] = np.full((n, n), 0.5, np.float32) if height is None else height.astype(np.float32)
    w["temperature"] = np.asarray(temperature, np.float32)
    w["moisture"] = np.asarray(moisture, np.float32)
    if ocean is not None:
        w["ocean"] = ocean
    for k, v in extra.items():
        w[k] = v
    return w


def const(v, n=N):
    return np.full((n, n), v, np.float32)


def run(world, **kw):
    return ts.Pipeline([ts.Biomes(**kw)]).run(world)


def name_at(world, table, y, x):
    return table[int(world["biome"][y, x])].name


@pytest.fixture(scope="module")
def full():
    steps = [ts.Noise(), ts.Continents(), ts.Thermal(iterations=5), ts.Hydraulic(drops=1500),
             ts.Ocean(), ts.Climate(), ts.Rivers(), ts.Biomes()]
    return ts.Pipeline(steps).run(ts.World(size=96, seed=3, wrap=True))


# ------------------------------------------------------------------ outputs
def test_layers_shapes_types_and_sums(full):
    table = ts.default_biome_table()
    b, w, s = full["biome"], full["biome_weights"], full["splat"]
    assert b.shape == (96, 96) and b.dtype == np.uint8 and b.max() < len(table)
    assert w.shape == (96, 96, len(table)) and w.dtype == np.float32
    assert np.isfinite(w).all() and w.min() >= 0
    assert np.allclose(w.sum(axis=-1), 1.0, atol=1e-5)
    assert s.shape == (96, 96, 4) and s.dtype == np.uint8
    assert (s.sum(axis=-1, dtype=np.int64) == 255).all()
    assert np.array_equal(b, w.argmax(axis=-1))


def test_default_table_has_the_requested_biomes():
    names = [b.name for b in ts.default_biome_table()]
    assert set(names) == {"ocean", "beach", "desert", "savanna", "grassland", "forest",
                          "rainforest", "taiga", "tundra", "snow", "rock"}


def test_default_pipeline_ends_with_biomes():
    w = ts.default_pipeline().run(ts.World(size=64, seed=2, wrap=True))
    assert "biome" in w and ts.default_pipeline().names[-1] == "biomes"
    assert ts.default_pipeline().names.index("climate") < ts.default_pipeline().names.index("rivers")


def test_deterministic_and_the_seed_changes_the_noise():
    t, m = np.linspace(-20, 35, N)[None, :] * np.ones((N, 1)), const(0.5)
    a, b = run(layers_world(t, m)), run(layers_world(t, m))
    for k in ("biome", "biome_weights", "splat"):
        assert np.array_equal(a[k], b[k])
    other = layers_world(t, m)
    other.seed = 5
    assert not np.array_equal(run(other)["biome_weights"], a["biome_weights"])
    assert np.array_equal(run(other, noise=0.0)["biome_weights"],
                          run(layers_world(t, m), noise=0.0)["biome_weights"])


def test_rerunning_from_biomes_gives_the_same_result(full):
    again = ts.Pipeline([ts.Biomes()]).run(full)
    assert np.array_equal(again["biome"], full["biome"])
    assert np.array_equal(again["splat"], full["splat"])


def test_biomes_need_the_climate_layers():
    w = ts.World(size=16)
    w["height"] = np.zeros((16, 16), np.float32)
    with pytest.raises(ValueError, match="Climate"):
        ts.Biomes()(w)


# --------------------------------------------------------------- the default table
@pytest.mark.parametrize("t, m, expected", [
    (-25.0, 0.5, "snow"), (30.0, 0.05, "desert"), (28.0, 0.9, "rainforest"),
    (28.0, 0.35, "savanna"), (12.0, 0.7, "forest"), (12.0, 0.3, "grassland"),
    (-2.0, 0.8, "taiga"), (-2.0, 0.1, "tundra"),
])
def test_default_table_places_the_classic_climates(t, m, expected):
    w = run(layers_world(const(t), const(m)), noise=0.0)
    assert {name_at(w, ts.default_biome_table(), y, x) for y in (5, 25) for x in (5, 25)} == {expected}


def test_ocean_cells_are_ocean_and_land_never_is():
    ocean = np.zeros((N, N), bool)
    ocean[:, :N // 2] = True
    h = np.where(ocean, 0.1, 0.6)
    w = run(layers_world(const(15), const(0.5), h, ocean), noise=0.0)
    names = [b.name for b in ts.default_biome_table()]
    sea = names.index("ocean")
    assert (w["biome"][ocean] == sea).all() and (w["biome"][~ocean] != sea).all()
    assert (w["splat"][ocean][:, 2] == 255).all()          # sea floor is sand


def test_steep_ground_is_rock_and_the_shore_is_a_beach():
    x = np.arange(N)[None, :] * np.ones((N, 1))
    h = np.clip(0.26 + (x - 16) * 0.005, 0.26, 1.0)          # flat shore, then a 45 degree slope
    h[:, :4] = 0.1
    ocean = np.zeros((N, N), bool)
    ocean[:, :4] = True
    w = run(layers_world(const(20), const(0.5), h, ocean, sea=0.25, height_range=(0, 200)),
            noise=0.0)
    names = [b.name for b in ts.default_biome_table()]
    assert names[w["biome"][10, 5]] == "beach"
    assert names[w["biome"][10, 36]] == "rock"
    assert names[w["biome"][10, 12]] not in ("rock", "ocean")


def test_deep_ocean_can_be_told_apart_with_height_ranges():
    table = (ts.Biome("shallows", water=True, height=(-0.2, None)),
             ts.Biome("deep", water=True, height=(None, -0.2)), ts.Biome("land"))
    ocean = np.zeros((N, N), bool)
    ocean[:, :N // 2] = True
    floor = np.concatenate([np.linspace(0.0, 0.39, N // 2), np.full(N - N // 2, 0.6)])
    h = floor[None, :] * np.ones((N, 1))
    w = run(layers_world(const(15), const(0.5), h, ocean, sea=0.4), table=table, noise=0.0)
    assert w["biome"][5, 1] == 1 and w["biome"][5, N // 2 - 2] == 0 and w["biome"][5, 40] == 2


# ------------------------------------------------------------------ transitions
def test_weights_change_smoothly_across_a_gradient():
    t = np.linspace(-30, 40, 128)[None, :] * np.ones((128, 1))
    w = run(layers_world(t, const(0.5, 128)), noise=0.0)["biome_weights"]
    assert np.abs(np.diff(w, axis=1)).max() < 0.5
    assert np.abs(np.diff(w, axis=0)).max() < 1e-6          # nothing changes along the rows
    mixed = (w.max(axis=-1) < 0.9).mean()
    assert 0.05 < mixed < 0.6                               # real blending, but not everywhere


def test_softness_widens_the_transitions():
    t = np.linspace(-30, 40, 128)[None, :] * np.ones((128, 1))
    narrow = run(layers_world(t, const(0.5, 128)), noise=0.0, softness=0.1)["biome_weights"]
    wide = run(layers_world(t, const(0.5, 128)), noise=0.0, softness=0.6)["biome_weights"]
    assert (wide.max(axis=-1) < 0.9).mean() > (narrow.max(axis=-1) < 0.9).mean()


def test_noise_moves_the_borders_but_keeps_them_looping():
    t = (5 + 30 * np.cos(2 * np.pi * np.arange(96) / 96))[:, None] * np.ones((1, 96))
    a = run(layers_world(t, const(0.5, 96), wrap=True), noise=0.0)["biome_weights"]
    b = run(layers_world(t, const(0.5, 96), wrap=True), noise=0.8)["biome_weights"]
    assert not np.array_equal(a, b)
    seam = np.abs(b[:, 0] - b[:, -1]).mean()
    inside = np.abs(np.diff(b, axis=1)).mean()
    assert seam < 2.0 * inside
    seam = np.abs(b[0] - b[-1]).mean()
    inside = np.abs(np.diff(b, axis=0)).mean()
    assert seam < 2.0 * inside


def test_biomes_commute_with_shifting_on_a_looping_world():
    h = ts.fbm(64, seed=6, scale=3, wrap=True)
    ocean = h < np.quantile(h, 0.4)
    t = 15 + 25 * ts.fbm(64, seed=7, scale=2, wrap=True) - 12
    m = ts.fbm(64, seed=8, scale=3, wrap=True)
    rivers = ts.fbm(64, seed=9, scale=4, wrap=True) > 0.8
    sea = float(np.quantile(h, 0.4))

    def make(shift):
        r = lambda a: np.roll(a, shift, (0, 1))  # noqa: E731
        return layers_world(r(t), r(m), r(h), r(ocean), sea, True, rivers=r(rivers))

    a = run(make((0, 0)), noise=0.0)
    for shift in ((11, 0), (0, 23), (11, 23)):
        b = run(make(shift), noise=0.0)
        assert np.allclose(np.roll(a["biome_weights"], shift, (0, 1)), b["biome_weights"],
                           atol=1e-5), shift
        assert np.array_equal(np.roll(a["splat"], shift, (0, 1)), b["splat"])


def test_distance_to_water_loops():
    mask = np.zeros((32, 32), bool)
    mask[0, 0] = True
    for wrap in (True, False):
        d = bm._distance(mask, wrap)
        assert d[0, 0] == 0 and d[1, 0] == pytest.approx(1) and d[1, 1] == pytest.approx(1.4142, abs=1e-3)
        assert (d[31, 31] < 2) == wrap
    assert (bm._distance(np.zeros((8, 8), bool), True) > 1e8).all()


# ------------------------------------------------------- water raises moisture
def test_rivers_and_lakes_make_their_surroundings_wetter():
    table = (ts.Biome("dry", (200, 180, 100), moisture=(None, 0.3)),
             ts.Biome("wet", (30, 120, 50), moisture=(0.3, None)))
    n = 64
    water = np.zeros((n, n), bool)
    water[:, 10] = True
    lake = np.zeros((n, n), bool)
    lake[50:54, 50:54] = True
    base = layers_world(const(20, n), const(0.15, n), rivers=water, lakes=lake)
    w = run(base, table=table, noise=0.0, wet_reach=0.1)
    assert w["biome"][20, 11] == 1 and w["biome"][20, 45] == 0     # river bank vs far away
    assert w["biome"][50, 55] == 1 and w["biome"][20, 63] == 0     # lake shore vs far away
    off = run(base, table=table, noise=0.0, wet_bonus=0.0)
    assert (off["biome"] == 0).all()


# ----------------------------------------------------- custom and partial tables
def test_incomplete_table_still_covers_every_cell():
    table = (ts.Biome("cold", temperature=(None, 0)), ts.Biome("hot", temperature=(30, None)))
    t = np.linspace(-40, 60, N)[None, :] * np.ones((N, 1))
    w = run(layers_world(t, const(0.5)), table=table, noise=0.0)
    assert np.isfinite(w["biome_weights"]).all()
    assert np.allclose(w["biome_weights"].sum(-1), 1, atol=1e-5)
    row = w["biome"][0]
    assert row[0] == 0 and row[-1] == 1                              # ends are inside the ranges
    assert row[np.argmin(np.abs(t[0] - 8))] == 0 and row[np.argmin(np.abs(t[0] - 22))] == 1
    assert (np.diff(row.astype(int)) >= 0).all()                     # one switch, no flicker


def test_a_table_where_nothing_matches_and_a_single_biome():
    far = (ts.Biome("nowhere", temperature=(500, 600)),)
    w = run(layers_world(const(10), const(0.5)), table=far)
    assert (w["biome"] == 0).all() and np.allclose(w["biome_weights"], 1)
    only = run(layers_world(const(10), const(0.5)), table=(ts.Biome("everything"),))
    assert (only["biome"] == 0).all()


def test_water_biomes_and_land_biomes_fall_back_to_each_other():
    ocean = np.zeros((N, N), bool)
    ocean[:, :N // 2] = True
    land_only = (ts.Biome("a"), ts.Biome("b", temperature=(30, 40)))
    w = run(layers_world(const(10), const(0.5), ocean=ocean), table=land_only, noise=0.0)
    assert np.array_equal(w["biome"] == 0, np.ones((N, N), bool))   # no sea biome: land ones take the sea
    sea_only = run(layers_world(const(10), const(0.5), ocean=ocean),
                   table=(ts.Biome("sea", water=True),), noise=0.0)
    assert (sea_only["biome"] == 0).all()                           # no land biome: the sea one takes the land


def test_a_custom_table_gives_the_biomes_its_colours_and_splats():
    table = (ts.Biome("ice", (4, 5, 6), temperature=(None, 0), splat=(0, 0, 0, 2)),
             ts.Biome("sand", (1, 2, 3), splat=(0, 0, 5, 0)))
    t = np.where(np.arange(N)[None, :] < N // 2, -20.0, 20.0) * np.ones((N, 1))
    w = run(layers_world(t, const(0.5)), table=table, noise=0.0, softness=0.05)
    assert w["biome"][3, 2] == 0 and w["biome"][3, N - 3] == 1
    assert tuple(w["splat"][3, 2]) == (0, 0, 0, 255) and tuple(w["splat"][3, N - 3]) == (0, 0, 255, 0)


def test_splat_always_sums_to_255_even_for_awkward_weights():
    rng = np.random.default_rng(1)
    for k in (2, 4, 7):
        weights = rng.dirichlet(np.ones(k) * 0.4, size=(40, 40))
        for part in (weights, np.eye(k)[rng.integers(0, k, (40, 40))], np.full((8, 8, k), 1 / k)):
            q = bm._quantize_to_255(part)
            assert q.dtype == np.uint8 and (q.sum(axis=-1, dtype=np.int64) == 255).all()
            assert np.abs(q / 255.0 - part).max() <= 1 / 255 + 1e-9


# ----------------------------------------------------------------- recipes
def test_table_round_trips_through_json_recipes(tmp_path):
    table = (ts.Biome("sea", (10, 20, 30), water=True, splat=(0, 0, 1, 0)),
             ts.Biome("mesa", (200, 100, 50), temperature=(10, None), moisture=(None, 0.3),
                      slope=(5, 30), height=(0.1, 0.9), splat=(0.5, 0, 0.5, 0)),
             ts.Biome("moor", (90, 100, 60), moisture=[0.3, 0.8]))
    step = ts.Biomes(table=table, softness=0.4, noise=0.1, wet_bonus=0.3, wet_reach=0.05)
    path = tmp_path / "r.json"
    ts.save_recipe(str(path), ts.World(size=16, seed=3), ts.Pipeline([ts.Noise(), step]))
    text = path.read_text(encoding="utf-8")
    assert "Infinity" not in text and "NaN" not in text
    assert json.loads(text)["steps"][1]["table"][1]["name"] == "mesa"
    _, loaded = ts.load_recipe(str(path))
    got = loaded.steps[1]
    assert got == step and all(isinstance(b, ts.Biome) for b in got.table)
    assert got.table[1].temperature == (10.0, None) and got.table[2].moisture == (0.3, 0.8)
    ts.save_recipe(str(path), ts.World(size=16, seed=3), loaded)          # and back again
    assert ts.load_recipe(str(path))[1].steps[1] == step


def test_default_table_recipe_round_trip(tmp_path):
    p = ts.Pipeline([ts.Noise(), ts.Climate(), ts.Biomes()])
    path = tmp_path / "r.json"
    ts.save_recipe(str(path), ts.World(size=16), p)
    assert ts.load_recipe(str(path))[1].steps == p.steps


def test_lists_and_tuples_are_the_same_biome():
    assert ts.Biome("a", temperature=[0, 5], color=[1, 2, 3]) == ts.Biome("a", temperature=(0, 5),
                                                                          color=(1, 2, 3))
    assert hash(ts.Biome("a")) == hash(ts.Biome("a"))


@pytest.mark.parametrize("bad", [
    dict(name=""), dict(name="x", color=(1, 2)), dict(name="x", color=(1, 2, 300)),
    dict(name="x", temperature=(5, 1)), dict(name="x", temperature=5),
    dict(name="x", moisture=(float("nan"), 1)), dict(name="x", splat=(0, 0, 0, 0)),
    dict(name="x", splat=(-1, 1, 0, 0)), dict(name="x", splat=(1, 1)),
])
def test_biome_validates(bad):
    with pytest.raises(ValueError):
        ts.Biome(**bad)


@pytest.mark.parametrize("kw", [dict(table=()), dict(softness=0), dict(noise=-1),
                                dict(wet_bonus=-1), dict(wet_reach=float("nan")),
                                dict(table=(ts.Biome("a"), ts.Biome("a"))),
                                dict(table=tuple(ts.Biome(str(i)) for i in range(257)))])
def test_biomes_validate(kw):
    with pytest.raises(ValueError):
        ts.Biomes(**kw)


# -------------------------------------------------------- degenerate worlds
def test_worlds_without_ocean_only_ocean_and_flat():
    plain = ts.Pipeline([ts.Noise(), ts.Climate(), ts.Biomes()]).run(
        ts.World(size=48, seed=1, wrap=True))
    flat = ts.World(size=48, wrap=True)
    flat["height"] = np.full((48, 48), 0.4, np.float32)
    flat = ts.Pipeline([ts.Climate(), ts.Rivers(), ts.Biomes()]).run(flat)
    sea = ts.World(size=48, wrap=False, sea_level=0.5)
    sea["height"] = np.full((48, 48), 0.1, np.float32)
    sea["ocean"] = np.ones((48, 48), bool)
    sea = ts.Pipeline([ts.Climate(), ts.Biomes()]).run(sea)
    for w in (plain, flat, sea):
        assert np.isfinite(w["biome_weights"]).all()
        assert np.allclose(w["biome_weights"].sum(-1), 1, atol=1e-5)
        assert (w["splat"].sum(-1, dtype=np.int64) == 255).all()
    assert (sea["biome"] == 0).all()                                  # ocean is the first biome
    assert (plain["biome"] != 0).all()


# ----------------------------------------------------------------- previews
def test_colorize_by_biome_keeps_shading_and_water():
    h = ts.fbm(48, seed=2, scale=3)
    ocean = h < 0.4
    b = np.where(h > 0.6, 1, 0).astype(np.uint8)
    colors = [(200, 0, 0), (0, 200, 0)]
    img = ts.colorize(h, sea_level=0.4, ocean=ocean, biome=b, biome_colors=colors)
    assert img.shape == (48, 48, 3) and img.dtype == np.uint8
    land_low = ~ocean & (b == 0)
    assert img[land_low][:, 0].mean() > 3 * img[land_low][:, 1].mean()         # red land
    assert img[ocean][:, 2].mean() > img[ocean][:, 0].mean()                    # blue sea on top
    weights = np.stack([1 - b, b], axis=-1).astype(np.float32)
    smooth = ts.colorize(h, sea_level=0.4, ocean=ocean, biome=b, biome_colors=colors,
                         biome_weights=weights)
    assert np.array_equal(smooth, img)
    with pytest.raises(ValueError):
        ts.colorize(h, biome=b)
    with pytest.raises(ValueError):
        ts.colorize(h, biome_colors=colors)
    with pytest.raises(ValueError):
        ts.colorize(h, biome=b + 5, biome_colors=colors)


def test_climate_layer_previews(full):
    for kind in ("temperature", "rainfall", "moisture"):
        img = ts.colorize_layer(full[kind], kind, full["height"], ocean=full["ocean"], wrap=True)
        assert img.shape == (96, 96, 3) and img.dtype == np.uint8
    cold = ts.colorize_layer(np.array([[-20.0, 35.0]]), "temperature")[0]
    assert cold[0][2] > cold[0][0] and cold[1][0] > cold[1][2]               # blue then red
    with pytest.raises(ValueError):
        ts.colorize_layer(full["rainfall"], "wind")


# ------------------------------------------------------- exports and the CLI
def test_biome_exports(tmp_path, full):
    table = ts.default_biome_table()
    ts.export.save_biome_map(str(tmp_path / "b.png"), full["biome"])
    ts.export.save_splat_map(str(tmp_path / "s.png"), full["splat"])
    ts.export.save_biome_legend(str(tmp_path / "l.json"), ts.biome_legend(table))
    assert np.array_equal(np.asarray(Image.open(tmp_path / "b.png")), full["biome"])
    s = Image.open(tmp_path / "s.png")
    assert s.mode == "RGBA" and np.array_equal(np.asarray(s), full["splat"])
    legend = json.loads((tmp_path / "l.json").read_text(encoding="utf-8"))
    assert [b["name"] for b in legend["biomes"]] == [b.name for b in table]
    assert legend["biomes"][3]["index"] == 3 and legend["splat_channels"][3] == "snow"
    with pytest.raises(ValueError):
        ts.export.save_biome_map(str(tmp_path / "x.png"), full["biome"].astype(np.int32))


FILES = ("biomes.png", "splat.png", "biomes.json", "preview_biomes.png",
         "preview_temperature.png", "preview_rainfall.png")


def test_cli_biomes_writes_the_extra_files_and_run_does_too(tmp_path):
    out = tmp_path / "gen"
    cli_main(["generate", "--size", "64", "--seed", "3", "--land", "0.6", "--wrap", "--biomes",
              "--drops", "300", "-o", str(out)])
    assert all((out / f).exists() for f in FILES), os.listdir(out)
    legend = json.loads((out / "biomes.json").read_text(encoding="utf-8"))["biomes"]
    idx = np.asarray(Image.open(out / "biomes.png"))
    assert idx.shape == (64, 64) and idx.max() < len(legend)
    splat = np.asarray(Image.open(out / "splat.png"))
    assert splat.shape == (64, 64, 4) and (splat.sum(-1, dtype=np.int64) == 255).all()

    again = tmp_path / "run"
    cli_main(["run", str(out / "recipe.json"), "-o", str(again)])
    assert all((again / f).exists() for f in FILES)
    assert np.array_equal(np.asarray(Image.open(again / "biomes.png")), idx)


def test_cli_without_biomes_writes_none_of_them(tmp_path):
    cli_main(["generate", "--size", "48", "--seed", "3", "--land", "0.6", "--drops", "100",
              "-o", str(tmp_path)])
    assert not any((tmp_path / f).exists() for f in FILES)


def test_low_plains_far_from_the_sea_are_not_beach():
    n = 128
    h = np.full((n, n), 0.501, np.float32)
    h[:, :20] = 0.499
    w = ts.World(size=n, sea_level=0.5, layers={
        "height": h, "ocean": h < 0.5, "temperature": np.full((n, n), 20, np.float32),
        "moisture": np.full((n, n), 0.45, np.float32)})
    out = ts.Pipeline([ts.Biomes(noise=0)]).run(w)
    names = [b.name for b in ts.Biomes().table]
    beach = out["biome"] == names.index("beach")
    assert beach[:, 20].all() and not beach[:, 25:].any()
    img = ts.colorize(h, None, 0.5, None, ocean=w["ocean"]).astype(int)
    assert img[64, 20, 0] > img[64, 60, 0] + 20
