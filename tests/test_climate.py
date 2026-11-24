import numpy as np
import pytest

import terrasylva as ts
from terrasylva.climate import Climate

LAYERS = ("temperature", "rainfall", "moisture")


def world_from(height, *, ocean=None, sea=None, wrap=False, seed=1):
    w = ts.World(size=height.shape[0], seed=seed, wrap=wrap, sea_level=sea)
    w["height"] = height.astype(np.float32)
    if ocean is not None:
        w["ocean"] = ocean
    return w


def run(world, **kw):
    out = ts.Pipeline([Climate(**kw)]).run(world)
    return out


@pytest.fixture(scope="module")
def sea_world():
    steps = [ts.Noise(), ts.Continents(), ts.Thermal(iterations=5), ts.Hydraulic(drops=1500),
             ts.Ocean(), Climate()]
    return ts.Pipeline(steps).run(ts.World(size=96, seed=3, wrap=True))


def ridge_world(n=128, margin=24):
    """Sea at both sides of a land strip with a north-south ridge in the middle."""
    x = np.arange(n)[None, :] * np.ones((n, 1))
    ridge = 0.25 + 0.75 * np.exp(-0.5 * ((x - n / 2 + 0.5) / 7.0) ** 2)
    sea = np.zeros((n, n), dtype=bool)
    sea[:, :margin] = sea[:, n - margin:] = True
    h = np.where(sea, 0.05, ridge)
    return world_from(h, ocean=sea, sea=0.2)


def west_east(rain, n=128):
    """Mean rain on the west and east foothills of the symmetric ridge."""
    return rain[:, 30:52].mean(), rain[:, n - 52:n - 30].mean()


# --------------------------------------------------------------- basic output
def test_layers_shapes_types_and_ranges(sea_world):
    t, r, m = (sea_world[k] for k in LAYERS)
    assert all(a.shape == (96, 96) and a.dtype == np.float32 for a in (t, r, m))
    assert all(np.isfinite(a).all() for a in (t, r, m))
    assert r.min() >= 0 and 0 <= m.min() and m.max() <= 1
    assert t.min() < 0 < t.max()


def test_deterministic_and_seed_dependent():
    def make(seed):
        steps = [ts.Noise(), ts.Continents(), ts.Hydraulic(drops=500), ts.Ocean(), Climate()]
        return ts.Pipeline(steps).run(ts.World(size=64, seed=seed, wrap=True))
    a, b, c = make(5), make(5), make(6)
    for k in LAYERS:
        assert np.array_equal(a[k], b[k])
    assert not np.array_equal(a["temperature"], c["temperature"])


def test_rerunning_ignores_stale_layers(sea_world):
    again = ts.Pipeline([Climate()]).run(sea_world)
    for k in LAYERS:
        assert np.array_equal(again[k], sea_world[k])
    # a leftover lakes layer from an older run must not matter
    stale = ts.Pipeline([Climate()]).run(sea_world)
    stale["lakes"] = np.ones((96, 96), dtype=bool)
    stale = ts.Pipeline([Climate()]).run(stale)
    assert np.array_equal(stale["rainfall"], sea_world["rainfall"])


def test_rainfall_has_mean_one_over_land(sea_world):
    land = ~sea_world["ocean"]
    assert land.any() and (~land).any()
    assert sea_world["rainfall"][land].astype(np.float64).mean() == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------- temperature
@pytest.mark.parametrize("wrap", [True, False])
def test_temperature_band_goes_north_south(wrap):
    n = 64
    warm, cold = 32.0, -8.0
    w = run(world_from(np.full((n, n), 0.1), sea=0.1, wrap=wrap), wobble=0.0,
            warm_temp=warm, cold_temp=cold)
    t = w["temperature"]
    assert np.allclose(t, t[:, :1])                  # flat world: one value per row
    if wrap:
        assert t[:, 0].max() == pytest.approx(warm, abs=0.01)   # warm band at the middle
        assert t[:, 0].min() == pytest.approx(cold, abs=0.2)
        assert t[n // 2, 0] == pytest.approx(warm)
        step = np.abs(np.diff(t[:, 0])).max()
        assert abs(t[0, 0] - t[-1, 0]) <= step + 1e-4          # continuous across the seam
    else:
        assert t[0, 0] == pytest.approx(cold) and t[-1, 0] == pytest.approx(warm)
        assert (np.diff(t[:, 0]) > 0).all()


def test_wobble_bends_the_bands_and_keeps_them_looping():
    n = 96
    flat = world_from(np.full((n, n), 0.1), sea=0.1, wrap=True)
    t = run(flat, warm_temp=30.0, cold_temp=-10.0)["temperature"]
    assert t[n // 4].std() > 1.0                       # bands are not straight any more
    assert t.min() >= -10.0 - 1e-4 and t.max() <= 30.0 + 1e-4
    for axis in (0, 1):                           # no seam in either direction
        seam = np.abs(np.take(t, 0, axis) - np.take(t, -1, axis)).mean()
        assert seam < 2.0 * np.abs(np.diff(t, axis=axis)).mean()
    assert np.array_equal(t, run(flat, warm_temp=30.0, cold_temp=-10.0)["temperature"])
    plain = run(flat, warm_temp=30.0, cold_temp=-10.0, wobble=0.0)["temperature"]
    assert np.abs(t - plain).max() > 3.0


def test_temperature_drops_with_altitude_by_the_lapse_rate():
    n = 48
    sea = 0.3
    flat = run(world_from(np.full((n, n), sea), sea=sea), wobble=0.0)["temperature"]
    for lapse in (25.0, 10.0):
        peak = np.full((n, n), sea)
        peak[10, 10], peak[20, 20] = 1.0, 0.65
        t = run(world_from(peak, sea=sea), wobble=0.0, lapse=lapse)["temperature"]
        assert flat[10, 10] - t[10, 10] == pytest.approx(lapse, abs=1e-4)      # full drop at height 1
        assert flat[20, 20] - t[20, 20] == pytest.approx(lapse * 0.5, abs=1e-4)
        assert np.allclose(np.delete(t, [10 * n + 10, 20 * n + 20]),
                           np.delete(flat, [10 * n + 10, 20 * n + 20]), atol=1e-5)


def test_it_is_colder_up_high_on_real_terrain(sea_world):
    h, t = sea_world["height"], sea_world["temperature"]
    land = ~sea_world["ocean"]
    rows = np.arange(96)[:, None] * np.ones((1, 96))
    for r in (24, 48, 72):                # compare within one latitude band
        band = land & (np.abs(rows - r) < 4)
        hi, lo = band & (h > np.quantile(h[band], 0.8)), band & (h < np.quantile(h[band], 0.2))
        assert t[hi].mean() < t[lo].mean() - 2


# -------------------------------------------------------------------- looping
@pytest.mark.parametrize("wind", [270.0, 225.0, 0.0])
def test_wrapped_climate_commutes_with_shifting(wind):
    n = 64
    h = ts.fbm(n, seed=9, scale=3, wrap=True)
    ocean = h < np.quantile(h, 0.4)
    kw = dict(wind=wind, wobble=0.0)
    a = run(world_from(h, ocean=ocean, sea=float(np.quantile(h, 0.4)), wrap=True), **kw)
    for shift in ((13, 0), (0, 29), (13, 29)):
        sh = tuple(np.roll(x, shift, axis=(0, 1)) for x in (h, ocean))
        b = run(world_from(sh[0], ocean=sh[1], sea=float(np.quantile(h, 0.4)), wrap=True), **kw)
        assert np.allclose(np.roll(a["rainfall"], shift, axis=(0, 1)), b["rainfall"],
                           rtol=1e-4, atol=1e-5), (wind, shift)
        assert np.allclose(np.roll(a["moisture"], shift, axis=(0, 1)), b["moisture"], atol=1e-5)
    # temperature only depends on the row, so shifting along x leaves it alone
    sh = np.roll(h, 13, axis=1)
    b = run(world_from(sh, ocean=np.roll(ocean, 13, axis=1), sea=float(np.quantile(h, 0.4)),
                       wrap=True), **kw)
    assert np.allclose(np.roll(a["temperature"], 13, axis=1), b["temperature"], atol=1e-5)


def test_wrapped_rain_from_a_reduced_grid_still_commutes_with_whole_block_shifts():
    n, grid = 192, 64                     # 3x3 blocks
    h = ts.fbm(n, seed=2, scale=3, wrap=True)
    ocean = h < np.quantile(h, 0.4)
    sea = float(np.quantile(h, 0.4))
    a = run(world_from(h, ocean=ocean, sea=sea, wrap=True), grid=grid)["rainfall"]
    shift = (3 * 7, 3 * 11)
    b = run(world_from(np.roll(h, shift, (0, 1)), ocean=np.roll(ocean, shift, (0, 1)), sea=sea,
                       wrap=True), grid=grid)["rainfall"]
    assert np.allclose(np.roll(a, shift, (0, 1)), b, rtol=1e-4, atol=1e-5)
    assert (np.abs(a[:, 0] - a[:, -1]).mean() < 1.5 * np.abs(np.diff(a, axis=1)).mean())


def test_unwrapped_and_wrapped_rain_are_different_on_the_same_land():
    h = ts.fbm(48, seed=4, scale=3, wrap=True)
    ocean = h < np.quantile(h, 0.5)
    sea = float(np.quantile(h, 0.5))
    a = run(world_from(h, ocean=ocean, sea=sea, wrap=True))["rainfall"]
    b = run(world_from(h, ocean=ocean, sea=sea, wrap=False))["rainfall"]
    assert not np.allclose(a, b)


# ----------------------------------------------------------------- rain shadow
def test_lee_side_of_a_mountain_is_dry():
    w = run(ridge_world(), wind=270.0)
    west, east = west_east(w["rainfall"])
    assert west > 1.5 * east


def test_reversing_the_wind_reverses_the_shadow():
    n = 128
    a = run(ridge_world(n), wind=270.0)["rainfall"]
    b = run(ridge_world(n), wind=90.0)["rainfall"]
    west_a, east_a = west_east(a)
    west_b, east_b = west_east(b)
    assert west_a > 1.5 * east_a and east_b > 1.5 * west_b
    # the world is mirror-symmetric, so the two results are mirror images
    assert np.allclose(a, b[:, ::-1], rtol=1e-3, atol=1e-4)


def test_north_south_wind_makes_a_shadow_too():
    n = 128
    y = np.arange(n)[:, None] * np.ones((1, n))
    ridge = 0.25 + 0.75 * np.exp(-0.5 * ((y - n / 2 + 0.5) / 7.0) ** 2)
    sea = np.zeros((n, n), dtype=bool)
    sea[:24] = sea[n - 24:] = True
    w = world_from(np.where(sea, 0.05, ridge), ocean=sea, sea=0.2)
    rain = run(w, wind=0.0)["rainfall"]          # from the north: air moves to higher rows
    assert rain[30:52].mean() > 1.5 * rain[n - 52:n - 30].mean()


# -------------------------------------------------------------- flow and rivers
@pytest.mark.parametrize("wrap", [True, False])
def test_rain_is_conserved_by_the_drainage(wrap):
    steps = [ts.Noise(), ts.Continents(), ts.Hydraulic(drops=800), ts.Ocean(), Climate(),
             ts.Rivers()]
    w = ts.Pipeline(steps).run(ts.World(size=96, seed=8, wrap=wrap))
    outlets = w["ocean"].copy()
    if not wrap:
        outlets[0, :] = outlets[-1, :] = outlets[:, 0] = outlets[:, -1] = True
    rain = w["rainfall"].astype(np.float64)
    assert w["flow"][outlets].astype(np.float64).sum() == pytest.approx(rain.sum(), rel=1e-4)


def test_rivers_follow_rainfall_when_the_layer_exists():
    base = ts.Pipeline([ts.Noise(), ts.Continents(), ts.Hydraulic(drops=800),
                        ts.Ocean()]).run(ts.World(size=96, seed=8, wrap=True))
    plain = ts.Pipeline([ts.Rivers(carve=False)]).run(base)
    wet = ts.Pipeline([Climate(), ts.Rivers(carve=False)]).run(base)
    assert not np.array_equal(plain["flow"], wet["flow"])
    uniform = base.layers.copy()
    uniform["rainfall"] = np.ones((96, 96), dtype=np.float32)
    same = ts.Pipeline([ts.Rivers(carve=False)]).run(ts.World(size=96, seed=8, wrap=True,
                                                              sea_level=base.sea_level,
                                                              layers=uniform))
    assert np.array_equal(same["flow"], plain["flow"])


@pytest.mark.parametrize("bad", [np.nan, -1.0])
def test_rivers_reject_a_broken_rainfall_layer(bad):
    w = ts.World(size=32, seed=1)
    w["height"] = ts.fbm(32, seed=1)
    r = np.ones((32, 32), dtype=np.float32)
    r[3, 3] = bad
    w["rainfall"] = r
    with pytest.raises(ValueError, match="rainfall"):
        ts.Rivers()(w)


# ------------------------------------------------------------ degenerate worlds
def test_world_without_an_ocean():
    w = ts.Pipeline([ts.Noise(), ts.Hydraulic(drops=300), Climate()]).run(
        ts.World(size=64, seed=2, wrap=True))
    assert all(np.isfinite(w[k]).all() for k in LAYERS)
    assert w["rainfall"].astype(np.float64).mean() == pytest.approx(1.0, abs=1e-4)


def test_flat_land_gets_even_rain_when_looping():
    w = run(world_from(np.full((40, 40), 0.5), wrap=True))
    assert np.allclose(w["rainfall"], 1.0)
    assert np.isfinite(w["temperature"]).all() and np.isfinite(w["moisture"]).all()
    assert np.allclose(w["moisture"], 0.5)


def test_flat_land_without_looping_is_wettest_where_the_air_comes_in():
    w = run(world_from(np.full((40, 40), 0.5), wrap=False))        # wind from the west
    rain = w["rainfall"]
    assert np.isfinite(rain).all() and rain.mean() == pytest.approx(1.0, abs=1e-4)
    assert (np.diff(rain[20]) < 0).all()


@pytest.mark.parametrize("wrap", [True, False])
def test_all_ocean_world(wrap):
    w = world_from(np.full((40, 40), 0.1), ocean=np.ones((40, 40), dtype=bool), sea=0.5, wrap=wrap)
    w = run(w)
    assert all(np.isfinite(w[k]).all() for k in LAYERS)
    assert w["rainfall"].mean() == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("wrap", [True, False])
def test_flat_land_beside_a_flat_sea(wrap):
    sea = np.zeros((40, 40), dtype=bool)
    sea[:, :10] = True
    h = np.where(sea, 0.1, 0.4)
    w = run(world_from(h, ocean=sea, sea=0.3, wrap=wrap))
    assert all(np.isfinite(w[k]).all() for k in LAYERS)
    assert w["rainfall"][~sea].mean() == pytest.approx(1.0, abs=1e-4)
    rain = w["rainfall"]
    assert rain[:, 10:16].mean() > rain[:, 30:36].mean()      # wetter next to the sea


def test_tiny_and_unusual_sizes_work():
    for n, grid in ((8, 8), (50, 16), (257, 256)):
        w = run(world_from(ts.fbm(n, seed=1, scale=2), wrap=False, sea=0.4), grid=grid)
        assert all(np.isfinite(w[k]).all() for k in LAYERS), n


# ----------------------------------------------------------- parameters, recipes
@pytest.mark.parametrize("kw", [{"rain_out": 0.0}, {"evaporation": -1.0}, {"orographic": -1.0},
                                {"lapse": -5.0}, {"grid": 4}, {"wind": float("nan")},
                                {"contrast": 0.0}, {"base": -0.1}, {"spread": -1.0}])
def test_climate_validates(kw):
    with pytest.raises(ValueError):
        run(world_from(ts.fbm(32, seed=1)), **kw)


def test_climate_in_a_recipe_and_in_the_default_pipeline(tmp_path):
    p = ts.Pipeline([ts.Noise(), Climate(wind=45.0, lapse=18.0, grid=64), ts.Rivers()])
    path = tmp_path / "r.json"
    ts.save_recipe(str(path), ts.World(size=32, seed=1), p)
    _, loaded = ts.load_recipe(str(path))
    assert loaded.steps[1] == Climate(wind=45.0, lapse=18.0, grid=64)
    names = ts.default_pipeline().names
    assert names.index("ocean") < names.index("climate") < names.index("rivers")


@pytest.mark.parametrize("wrap", [False, True])
def test_single_cell_roughness_does_not_speckle_the_rain(wrap):
    rng = np.random.default_rng(0)
    h = np.full((96, 96), 0.5)
    h[:, :30] = 0.1
    rough = h + rng.uniform(-0.02, 0.02, h.shape)
    rain = []
    for height in (h, rough):
        w = ts.World(size=96, wrap=wrap, sea_level=0.3,
                     layers={"height": height.astype(np.float32), "ocean": h < 0.3})
        rain.append(ts.Pipeline([ts.Climate(wobble=0)]).run(w)["rainfall"][:, 45:85])
    assert np.abs(rain[1] - rain[0]).mean() < 0.2 * rain[0].mean()
    step = np.abs(np.diff(rain[1], axis=0)).mean()
    assert step < 0.06 * rain[1].mean()
