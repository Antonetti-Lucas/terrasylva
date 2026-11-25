import numpy as np
import pytest

import terrasylva as ts
from terrasylva import hydrology as hy
from terrasylva.cli import main as cli_main
from terrasylva.fluvial import _DIST, _OX, _OY, _drainage, plate_collision

STEPS = [ts.Noise(), ts.Continents(), ts.Ranges(), ts.Ocean(), ts.Fluvial(steps=30),
         ts.Hydraulic(capacity=0.5), ts.Rivers(carve=False)]


@pytest.fixture(scope="module", params=[True, False], ids=["looping", "bordered"])
def world(request):
    return ts.Pipeline(STEPS).run(ts.World(size=256, seed=3, wrap=request.param))


def routing(h, sinks, wrap):
    out = hy._outlet_mask(h, sinks, wrap)
    filled, parent, order = hy._flood(h, out, wrap, _OX, _OY)
    return out, filled, hy._receivers(h, filled, parent, out, _OX, _OY, _DIST), order


def seam_ratio(h):
    seam = (np.abs(h[:, 0] - h[:, -1]).mean() + np.abs(h[0] - h[-1]).mean()) / 2
    inside = (np.abs(np.diff(h, axis=0)).mean() + np.abs(np.diff(h, axis=1)).mean()) / 2
    return seam / inside


# ------------------------------------------------------------- stream power
@pytest.mark.parametrize("wrap", [True, False])
@pytest.mark.parametrize("n", [8, 33, 100, 200])
def test_shapes_range_and_base_level(n, wrap):
    sinks = np.zeros((n, n), bool)
    sinks[:2] = True
    h = ts.stream_power(np.ones((n, n)), outlets=sinks, steps=8, wrap=wrap)
    assert h.shape == (n, n) and np.isfinite(h).all() and h.min() >= 0
    assert (h[sinks] == 0).all() and h.max() > 0


def test_deterministic_and_seeded():
    u = np.ones((64, 64))
    a = ts.stream_power(u, seed=1, steps=10)
    assert np.array_equal(a, ts.stream_power(u, seed=1, steps=10))
    assert not np.array_equal(a, ts.stream_power(u, seed=2, steps=10))


def test_input_is_not_modified():
    u = np.ones((32, 32))
    ts.stream_power(u, steps=5)
    assert (u == 1).all()


def test_faster_uplift_and_harder_rock_stand_higher():
    u = np.ones((96, 96))
    u[:, 48:] = 4.0
    h = ts.stream_power(u, steps=40)
    assert h[10:-10, 58:86].mean() > 1.5 * h[10:-10, 10:38].mean()
    rock = np.ones((96, 96))
    rock[:, 48:] = 4.0
    h = ts.stream_power(np.ones((96, 96)), hardness=rock, steps=40)
    assert h[10:-10, 58:86].mean() > 1.5 * h[10:-10, 10:38].mean()


def test_more_rain_cuts_deeper():
    wet = np.full((96, 96), 4.0)
    u = np.ones((96, 96))
    assert ts.stream_power(u, rain=wet, steps=40).mean() < 0.8 * ts.stream_power(u, steps=40).mean()


def test_no_uplift_stays_flat():
    h = ts.stream_power(np.zeros((64, 64)), steps=30)
    assert h.max() < 1e-3


def test_split_drainage_conserves_rain():
    for wrap in (True, False):
        h = ts.fbm(64, seed=2, wrap=wrap).astype(np.float64)
        out, filled, recv, order = routing(h, h < 0.3, wrap)
        acc = _drainage(h, filled, recv, order, np.ones_like(h), 1.5, _OX, _OY, _DIST)
        assert float(acc.reshape(h.shape)[out].sum()) == pytest.approx(h.size)


@pytest.mark.parametrize("bad", [dict(uplift=np.ones((8, 9))), dict(uplift=np.ones((4, 4))),
                                 dict(uplift=-np.ones((16, 16))),
                                 dict(uplift=np.full((16, 16), np.nan)),
                                 dict(steps=0), dict(smoothing=0.5), dict(concavity=1.0),
                                 dict(spread=0), dict(hardness=np.zeros((16, 16))),
                                 dict(rain=np.ones((8, 8)))])
def test_stream_power_validates(bad):
    kw = {"uplift": np.ones((16, 16)), **bad}
    with pytest.raises(ValueError):
        ts.stream_power(kw.pop("uplift"), **kw)


# ------------------------------------------------------------------ realism
@pytest.mark.parametrize("concavity", [0.3, 0.45, 0.6])
def test_slope_falls_with_drainage_as_the_law_says(concavity):
    """In balance with even uplift, slope ~ area ** -concavity."""
    h = ts.stream_power(np.ones((192, 192)), concavity=concavity)
    _, filled, recv, _ = routing(h, None, False)
    n = len(h)
    i = np.flatnonzero((recv >= 0) & (filled <= h).ravel())
    dist = np.hypot(i // n - recv[i] // n, i % n - recv[i] % n)
    slope = (h.ravel()[i] - h.ravel()[recv[i]]) / dist
    area = ts.flow_accumulation(h).ravel()[i]
    x, y = [], []
    edges = np.logspace(1.3, 3, 7)
    for lo, hi in zip(edges, edges[1:]):
        m = (area >= lo) & (area < hi)
        if m.sum() > 20:
            x.append(np.log(np.sqrt(lo * hi)))
            y.append(np.log(slope[m].mean()))
    assert -np.polyfit(x, y, 1)[0] == pytest.approx(concavity, abs=0.12)


def test_landscape_drains_almost_without_pits():
    h = ts.stream_power(np.ones((128, 128)))
    assert (ts.fill_depressions(h) > h).mean() < 0.01


def test_fewer_pits_than_classic_erosion_on_average():
    def pits(w):
        h = w["height"].astype(np.float64)
        f = ts.fill_depressions(h, outlets=w["ocean"], wrap=True)
        return ((f > h) & ~w["ocean"]).sum() / (~w["ocean"]).sum()

    new, old = [], []
    for seed in (1, 2, 3):
        start = ts.World(size=128, seed=seed, wrap=True)
        new.append(pits(ts.default_pipeline().run(start, stop="hydraulic")))
        old.append(pits(ts.classic_pipeline().run(start, stop="ocean")))
    assert np.mean(new) < 0.6 * np.mean(old)


def test_channels_are_not_locked_to_the_grid():
    h = ts.stream_power(np.ones((192, 192)), steps=40, outlets=_frame(192))
    _, _, recv, order = routing(h, _frame(192), False)
    n = len(h)
    step = recv - np.arange(recv.size)
    run = np.zeros(recv.size, int)
    acc = ts.flow_accumulation(h).ravel()
    for c in order:
        q = recv[c]
        if q >= 0:
            run[c] = run[q] + 1 if step[q] == step[c] else 1
    assert (run[acc > 60] >= 12).mean() < 0.08


def _frame(n):
    m = np.ones((n, n), bool)
    m[1:-1, 1:-1] = False
    return m


# ------------------------------------------------------------------- steps
def test_coast_and_sea_floor_stay_put(world):
    before = ts.Pipeline(STEPS[:4]).run(ts.World(size=256, seed=3, wrap=world.wrap))
    assert np.array_equal(before["ocean"], world["ocean"])
    assert np.array_equal(before["height"][before["ocean"]], world["height"][world["ocean"]])
    assert (world["height"][~world["ocean"]] > world.sea_level).all()
    assert world["height"].max() <= 1.0


def test_looping_has_no_seam():
    w = ts.Pipeline(STEPS[:5]).run(ts.World(size=128, seed=5, wrap=True))
    assert seam_ratio(w["height"]) < 1.8 and seam_ratio(w["uplift"]) < 1.8


@pytest.mark.parametrize("wrap", [True, False])
def test_works_without_ocean_and_height(wrap):
    w = ts.Pipeline([ts.Ranges(), ts.Fluvial(steps=10)]).run(ts.World(size=64, seed=1, wrap=wrap))
    assert np.isfinite(w["height"]).all() and np.ptp(w["height"]) > 0.3


def test_world_with_only_ocean_is_left_alone():
    w = ts.World(size=32, sea_level=0.5, wrap=True, layers={
        "height": np.full((32, 32), 0.2, np.float32), "ocean": np.ones((32, 32), bool)})
    out = ts.Pipeline([ts.Ranges(raise_land=0, coast=0), ts.Fluvial(steps=5)]).run(w)
    assert np.isfinite(out["height"]).all()


@pytest.mark.parametrize("step", [ts.Ranges(plates=1), ts.Ranges(width=0), ts.Ranges(coast=-1),
                                  ts.Fluvial(hardness=2), ts.Fluvial(lowlands=1),
                                  ts.Fluvial(max_slope=0)])
def test_steps_validate(step):
    with pytest.raises(ValueError):
        ts.Pipeline([ts.Ranges(name="first"), step]).run(ts.World(size=32))


def test_fluvial_needs_uplift():
    with pytest.raises(ValueError, match="Ranges"):
        ts.Pipeline([ts.Fluvial()]).run(ts.World(size=32))


def test_plates_loop_and_stay_in_range():
    kw = dict(seed=4, plates=6, width=0.08)
    a = plate_collision(96, wrap=True, **kw)
    assert a.min() >= 0 and a.max() == 1.0 and seam_ratio(a) < 2.0
    assert not np.array_equal(a, plate_collision(96, wrap=True, **{**kw, "seed": 5}))


def test_erosion_after_ocean_respects_the_coast():
    w = ts.Pipeline([ts.Noise(), ts.Continents(), ts.Ocean()]).run(ts.World(size=96, seed=2, wrap=True))
    out = ts.Pipeline([ts.Hydraulic(), ts.Thermal(talus=0.1)]).run(w)
    sea = w["ocean"]
    assert np.array_equal(out["height"][sea], w["height"][sea])
    land = ~sea & (w["height"] > w.sea_level)
    assert (out["height"][land] >= np.float32(w.sea_level)).all()
    assert not np.array_equal(out["height"], w["height"])


def test_recipe_and_cli(tmp_path):
    path = str(tmp_path / "r.json")
    ts.save_recipe(path, ts.World(size=64, wrap=True), ts.default_pipeline())
    assert ts.load_recipe(path)[1].steps == ts.default_pipeline().steps
    out = str(tmp_path / "o")
    assert cli_main(["generate", "--size", "64", "--wrap", "--fluvial", "--biomes", "-o", out]) == 0
    kinds = [type(s).__name__ for s in ts.load_recipe(out + "/recipe.json")[1].steps]
    assert kinds == ["Noise", "Continents", "Ranges", "Ocean", "Fluvial", "Hydraulic",
                     "Climate", "Rivers", "Biomes"]


@pytest.mark.parametrize("max_slope", [0.3, 0.65])
def test_slopes_stay_near_the_limit(max_slope):
    w = ts.Pipeline([*STEPS[:4], ts.Fluvial(steps=20, max_slope=max_slope)]).run(
        ts.World(size=128, seed=3, wrap=True))
    s = np.tan(ts.slope(w["height"], relief=w.relief, wrap=True))[~w["ocean"]]
    assert np.quantile(s, 0.95) < 1.5 * max_slope
