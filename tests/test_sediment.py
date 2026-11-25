import numpy as np
import pytest

import terrasylva as ts
from terrasylva import hydrology as hy
from terrasylva.hydrology import _DIST, _OX, _OY
from terrasylva.sediment import _label, _valley_floors, build_deltas

BASE = [ts.Noise(), ts.Continents(), ts.Ranges(), ts.Ocean()]
FULL = [*BASE, ts.SeaFloor(), ts.Basins(), ts.Fluvial(steps=25), ts.Hydraulic(capacity=0.5),
        ts.Climate(), ts.Rivers(carve=False), ts.Biomes()]


def seam_ratio(h):
    seam = (np.abs(h[:, 0] - h[:, -1]).mean() + np.abs(h[0] - h[-1]).mean()) / 2
    inside = (np.abs(np.diff(h, axis=0)).mean() + np.abs(np.diff(h, axis=1)).mean()) / 2
    return seam / inside


@pytest.fixture(scope="module", params=[True, False], ids=["looping", "bordered"])
def world(request):
    return ts.Pipeline(FULL).run(ts.World(size=256, seed=42, wrap=request.param))


def routed(h, outlets=None, wrap=False):
    out = hy._outlet_mask(h, outlets, wrap)
    filled, parent, order = hy._flood(h, out, wrap, _OX, _OY)
    return hy._receivers(h, filled, parent, out, _OX, _OY, _DIST), order


# ----------------------------------------------------------- whole worlds
def test_layers_are_finite_consistent_and_deterministic(world):
    again = ts.Pipeline(FULL).run(ts.World(size=256, seed=42, wrap=world.wrap))
    for name, layer in world.layers.items():
        assert np.isfinite(layer.astype(float)).all(), name
        assert np.array_equal(layer, again[name]), name
    h, ocean = world["height"], world["ocean"]
    assert 0 <= h.min() and h.max() <= 1
    assert (h[ocean] < world.sea_level).all()
    assert np.allclose(world["ocean_depth"][ocean], world.sea_level - h[ocean], atol=1e-6)
    assert not (ocean & world["lakes"]).any() and not (ocean & world["rivers"]).any()


def test_there_is_dry_land_below_the_sea(world):
    below = ~world["ocean"] & (world["height"] < world.sea_level)
    dry = below & ~world["lakes"]
    assert dry.sum() > 20
    # and it is not connected to the open sea: ground above sea level surrounds it
    grown = below.copy()
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            grown |= np.roll(below, (a, b), (0, 1))
    assert not (grown & world["ocean"]).any()


def test_all_rain_ends_in_the_sea_or_a_basin(world):
    out = hy._outlet_mask(world["height"], world["ocean"] | world["sinks"], world.wrap)
    assert float(world["flow"][out].sum()) == pytest.approx(float(world["rainfall"].sum()), rel=1e-4)


def test_basin_lake_is_smaller_than_the_spill_lake(world):
    full = ts.detect_lakes(world["height"], outlets=world["ocean"], wrap=world.wrap).mask
    assert world["lakes"].any()
    assert (world["lakes"] & ~full).sum() <= 0.02 * world["lakes"].sum() + 4
    assert world["lakes"].sum() < full.sum()


def test_looping_worlds_have_no_seam():
    w = ts.Pipeline(FULL[:7]).run(ts.World(size=192, seed=7, wrap=True))
    assert seam_ratio(w["height"]) < 1.8


def test_basins_survive_removing_the_ocean():
    w = ts.Pipeline([ts.Ranges(), ts.Basins(), ts.Fluvial(steps=10), ts.Rivers(carve=False)]).run(
        ts.World(size=96, seed=1, wrap=True))
    assert np.isfinite(w["height"]).all()


def test_without_basins_land_stays_above_the_sea():
    w = ts.Pipeline([*BASE, ts.SeaFloor(), ts.Fluvial(steps=15)]).run(
        ts.World(size=128, seed=3, wrap=True))
    assert "sinks" not in w
    assert (w["height"][~w["ocean"]] > w.sea_level).all()


def test_classic_pipeline_is_unaffected():
    w = ts.classic_pipeline().run(ts.World(size=64, seed=3, wrap=True))
    assert "sinks" not in w and "basin" not in w


# ---------------------------------------------------------- valley floors
def test_valley_floors_only_raise_and_still_drain():
    h = ts.stream_power(np.ones((128, 128)))
    recv, order = routed(h)
    share = ts.flow_accumulation(h).ravel().astype(np.float64) / h.size
    room = np.full(h.size, np.inf)
    new, filled = _valley_floors(h, recv, order, share, room, 1e-3, h.max(), 0.01, 1e-6)
    assert (new >= h).all() and filled.any() and np.array_equal(new > h, filled)
    # every cell still lies above the one it drained to, so nothing is dammed
    up = np.flatnonzero(recv >= 0)
    flat = filled.ravel()[up]
    assert (new.ravel()[up][flat] >= new.ravel()[recv[up]][flat]).all()
    gone, _ = _valley_floors(h, recv, order, share, np.zeros(h.size), 1e-3, h.max(), 0.01, 0.0)
    pits = ts.fill_depressions(h) > h
    assert (gone > h)[~pits].sum() < 0.01 * h.size      # no room, no sediment


def test_valley_floors_make_more_flat_ground():
    def flat(v):
        w = ts.Pipeline([*BASE, ts.Fluvial(steps=25, valleys=v, deltas=0)]).run(
            ts.World(size=256, seed=3, wrap=True))
        s = np.degrees(ts.slope(w["height"], relief=w.relief, wrap=True))[~w["ocean"]]
        return (s < 4).mean()

    assert flat(3.0) > 1.15 * flat(0.0)


# ----------------------------------------------------------------- deltas
def test_delta_builds_connected_land_of_the_right_size():
    h = np.full((64, 64), 0.6)
    h[:, 20:] = 0.45
    ocean = h < 0.5
    mouth = 32 * 64 + 19
    new, wet = build_deltas(h, ocean, 0.5, np.array([mouth]), np.array([60]),
                            np.ones((64, 64)), False)
    gained = ocean & ~wet
    assert gained.sum() == 60 and gained[32, 20]
    assert _label(gained, False, _OX, _OY)[1] == 1
    assert (new[gained] > 0.5).all() and (new[wet] < 0.5).all()
    assert np.array_equal(new[~ocean], h[~ocean])
    apron = wet & (new > h)
    assert apron.any()


def test_deltas_can_be_switched_off(world):
    w = ts.Pipeline([*BASE, ts.Fluvial(steps=10, deltas=0)]).run(
        ts.World(size=128, seed=42, wrap=world.wrap))
    before = ts.Pipeline(BASE).run(ts.World(size=128, seed=42, wrap=world.wrap))
    assert np.array_equal(w["ocean"], before["ocean"])
    d = ts.Pipeline([*BASE, ts.Fluvial(steps=10, deltas=4)]).run(
        ts.World(size=128, seed=42, wrap=world.wrap))
    assert d["ocean"].sum() < before["ocean"].sum()


# -------------------------------------------------------------- sea floor
@pytest.mark.parametrize("wrap", [True, False])
def test_sea_floor_is_shallow_near_land_and_keeps_the_coast(wrap):
    w = ts.Pipeline(BASE).run(ts.World(size=192, seed=3, wrap=wrap))
    out = ts.Pipeline([ts.SeaFloor()]).run(w)
    ocean = w["ocean"]
    assert np.array_equal(out["height"][~ocean], w["height"][~ocean])
    assert (out["height"][ocean] < w.sea_level).all() and out["height"].min() >= 0
    from terrasylva.biomes import _distance
    away = _distance(~ocean, wrap) / 192
    depth = out["ocean_depth"]
    assert depth[ocean & (away < 0.02)].mean() < 0.4 * depth[ocean & (away > 0.1)].mean()


def test_sea_floor_needs_an_ocean():
    with pytest.raises(ValueError, match="Ocean"):
        ts.Pipeline([ts.Noise(), ts.SeaFloor()]).run(ts.World(size=32))


# --------------------------------------------------------- terminal lakes
def bowl():
    y, x = np.mgrid[:65, :65]
    return np.hypot(y - 32, x - 32) / 46.0


def test_lake_area_balances_inflow_and_evaporation():
    h = bowl()
    sinks = np.zeros((65, 65), bool)
    sinks[32, 32] = True
    flow = np.zeros((65, 65))
    flow[32, 32] = 2400
    area = lambda e: (ts.terminal_lakes(h, flow, sinks, None, evaporation=e) > 0).sum()  # noqa: E731
    assert abs(area(12.0) - 200) <= 8
    assert area(48.0) < area(24.0) < area(12.0)
    assert area(1e9) == 0
    spill = ts.detect_lakes(h, min_depth=0, min_area=0).mask.sum()
    assert area(0.01) <= spill


def test_lake_is_level_and_surrounds_its_sink():
    h = bowl()
    sinks = np.zeros((65, 65), bool)
    sinks[32, 32] = True
    flow = np.full((65, 65), 0.0)
    flow[32, 32] = 3000
    d = ts.terminal_lakes(h, flow, sinks, None)
    wet = d > 0
    assert wet[32, 32] and np.ptp((h + d)[wet]) < 1e-6
    assert (h[~wet] >= (h + d)[wet].max() - 1e-6).all()


def test_no_lake_where_water_can_run_off():
    h = np.tile(np.linspace(1, 0, 32), (32, 1))
    sinks = np.zeros((32, 32), bool)
    sinks[16, 16] = True
    assert not ts.terminal_lakes(h, np.full((32, 32), 100.0), sinks, None).any()


# ---------------------------------------------------- deposition and base
def test_deposition_builds_up_the_foreland():
    u = np.full((128, 128), 0.02)
    u[:, 40:70] = 1.0
    bare, full = (ts.stream_power(u, deposition=d) for d in (0.0, 0.8))
    assert full[30:100, 85:115].mean() > 1.2 * bare[30:100, 85:115].mean()


def test_base_lowers_inner_outlets():
    sinks = np.zeros((64, 64), bool)
    sinks[28:36, 28:36] = True
    base = np.where(sinks, -0.5, 0.0)
    h = ts.stream_power(np.ones((64, 64)) * 0.1, outlets=sinks, base=base, steps=20)
    assert (h[sinks] == -0.5).all() and h.min() == -0.5


@pytest.mark.parametrize("kw", [dict(deposition=-1), dict(deposition=5),
                                dict(base=np.ones((16, 16))), dict(base=np.zeros((8, 8)))])
def test_stream_power_validates_new_arguments(kw):
    with pytest.raises(ValueError):
        ts.stream_power(np.ones((16, 16)), **kw)


@pytest.mark.parametrize("step", [ts.Basins(count=-1), ts.Basins(depth=0), ts.Basins(width=0),
                                  ts.SeaFloor(slope=0), ts.SeaFloor(rough=2),
                                  ts.Fluvial(valleys=-1), ts.Fluvial(deltas=-1),
                                  ts.Fluvial(deposition=3), ts.Rivers(evaporation=0)])
def test_steps_validate(step):
    w = ts.Pipeline(BASE).run(ts.World(size=64, seed=3, wrap=True))
    w["sinks"] = np.zeros((64, 64), bool)
    with pytest.raises(ValueError):
        ts.Pipeline([step]).run(w)


def test_basin_placement_is_deterministic_and_inland():
    w = ts.Pipeline([*BASE, ts.Basins(count=3)]).run(ts.World(size=128, seed=42, wrap=True))
    assert (w["basin"] > 0).any() and not (w["basin"][w["ocean"]] > 0).any()
    again = ts.Pipeline([*BASE, ts.Basins(count=3)]).run(ts.World(size=128, seed=42, wrap=True))
    assert np.array_equal(w["basin"], again["basin"])
    none = ts.Pipeline([*BASE, ts.Basins(count=0)]).run(ts.World(size=128, seed=42, wrap=True))
    assert not none["basin"].any()
