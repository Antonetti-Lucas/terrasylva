import dataclasses
import json
import os

import numpy as np
import pytest

import terrasylva as ts
from terrasylva.editor import params, scene, session
from terrasylva.editor.session import Document, Generator, History, Item
from terrasylva.world import STEPS


def doc(**kw):
    return dataclasses.replace(session.default_document(), size=96, **kw)


def kinds(d):
    return [type(i.step).__name__ for i in d.items]


# ------------------------------------------------------------- generation
def test_generator_matches_pipeline():
    d = doc()
    w = Generator().run(d)
    direct = ts.Pipeline(d.steps()).run(d.world())
    assert w.layers.keys() == direct.layers.keys()
    for name in w.layers:
        assert np.array_equal(w[name], direct[name])


def test_only_changed_steps_run_again():
    ran = []
    g = Generator()
    d = doc()
    record = dict(progress=lambda i, n, name: ran.append(name))
    g.run(d, 64, **record)
    assert len(ran) == len(d.items)
    ran.clear()
    g.run(d, 64, **record)
    assert ran == []
    i = kinds(d).index("Climate")
    climate = dataclasses.replace(d.items[i].step, wind=90.0)
    d2 = session.replace_item(d, i, Item(climate))
    cached = g.run(d2, 64, **record)
    assert ran == ["climate", "rivers", "biomes"]
    assert np.array_equal(cached["biome"], Generator().run(d2, 64)["biome"])


def test_switching_a_step_off_drops_its_layers():
    g, d = Generator(), doc()
    assert "rainfall" in g.run(d, 64)
    for kind in ("Climate", "Biomes"):
        i = kinds(d).index(kind)
        d = session.replace_item(d, i, Item(d.items[i].step, enabled=False))
    w = g.run(d, 64)
    assert "rainfall" not in w and "biome" not in w


def test_cancel_stops_and_next_run_is_correct():
    g, d, seen = Generator(), doc(), []
    with pytest.raises(session.Cancelled):
        g.run(d, 64, progress=lambda i, n, name: seen.append(name),
              cancelled=lambda: len(seen) >= 2)
    assert len(seen) == 2
    assert np.array_equal(g.run(d, 64)["height"], Generator().run(d, 64)["height"])


def test_failing_step_keeps_earlier_results():
    d = doc()
    i = kinds(d).index("Climate")
    bad = session.remove_step(d, i)   # Biomes without Climate
    g, ran = Generator(), []
    with pytest.raises(ValueError, match="Climate"):
        g.run(bad, 64)
    g.run(session.remove_step(bad, len(bad.items) - 1), 64,
          progress=lambda i, n, name: ran.append(name))
    assert ran == []


def test_preview_covers_the_same_ground():
    d = doc(cell_size=2.0, relief=0.2)
    small, full = d.world(48), d.world()
    assert small.size * small.cell_size == full.size * full.cell_size == 192
    assert small.height_range == full.height_range
    assert small.relief / small.size == pytest.approx(full.relief / full.size)


def test_per_cell_counts_scale_with_preview_size():
    d = Document(size=1000, items=(Item(ts.Hydraulic(drops=10_000)), Item(ts.Rivers(threshold=500)),
                                   Item(ts.Hydraulic(name="b"))))
    a, b, c = d.steps(100)
    assert a.drops == 100 and b.threshold == pytest.approx(5) and c.drops is None
    assert d.steps()[0].drops == 10_000


# ---------------------------------------------------------------- editing
def test_add_remove_reorder():
    d = doc()
    n = len(d.items)
    d2 = session.add_step(session.add_step(d, "Thermal"), "Thermal")
    names = [i.step.name for i in d2.items]
    assert names[-2:] == ["thermal 2", "thermal 3"] and len(set(names)) == n + 2
    assert session.remove_step(session.remove_step(d2, n + 1), n) == d
    flipped = session.reorder(d, list(range(n))[::-1])
    assert kinds(flipped) == kinds(d)[::-1]
    with pytest.raises(ValueError):
        session.reorder(d, [0] * n)


def test_history_undo_redo_and_merging():
    now = [0.0]
    h = History(doc(seed=1), clock=lambda: now[0])
    for seed in (2, 3, 4):           # a slider being dragged
        h.commit(doc(seed=seed), "seed")
        now[0] += 0.1
    assert h.undo() and h.doc.seed == 1 and not h.can_undo
    assert h.redo() and h.doc.seed == 4
    now[0] += 10
    h.commit(doc(seed=5), "seed")    # later: its own step
    assert h.undo() and h.doc.seed == 4
    h.commit(doc(seed=9))
    assert not h.can_redo
    assert not h.commit(doc(seed=9))


def test_dirty_tracks_saved_state():
    h = History(doc())
    assert not h.dirty
    h.commit(doc(seed=2), "seed")
    assert h.dirty
    h.mark_saved()
    assert not h.dirty
    h.commit(doc(seed=3), "seed")    # must not merge into the saved state
    assert h.dirty and h.undo() and not h.dirty


# ------------------------------------------------------------------ files
def test_save_load_roundtrip_keeps_disabled_steps(tmp_path):
    path = str(tmp_path / "w.json")
    d = doc(seed=5, cell_size=3.0, relief=0.15, base_height=-40.0, wrap=False)
    d = session.replace_item(d, 2, Item(d.items[2].step, enabled=False))
    session.save(path, d)
    back = session.load(path)
    assert back.items == d.items
    assert (back.size, back.seed, back.cell_size, back.wrap) == (96, 5, 3.0, False)
    assert back.relief == pytest.approx(0.15) and back.base_height == -40.0


def test_cli_recipes_agree_with_editor(tmp_path):
    path = str(tmp_path / "w.json")
    d = doc()
    d = session.replace_item(d, 2, Item(d.items[2].step, enabled=False))
    session.save(path, d)
    world, pipeline = ts.load_recipe(path)
    assert len(pipeline.steps) == len(d.items) - 1
    assert np.array_equal(pipeline.run(world)["height"], Generator().run(d)["height"])
    ts.save_recipe(path, ts.World(size=64, wrap=True), ts.default_pipeline())
    assert session.load(path).steps() == ts.default_pipeline().steps


@pytest.mark.parametrize("bad", ["", "[]", "{}", '{"world": {}, "steps": [{"step": "Nope"}]}',
                                 '{"world": {"size": 2}, "steps": []}',
                                 '{"world": {}, "steps": [{"step": "Noise", "x": 1}]}'])
def test_load_rejects_bad_files(tmp_path, bad):
    path = tmp_path / "w.json"
    path.write_text(bad)
    with pytest.raises(ValueError, match="cannot open"):
        session.load(str(path))
    with pytest.raises(ValueError):
        session.load(str(tmp_path / "missing.json"))


# ----------------------------------------------------------------- params
@pytest.mark.parametrize("kind", sorted(STEPS))
def test_every_step_has_usable_controls(kind):
    step = STEPS[kind]()
    found = params.params_for(step)
    assert found
    fields = {f.name for f in dataclasses.fields(step)} - params.HIDDEN
    assert {p.field for p in found} == fields
    for p in found:
        value = getattr(step, p.field)
        if p.kind != "bool":
            assert p.low < p.high
            assert value is None or p.low <= value <= p.high
            assert p.optional or value is not None
            if p.optional:
                assert p.low <= p.fallback <= p.high


def test_unknown_steps_get_generic_controls_and_values_stay_in_range():
    @dataclasses.dataclass(frozen=True)
    class Custom:
        power: float = 3.0
        count: int = 4
        flag: bool = True
        label: str = "x"
        name: str = "custom"

    found = {p.field: p for p in params.params_for(Custom(power=99.0))}
    assert set(found) == {"power", "count", "flag"}
    assert found["power"].high >= 99.0 and found["count"].kind == "int"
    assert type(params.coerce(Custom(), "power", 2)) is float
    assert type(params.coerce(ts.Noise(), "scale", 2)) is float
    assert type(params.coerce(ts.Noise(), "octaves", 2.0)) is int
    assert params.coerce(ts.Hydraulic(), "drops", None) is None


# ------------------------------------------------------------------ scene
@pytest.fixture(scope="module")
def world():
    return Generator().run(doc(), 64)


@pytest.mark.parametrize("mode", scene.MODES)
def test_colours_every_mode(world, mode):
    img = scene.colours(world, mode, doc().steps())
    assert img.shape == (64, 64, 3) and img.dtype == np.uint8
    bare = ts.Pipeline([ts.Noise()]).run(ts.World(size=64))
    assert scene.colours(bare, mode).shape == (64, 64, 3)   # missing layers fall back


def test_surface_is_flat_over_the_sea(world):
    top = scene.surface(world)
    sea = top[world["ocean"]]
    assert np.ptp(sea) == 0
    assert (top >= world["height"] * world.relief / world.size - 1e-7).all()


@pytest.mark.parametrize("wrap", [True, False])
@pytest.mark.parametrize("resolution", [16, 64, 512])
def test_mesh_is_complete_and_tiles(wrap, resolution):
    w = ts.Pipeline([ts.Noise()]).run(ts.World(size=64, wrap=wrap))
    verts, tris = scene.mesh(w, resolution)
    k = int(round(np.sqrt(len(verts))))
    assert len(tris) == 6 * (k - 1) ** 2 and tris.max() == len(verts) - 1
    g = verts.reshape(k, k, 5)
    assert g[0, 0, 0] == -0.5 and g[-1, -1, 0] == pytest.approx(0.5)
    if wrap:
        assert np.array_equal(g[:, 0, 1], g[:, -1, 1]) and np.array_equal(g[0, :, 1], g[-1, :, 1])
        assert g[0, -1, 3] - g[0, 0, 3] == pytest.approx(1.0)
    p = verts[:, :3][tris.reshape(-1, 3)]
    normal_y = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])[:, 1]
    assert (normal_y > 0).all()     # every triangle faces up


def test_camera_projects_target_to_centre():
    c = scene.Camera()
    for _ in range(20):
        c.orbit(37, 11)
        c.zoom(0.5)
        c.pan(0.1, -0.05)
        m = c.matrix(1.6)
        p = m @ np.append(c.target, 1.0)
        assert p[3] > 0 and np.allclose(p[:2] / p[3], 0, atol=1e-9)
        above = m @ np.append(c.target + [0, 0.01, 0], 1.0)
        assert above[1] / above[3] > 0     # up stays up


# --------------------------------------------------------------------- UI
@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("moderngl")
    widgets = pytest.importorskip("PySide6.QtWidgets")
    return widgets.QApplication.instance() or widgets.QApplication([])


@pytest.fixture
def window(app):
    from terrasylva.editor.window import MainWindow
    w = MainWindow()
    w.history.reset(doc())
    w.refresh()
    w.preview_size.setCurrentIndex(0)
    w.generate_preview()
    wait(app, w)
    yield w
    w.history.mark_saved()
    w.close()


def wait(app, w, seconds=60):
    import time
    end = time.monotonic() + seconds
    while w._running or w._queued or w._debounce.isActive():
        app.processEvents()
        time.sleep(0.002)
        assert time.monotonic() < end, "generation did not finish"


def test_window_generates_and_slider_changes_world(app, window):
    before = window.result["world"]["height"]
    window.list.setCurrentRow(kinds(window.doc).index("Ocean"))
    window.step_form.controls["land"].box.setValue(0.3)
    wait(app, window)
    land = 1 - window.result["world"]["ocean"].mean()
    assert 0.28 < land < 0.4 and window.history.dirty
    window.undo()
    wait(app, window)
    assert np.array_equal(window.result["world"]["height"], before)
    assert window.step_form.controls["land"].box.value() == pytest.approx(0.6)


def test_window_untick_reorder_add_remove(app, window):
    from PySide6.QtCore import Qt
    row = kinds(window.doc).index("Biomes")
    window.list.item(row).setCheckState(Qt.CheckState.Unchecked)
    wait(app, window)
    assert "biome" not in window.result["world"]
    window.add_step("Thermal")
    assert window.list.currentRow() == len(window.doc.items) - 1
    window.remove_step()
    assert kinds(window.doc) == kinds(doc())
    model = window.list.model()
    assert model.moveRow(window.list.rootIndex(), 2, window.list.rootIndex(), 4)
    app.processEvents()
    wait(app, window)
    assert kinds(window.doc)[2:4] == ["Hydraulic", "Thermal"]


def test_window_shows_errors_and_recovers(app, window):
    from PySide6.QtCore import Qt
    item = window.list.item(kinds(window.doc).index("Climate"))
    item.setCheckState(Qt.CheckState.Unchecked)
    wait(app, window)
    assert "Climate" in window.message.text()
    item.setCheckState(Qt.CheckState.Checked)
    wait(app, window)
    assert window.message.text().startswith("Ready")


def test_rapid_edits_end_on_the_last_one(app, window):
    for seed in range(20, 30):
        window._world_changed("seed", seed)
        window.generate_preview()
    wait(app, window)
    assert window.result["job"]["doc"].seed == 29


def test_export_survives_edits_made_while_it_runs(app, window, tmp_path):
    window.export_to(str(tmp_path))
    window._world_changed("seed", 77)
    window.generate_preview()
    wait(app, window)
    files = set(os.listdir(tmp_path))
    assert {"heightmap.png", "biomes.png", "water.png", "recipe.json", "terrain.obj"} <= files
    with open(tmp_path / "recipe.json") as f:
        assert json.load(f)["world"]["seed"] != 77
    assert window.result["job"]["doc"].seed == 77


def test_window_save_and_reopen(app, window, tmp_path):
    path = str(tmp_path / "w.json")
    window._world_changed("seed", 5)
    assert window._write(path) and not window.history.dirty
    saved = window.doc
    window._world_changed("seed", 6)
    window.open_path(path)
    wait(app, window)
    assert window.doc.items == saved.items and window.doc.seed == 5 and window.path == path


def test_modes_switch_without_regenerating(app, window):
    world = window.result["world"]
    for mode in scene.MODES:
        window.mode.setCurrentText(mode)
    assert window.result["world"] is world


def test_auto_control_toggles_none(app, window):
    window.list.setCurrentRow(kinds(window.doc).index("Hydraulic"))
    control = window.step_form.controls["drops"]
    assert control.auto.isChecked() and not control.slider.isEnabled()
    control.auto.setChecked(False)
    row = kinds(window.doc).index("Hydraulic")
    assert window.doc.items[row].step.drops == 300_000
    control.auto.setChecked(True)
    assert window.doc.items[row].step.drops is None
