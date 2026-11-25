"""Main window of the visual editor."""
from __future__ import annotations

import dataclasses
import os
import random
import sys
import threading
import time

from PySide6.QtCore import QObject, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence, QSurfaceFormat
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDockWidget, QFileDialog,
                               QGroupBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                               QMainWindow, QMenu, QMessageBox, QProgressBar, QPushButton,
                               QScrollArea, QSlider, QToolBar, QToolButton, QVBoxLayout, QWidget)

from ..cli import export_world
from ..world import STEPS, Pipeline
from . import params, scene, session
from .view import TerrainView
from .widgets import ParamForm

_STEP_LABELS = {"Noise": "Base terrain", "Continents": "Continents",
                "Ranges": "Mountain ranges", "Fluvial": "River landscape",
                "SeaFloor": "Sea floor", "Basins": "Basins below sea", "Thermal": "Slope erosion",
                "Hydraulic": "Water erosion", "Ocean": "Ocean", "Climate": "Climate",
                "Rivers": "Rivers and lakes", "Biomes": "Biomes"}
_FILTER = "terrasylva recipe (*.json)"


def step_label(step) -> str:
    label = _STEP_LABELS.get(type(step).__name__, step.name)
    suffix = step.name.rpartition(" ")[2]
    return f"{label} {suffix}" if suffix.isdigit() else label


class Worker(QObject):
    """Runs one job at a time on its own thread.

    Jobs must never overlap: Numba's parallel kernels aren't safe to enter from
    two threads at once.
    """

    progress = Signal(int, int, str)
    finished = Signal(object)

    def __init__(self):
        super().__init__()
        self.cancel = threading.Event()
        self._preview = session.Generator()

    def run(self, job: dict) -> None:
        result = {"job": job, "world": None, "error": None, "cancelled": False}
        t0 = time.perf_counter()
        doc, size, folder = job["doc"], job["size"], job["folder"]
        try:
            # full-size results are big, so only previews are cached
            gen = session.Generator() if job["fresh"] else self._preview
            world = gen.run(doc, size, progress=self.progress.emit,
                            cancelled=self.cancel.is_set)
            if folder is not None:
                self.progress.emit(0, 0, "writing files")
                export_world(world, folder, 256, Pipeline(doc.steps(size)))
                session.save(os.path.join(folder, "recipe.json"), doc)
            result["world"] = world
            result["pixels"] = scene.colours(world, job["mode"], doc.steps(size))
        except session.Cancelled:
            result["cancelled"] = True
        except Exception as e:  # any step may fail; the editor must survive it
            result["error"] = str(e) or type(e).__name__
        result["seconds"] = time.perf_counter() - t0
        self.finished.emit(result)


class MainWindow(QMainWindow):
    _start = Signal(object)

    def __init__(self):
        super().__init__()
        self.history = session.History(session.preset("World map"))
        self.path: str | None = None
        self.result = None           # last successful result
        self._running = None         # job in progress
        self._queued = None          # job waiting for it
        self._loading = False

        self._thread = QThread(self)
        self.worker = Worker()
        self.worker.moveToThread(self._thread)
        self._start.connect(self.worker.run)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self._thread.start()

        self._debounce = QTimer(self, singleShot=True, interval=200)
        self._debounce.timeout.connect(self.generate_preview)

        self.view = TerrainView()
        self.setCentralWidget(self.view)
        self._build_actions()
        self._build_docks()
        self._build_status()
        self.resize(QSize(1280, 800))
        self.refresh()
        self.generate_preview()

    # ------------------------------------------------------------- layout
    def _action(self, text, slot, shortcut=None, tip=""):
        a = QAction(text, self)
        a.triggered.connect(lambda *_: slot())
        if shortcut:
            a.setShortcut(shortcut)
        a.setToolTip(tip or text)
        return a

    def _build_actions(self):
        K = QKeySequence.StandardKey
        self.act_new = self._action("New", self.new, K.New)
        self.act_open = self._action("Open…", self.open, K.Open)
        self.act_save = self._action("Save", self.save, K.Save)
        self.act_save_as = self._action("Save as…", self.save_as, K.SaveAs)
        self.act_export = self._action("Export…", self.export, "Ctrl+E",
                                       "Generate at the final size and write files for game engines")
        self.act_undo = self._action("Undo", self.undo, K.Undo)
        self.act_redo = self._action("Redo", self.redo, K.Redo)
        self.act_random = self._action("New seed", self.random_seed, "Ctrl+R")
        self.act_full = self._action("Show final size", self.generate_full, "F5",
                                     "Generate and show the world at its final size")
        self.act_camera = self._action("Reset camera", self.view.reset_camera, "Home")
        self.act_top = self._action("Map view", self.view.top_view, "M", "Look straight down")
        self.act_3d = self._action("3D view", self.view.reset_camera, "N")
        file = self.menuBar().addMenu("&File")
        file.addActions([self.act_new, self.act_open, self.act_save, self.act_save_as,
                         self.act_export])
        edit = self.menuBar().addMenu("&Edit")
        edit.addActions([self.act_undo, self.act_redo, self.act_random])
        view = self.menuBar().addMenu("&View")
        view.addActions([self.act_full, self.act_top, self.act_3d, self.act_camera])
        bar = QToolBar("Main")
        bar.setObjectName("main")
        bar.setMovable(False)
        self.presets = QToolButton()
        self.presets.setText("New world")
        self.presets.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self.presets)
        for name in session.PRESETS:
            menu.addAction(name, lambda n=name: self.use_preset(n))
        self.presets.setMenu(menu)
        bar.addWidget(self.presets)
        bar.addActions([self.act_open, self.act_save, self.act_export])
        bar.addSeparator()
        bar.addActions([self.act_undo, self.act_redo, self.act_random])
        bar.addSeparator()
        bar.addWidget(QLabel(" Show "))
        self.mode = QComboBox()
        self.mode.addItems(scene.MODES)
        self.mode.currentTextChanged.connect(lambda _: self.recolour())
        bar.addWidget(self.mode)
        bar.addWidget(QLabel("  Preview "))
        self.preview_size = QComboBox()
        for n in (128, 256, 512, 1024):
            self.preview_size.addItem(str(n), n)
        self.preview_size.setCurrentIndex(2)
        self.preview_size.currentIndexChanged.connect(lambda _: self.generate_preview())
        bar.addWidget(self.preview_size)
        bar.addAction(self.act_full)
        bar.addActions([self.act_top, self.act_3d])
        bar.addSeparator()
        bar.addWidget(QLabel(" Exaggerate "))
        self.lift = QSlider(Qt.Orientation.Horizontal)
        self.lift.setRange(10, 40)
        self.lift.setFixedWidth(90)
        self.lift.valueChanged.connect(lambda v: self.view.set_lift(v / 10))
        bar.addWidget(self.lift)
        self.tiled = QCheckBox("Repeat")
        self.tiled.setToolTip("Draw copies around a looping world")
        self.tiled.toggled.connect(self.view.set_tiled)
        bar.addWidget(self.tiled)
        self.addToolBar(bar)

    def _build_docks(self):
        panel = QWidget()
        column = QVBoxLayout(panel)
        self.world_box = QGroupBox("World")
        self.world_layout = QVBoxLayout(self.world_box)
        column.addWidget(self.world_box)

        steps_box = QGroupBox("Steps")
        steps_layout = QVBoxLayout(steps_box)
        self.list = QListWidget()
        self.list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.list.setToolTip("Steps run from top to bottom. Drag to reorder, untick to skip.")
        self.list.setMinimumHeight(315)
        self.list.currentRowChanged.connect(lambda _: self._show_params())
        self.list.itemChanged.connect(self._item_changed)
        # the list is mid-move when this fires, so read it once Qt is done
        self.list.model().rowsMoved.connect(lambda *_: QTimer.singleShot(0, self._rows_moved))
        steps_layout.addWidget(self.list)
        row = QHBoxLayout()
        self.add_button = QToolButton()
        self.add_button.setText("Add step")
        self.add_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self.add_button)
        for kind in STEPS:
            menu.addAction(_STEP_LABELS.get(kind, kind), lambda k=kind: self.add_step(k))
        self.add_button.setMenu(menu)
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self.remove_step)
        row.addWidget(self.add_button)
        row.addWidget(self.remove_button)
        row.addStretch(1)
        steps_layout.addLayout(row)
        column.addWidget(steps_box)

        self.params_box = QGroupBox("Settings")
        self.params_layout = QVBoxLayout(self.params_box)
        column.addWidget(self.params_box)
        column.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(370)
        dock = QDockWidget("World builder")
        dock.setObjectName("builder")
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        dock.setWidget(scroll)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def _build_status(self):
        self.message = QLabel()
        self.bar = QProgressBar()
        self.bar.setFixedWidth(200)
        self.bar.hide()
        self.statusBar().addWidget(self.message, 1)
        self.statusBar().addPermanentWidget(self.bar)

    # -------------------------------------------------------------- state
    @property
    def doc(self) -> session.Document:
        return self.history.doc

    def edit(self, doc: session.Document, key=None, rebuild: bool = True) -> None:
        """Make an undoable change and regenerate."""
        if not self.history.commit(doc, key):
            return
        if rebuild:
            self.refresh()
        else:
            self._update_titles()
        self._debounce.start()

    def refresh(self) -> None:
        """Rebuild the panels from the document."""
        self._loading = True
        row = max(self.list.currentRow(), 0)
        self.list.clear()
        for i, item in enumerate(self.doc.items):
            entry = QListWidgetItem(step_label(item.step))
            entry.setFlags((entry.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                           & ~Qt.ItemFlag.ItemIsDropEnabled)
            entry.setCheckState(Qt.CheckState.Checked if item.enabled else Qt.CheckState.Unchecked)
            entry.setData(Qt.ItemDataRole.UserRole, i)
            self.list.addItem(entry)
        if self.list.count():
            self.list.setCurrentRow(min(row, self.list.count() - 1))
        self._show_world()
        self._loading = False
        self._show_params()
        self._update_titles()

    def _show_world(self) -> None:
        row = session.ocean_index(self.doc)
        values = {p.field: getattr(self.doc, p.field, None) for p in params.WORLD}
        if row is not None:
            values["land"] = self.doc.items[row].step.land
        shown = [p for p in params.WORLD if values[p.field] is not None]
        # widen ranges for values that came from a file
        fields = [p if p.kind == "bool" else dataclasses.replace(
            p, low=min(p.low, values[p.field]), high=max(p.high, values[p.field]))
            for p in shown]
        self.world_form = ParamForm(fields, values)
        self.world_form.changed.connect(self._world_changed)
        self._swap(self.world_layout, self.world_form)

    @staticmethod
    def _swap(layout, widget) -> None:
        while layout.count():
            old = layout.takeAt(0).widget()
            old.hide()
            old.deleteLater()
        layout.addWidget(widget)

    def _show_params(self) -> None:
        if self._loading:
            return
        row = self.list.currentRow()
        self.step_form = None
        if not 0 <= row < len(self.doc.items):
            self._swap(self.params_layout, QLabel("Add a step to begin"))
            return
        step = self.doc.items[row].step
        fields = params.params_for(step)
        self.step_form = ParamForm(fields, {p.field: getattr(step, p.field) for p in fields})
        self.step_form.changed.connect(lambda f, v, r=row: self._step_changed(r, f, v))
        self.params_box.setTitle(f"Settings: {step_label(step)}")
        self._swap(self.params_layout, self.step_form)

    def _update_titles(self) -> None:
        name = os.path.basename(self.path) if self.path else "Untitled"
        self.setWindowTitle(f"{name}{'*' if self.history.dirty else ''} – terrasylva")
        self.act_undo.setEnabled(self.history.can_undo)
        self.act_redo.setEnabled(self.history.can_redo)
        self.remove_button.setEnabled(bool(self.doc.items))
        self.tiled.setEnabled(self.doc.wrap)

    # -------------------------------------------------------------- edits
    def _world_changed(self, field, value):
        if field == "land":
            row = session.ocean_index(self.doc)
            item = self.doc.items[row]
            step = dataclasses.replace(item.step, land=float(value))
            self.edit(session.replace_item(self.doc, row, session.Item(step, item.enabled)),
                      ("world", "land"), rebuild=False)
            if self.list.currentRow() == row:
                self._show_params()
            return
        self.edit(dataclasses.replace(self.doc, **{field: value}), ("world", field), rebuild=False)

    def _step_changed(self, row, field, value):
        item = self.doc.items[row]
        step = dataclasses.replace(item.step, **{field: params.coerce(item.step, field, value)})
        self.edit(session.replace_item(self.doc, row, session.Item(step, item.enabled)),
                  (row, field), rebuild=False)
        if type(step).__name__ == "Ocean":
            self._show_world()

    def _item_changed(self, entry):
        if self._loading:
            return
        row = self.list.row(entry)
        item = self.doc.items[row]
        on = entry.checkState() == Qt.CheckState.Checked
        self.edit(session.replace_item(self.doc, row, session.Item(item.step, on)), rebuild=False)
        if type(item.step).__name__ == "Ocean":
            self._show_world()

    def _rows_moved(self):
        order = [self.list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.list.count())]
        self.edit(session.reorder(self.doc, order))

    def add_step(self, kind):
        self.edit(session.add_step(self.doc, kind))
        self.list.setCurrentRow(self.list.count() - 1)

    def remove_step(self):
        row = self.list.currentRow()
        if 0 <= row < len(self.doc.items):
            self.edit(session.remove_step(self.doc, row))

    def use_preset(self, name):
        self.edit(session.preset(name, self.doc.seed))

    def random_seed(self):
        self.edit(dataclasses.replace(self.doc, seed=random.SystemRandom().randrange(1_000_000)))

    def undo(self):
        if self.history.undo():
            self.refresh()
            self._debounce.start()

    def redo(self):
        if self.history.redo():
            self.refresh()
            self._debounce.start()

    # --------------------------------------------------------- generation
    def generate_preview(self):
        size = min(self.preview_size.currentData(), self.doc.size)
        self._submit(size=size, fresh=False, folder=None)

    def generate_full(self):
        self._submit(size=self.doc.size, fresh=True, folder=None)

    def _submit(self, *, size, fresh, folder):
        self._debounce.stop()
        job = dict(doc=self.doc, size=size, fresh=fresh, folder=folder,
                   mode=self.mode.currentText())
        if self._running is None:
            self._launch(job)
            return
        if self._queued is not None and self._queued["folder"] is not None and folder is None:
            return  # a waiting export must not be lost
        self._queued = job
        # an export in progress always finishes; anything else is out of date
        if self._running["folder"] is None:
            self.worker.cancel.set()

    def _launch(self, job):
        self.worker.cancel.clear()
        self._running = job
        self.bar.setRange(0, 0)
        self.bar.show()
        self._start.emit(job)

    def _on_progress(self, i, total, name):
        self.bar.setRange(0, total)
        self.bar.setValue(i)
        self.message.setText(f"Generating: {name}…" if total else f"{name.capitalize()}…")

    def _on_finished(self, result):
        job, self._running = result["job"], None
        self.bar.hide()
        if result["error"]:
            self.message.setText(f"⚠ {result['error']}")
            if job["folder"] is not None:
                QMessageBox.warning(self, "Export failed", result["error"])
        elif not result["cancelled"]:
            self.result = result
            world = result["world"]
            self.view.show_world(world, result["pixels"])
            done = f"Exported to {job['folder']}" if job["folder"] else "Ready"
            self.message.setText(f"{done} · {job['size']}×{job['size']} in {result['seconds']:.1f} s"
                                 f" · {world.size * world.cell_size / 1000:.1f} km across")
            if job["mode"] != self.mode.currentText():
                self.recolour()
        if self._queued is not None:
            job, self._queued = self._queued, None
            self._launch(job)

    def recolour(self):
        if self.result is None:
            return
        world, job = self.result["world"], self.result["job"]
        mode = self.mode.currentText()
        self.view.show_colours(scene.colours(world, mode, job["doc"].steps(job["size"])))
        if not scene.available(world, mode):
            self.message.setText(f"This world has no {mode.lower()} yet: add the step that makes it")

    # -------------------------------------------------------------- files
    def _confirm_discard(self) -> bool:
        if not self.history.dirty:
            return True
        B = QMessageBox.StandardButton
        answer = QMessageBox.question(self, "Unsaved changes", "Save your changes first?",
                                      B.Save | B.Discard | B.Cancel)
        if answer == B.Save:
            return self.save()
        return answer == B.Discard

    def _load(self, doc, path):
        self.history.reset(doc)
        self.path = path
        self.refresh()
        self.generate_preview()

    def new(self):
        if self._confirm_discard():
            self._load(session.preset("World map"), None)

    def open(self):
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open recipe", "", _FILTER)
        if path:
            self.open_path(path)

    def open_path(self, path):
        try:
            doc = session.load(path)
        except ValueError as e:
            QMessageBox.warning(self, "Cannot open", str(e))
            return
        self._load(doc, path)

    def save(self) -> bool:
        if self.path is None:
            return self.save_as()
        return self._write(self.path)

    def save_as(self) -> bool:
        path, _ = QFileDialog.getSaveFileName(self, "Save recipe", "world.json", _FILTER)
        return bool(path) and self._write(path)

    def _write(self, path) -> bool:
        try:
            session.save(path, self.doc)
        except OSError as e:
            QMessageBox.warning(self, "Cannot save", str(e))
            return False
        self.path = path
        self.history.mark_saved()
        self._update_titles()
        self.message.setText(f"Saved {path}")
        return True

    def export(self):
        folder = QFileDialog.getExistingDirectory(self, "Export to folder")
        if folder:
            self.export_to(folder)

    def export_to(self, folder):
        self._submit(size=self.doc.size, fresh=True, folder=folder)

    def closeEvent(self, event):
        if not self._confirm_discard():
            event.ignore()
            return
        self.shutdown()
        event.accept()

    def shutdown(self):
        """Stop the worker, letting an export in progress finish."""
        self._debounce.stop()
        self._queued = None
        if self._running is not None and self._running["folder"] is None:
            self.worker.cancel.set()
        self._thread.quit()
        self._thread.wait()


def main(argv: list[str] | None = None) -> int:
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)
    QSurfaceFormat.setDefaultFormat(fmt)
    args = list(sys.argv[1:] if argv is None else argv)
    app = QApplication.instance() or QApplication(["terrasylva"])
    app.setApplicationName("terrasylva")
    window = MainWindow()
    if args:
        window.open_path(args[0])
    window.show()
    return app.exec()
