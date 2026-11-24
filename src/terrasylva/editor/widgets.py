"""Parameter controls."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel,
                               QSlider, QSpinBox, QWidget)

from .params import Param

_TICKS = 1000


class _Slider(QSlider):
    """Ignores the wheel, so scrolling the panel doesn't change values."""

    def wheelEvent(self, event):
        event.ignore()


class NumberControl(QWidget):
    changed = Signal(object)

    def __init__(self, param: Param, value):
        super().__init__()
        self.param = param
        self._busy = False
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.auto = None
        if param.optional:
            self.auto = QCheckBox("Auto")
            self.auto.setChecked(value is None)
            self.auto.toggled.connect(self._auto_toggled)
            row.addWidget(self.auto)
        self.slider = _Slider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, _TICKS)
        if param.kind == "int":
            self.box = QSpinBox()
            self.box.setRange(int(param.low), int(param.high))
        else:
            self.box = QDoubleSpinBox()
            self.box.setDecimals(3 if param.high - param.low <= 2 else 1)
            self.box.setRange(param.low, param.high)
            self.box.setSingleStep((param.high - param.low) / 100)
        self.box.setKeyboardTracking(False)
        self.box.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        row.addWidget(self.slider, 1)
        row.addWidget(self.box)
        self._show(param.fallback if value is None else value)
        self._enable(value is not None)
        self.slider.valueChanged.connect(self._slider_moved)
        self.box.valueChanged.connect(self._box_changed)

    def _enable(self, on: bool) -> None:
        self.slider.setEnabled(on)
        self.box.setEnabled(on)

    def _show(self, value) -> None:
        self._busy = True
        self.box.setValue(int(round(value)) if self.param.kind == "int" else value)
        self._move_slider()
        self._busy = False

    def _move_slider(self) -> None:
        p = self.param
        self.slider.setValue(round((self.box.value() - p.low) / (p.high - p.low) * _TICKS))

    def _slider_moved(self, tick: int) -> None:
        if self._busy:
            return
        p = self.param
        value = p.low + (p.high - p.low) * tick / _TICKS
        self._busy = True
        self.box.setValue(round(value) if p.kind == "int" else value)
        self._busy = False
        self.changed.emit(self.box.value())

    def _box_changed(self, value) -> None:
        if self._busy:
            return
        self._busy = True
        self._move_slider()
        self._busy = False
        self.changed.emit(value)

    def _auto_toggled(self, auto: bool) -> None:
        self._enable(not auto)
        self.changed.emit(None if auto else self.box.value())


class ParamForm(QWidget):
    """A labelled control per parameter; emits (field, value)."""

    changed = Signal(str, object)

    def __init__(self, params, values: dict):
        super().__init__()
        self.controls = {}
        form = QFormLayout(self)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        for p in params:
            if p.kind == "bool":
                control = QCheckBox()
                control.setChecked(bool(values[p.field]))
                control.toggled.connect(lambda v, f=p.field: self.changed.emit(f, v))
            else:
                control = NumberControl(p, values[p.field])
                control.changed.connect(lambda v, f=p.field: self.changed.emit(f, v))
            label = QLabel(p.label)
            label.setToolTip(p.help)
            control.setToolTip(p.help)
            form.addRow(label, control)
            self.controls[p.field] = control
