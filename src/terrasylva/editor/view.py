"""OpenGL terrain view with an orbit camera."""
from __future__ import annotations

import moderngl
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from ..world import World
from . import scene

_VERTEX = """
#version 330
in vec3 pos; in vec2 uv; out vec2 t;
uniform mat4 mvp; uniform vec2 offset; uniform float lift;
void main() {
    t = uv;
    gl_Position = mvp * vec4(pos.x + offset.x, pos.y * lift, pos.z + offset.y, 1.0);
}
"""
_FRAGMENT = """
#version 330
in vec2 t; out vec4 f; uniform sampler2D tex;
void main() { f = vec4(texture(tex, t).rgb, 1.0); }
"""


class TerrainView(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.camera = scene.Camera()
        self.lift = 1.0
        self.tiled = False
        self.error: str | None = None
        self._ctx = None
        self._mesh = self._pixels = None     # waiting to be uploaded
        self._wrap = False
        self._vao = self._texture = None
        self._buffers: list = []
        self._last = None

    # ------------------------------------------------------------ content
    def show_world(self, world: World, pixels: np.ndarray) -> None:
        self._wrap = world.wrap
        self._mesh = scene.mesh(world)
        self.show_colours(pixels)

    def show_colours(self, pixels: np.ndarray) -> None:
        self._pixels = np.ascontiguousarray(pixels)
        self.update()

    def set_lift(self, lift: float) -> None:
        self.lift = lift
        self.update()

    def set_tiled(self, tiled: bool) -> None:
        self.tiled = tiled
        self.update()

    def reset_camera(self) -> None:
        self.camera = scene.Camera()
        self.update()

    # ----------------------------------------------------------------- GL
    def initializeGL(self):
        try:
            self._ctx = moderngl.create_context()
            self._program = self._ctx.program(vertex_shader=_VERTEX, fragment_shader=_FRAGMENT)
        except Exception as e:  # driver problems come in many types
            self._ctx = None
            self.error = f"3D view unavailable: {e}"

    def _upload(self):
        if self._mesh is not None:
            for b in self._buffers:
                b.release()
            verts, tris = self._mesh
            vbo, ibo = self._ctx.buffer(verts.tobytes()), self._ctx.buffer(tris.tobytes())
            self._vao = self._ctx.vertex_array(self._program, [(vbo, "3f 2f", "pos", "uv")], ibo)
            self._buffers = [self._vao, vbo, ibo]
            self._mesh = None
        if self._pixels is not None:
            if self._texture is not None:
                self._texture.release()
            h, w = self._pixels.shape[:2]
            self._texture = self._ctx.texture((w, h), 3, self._pixels.tobytes(), alignment=1)
            self._texture.repeat_x = self._texture.repeat_y = self._wrap
            self._texture.build_mipmaps()
            self._texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            self._texture.anisotropy = 8.0
            self._pixels = None

    def paintGL(self):
        if self._ctx is None:
            painter = QPainter(self)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.error or "")
            painter.end()
            return
        self._ctx.detect_framebuffer(self.defaultFramebufferObject()).use()
        self._ctx.clear(0.11, 0.13, 0.17, 1.0)
        self._upload()
        if self._vao is None:
            return
        self._ctx.enable(moderngl.DEPTH_TEST)
        aspect = max(self.width(), 1) / max(self.height(), 1)
        mvp = self.camera.matrix(aspect)
        self._program["mvp"].write(mvp.T.astype("f4").tobytes())
        self._program["lift"].value = self.lift
        self._texture.use(0)
        around = (-1, 0, 1) if self.tiled and self._wrap else (0,)
        for dz in around:
            for dx in around:
                self._program["offset"].value = (dx, dz)
                self._vao.render()

    # -------------------------------------------------------------- mouse
    def mousePressEvent(self, event):
        self._last = event.position()

    def mouseMoveEvent(self, event):
        if self._last is None:
            return
        d = event.position() - self._last
        self._last = event.position()
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.camera.orbit(d.x() * 0.4, d.y() * 0.4)
        elif event.buttons() & (Qt.MouseButton.RightButton | Qt.MouseButton.MiddleButton):
            scale = 1.0 / max(self.height(), 1)
            self.camera.pan(d.x() * scale, d.y() * scale)
        self.update()

    def wheelEvent(self, event):
        self.camera.zoom(event.angleDelta().y() / 120)
        self.update()
