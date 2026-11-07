"""
core/kinect/viewer.py — KinectScanWindow: the small separate scan window.

Two phases inside one frameless window:

    SCAN  — live IR-intensity preview (the Kinect's IR sensor stream) with the
            detected object region highlighted, plus frame/point counters and
            a "Finish & View" button. The user places the object on a flat
            surface and rotates it; each tick feeds the newest frame through
            KinectScanner.step() which subtracts the tabletop plane and keeps
            only object points.

    VIEW  — the accumulated cloud rendered as a rotating point cloud. The
            renderer is pure QPainter + numpy (no OpenGL, no extra deps —
            same approach as the HUD avatar): points are projected
            orthographically, z-buffered, and shaded cyan→white by IR
            intensity and depth. Mouse drag orbits the model. Buttons export
            .ply (with vertex colours) or .obj into data/scans/.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PyQt6.QtCore import Qt, QPoint, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap, QCursor
from PyQt6.QtWidgets import (
    QFileDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from core.hud_video.layering import make_frameless_overlay, raise_overlay
from . import geometry
from .scanner import KinectScanner, ScanStatus, store_cloud

log = logging.getLogger(__name__)

# ── Palette (matches the HUD tokens used across ui.py / image_viewer) ────────
C_BG = "#050b14"
C_PANEL = "#0a1420"
C_PRI = "#00D4FF"
C_PRI_DIM = "#007A9A"
C_TEXT_DIM = "#405060"
C_TEXT_MED = "#7a93a8"
C_WHITE = "#e8f4ff"
C_WARN = "#ffb347"

SCAN_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "scans"
TICK_MS = 33                     # ~30 fps preview/accumulation
MIN_SCAN_POINTS = 300            # below this "Finish" is refused as too sparse


def _font(size: int, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    f = QFont("Consolas")
    f.setPointSizeF(float(size))
    f.setWeight(weight)
    f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.4)
    return f


def _btn_style(bg: str, fg: str, border: str) -> str:
    return f"""
        QPushButton {{
            background: {bg}; color: {fg}; border: 1px solid {border};
            border-radius: 2px; padding: 5px 14px;
            font-family: Consolas; font-size: 9pt; font-weight: bold;
        }}
        QPushButton:hover {{ background: {C_PRI}; color: #000; border-color: {C_WHITE}; }}
        QPushButton:pressed {{ background: {C_PRI_DIM}; color: {C_WHITE}; }}
        QPushButton:disabled {{ background: {C_PANEL}; color: {C_TEXT_DIM}; border-color: {C_TEXT_DIM}; }}
    """


# ---------------------------------------------------------------------------
# Point-cloud canvas — software projection, no OpenGL
# ---------------------------------------------------------------------------

class _CloudCanvas(QWidget):
    """Rotating orthographic point-cloud view rendered with QPainter."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(300)
        self._cloud: Optional[np.ndarray] = None     # (N,4) x,y,z,intensity
        self._pts: Optional[np.ndarray] = None       # centred xyz
        self._inten: Optional[np.ndarray] = None
        self._scale = 1.0
        self._yaw = 0.6
        self._pitch = 0.32
        self._drag = QPoint()
        self._dragging = False
        self._spin = QTimer(self)
        self._spin.setInterval(TICK_MS)
        self._spin.timeout.connect(self._advance)
        self._buf = np.zeros((2, 2, 3), np.uint8)
        self._qimg: Optional[QImage] = None

    # -- data --------------------------------------------------------------
    def set_cloud(self, cloud: np.ndarray) -> None:
        if cloud is None or cloud.shape[0] == 0:
            self._cloud = None
            self._pts = None
            return
        pts = cloud[:, :3].astype(np.float32)
        centre = pts.mean(axis=0)
        self._pts = pts - centre
        self._inten = cloud[:, 3].astype(np.float32) if cloud.shape[1] > 3 \
            else np.full(pts.shape[0], 0.5, np.float32)
        self._cloud = cloud
        extent = float(np.abs(self._pts).max()) or 1.0
        self._scale = 1.0 / extent
        self._yaw = 0.6
        self.update()

    def point_count(self) -> int:
        return 0 if self._pts is None else int(self._pts.shape[0])

    # -- rotation ----------------------------------------------------------
    def _advance(self) -> None:
        if not self._dragging:
            self._yaw += 0.018
        self.update()

    def start_spin(self) -> None:
        self._spin.start()

    def stop_spin(self) -> None:
        self._spin.stop()

    # -- mouse orbit -------------------------------------------------------
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag = e.position().toPoint()

    def mouseMoveEvent(self, e):
        if self._dragging and self._pts is not None:
            p = e.position().toPoint()
            self._yaw += (p.x() - self._drag.x()) * 0.01
            self._pitch = float(np.clip(self._pitch + (p.y() - self._drag.y()) * 0.005,
                                        -1.2, 1.2))
            self._drag = p
            self.update()

    def mouseReleaseEvent(self, e):
        self._dragging = False

    # -- render ------------------------------------------------------------
    def _render_frame(self) -> QImage:
        w, h = max(2, self.width()), max(2, self.height())
        if self._buf.shape[0] != h or self._buf.shape[1] != w:
            self._buf = np.zeros((h, w, 3), np.uint8)
        img = self._buf
        img[:] = (5, 11, 20)                       # C_BG

        if self._pts is None or self._pts.shape[0] == 0:
            self._qimg = QImage(img.data, w, h, w * 3, QImage.Format.Format_RGB888)
            return self._qimg

        cy, sy = np.cos(self._yaw), np.sin(self._yaw)
        cp, sp = np.cos(self._pitch), np.sin(self._pitch)
        p = self._pts
        # Ry(yaw) then Rx(pitch)
        x1 = p[:, 0] * cy + p[:, 2] * sy
        z1 = -p[:, 0] * sy + p[:, 2] * cy
        y2 = p[:, 1] * cp - z1 * sp
        z2 = p[:, 1] * sp + z1 * cp
        x2 = x1

        s = (min(w, h) * 0.44) * self._scale
        ix = np.rint(w / 2 + x2 * s).astype(np.int32)
        iy = np.rint(h / 2 - y2 * s).astype(np.int32)
        valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        if not valid.any():
            self._qimg = QImage(img.data, w, h, w * 3, QImage.Format.Format_RGB888)
            return self._qimg

        ix, iy, z2 = ix[valid], iy[valid], z2[valid]
        inten = self._inten[valid] if self._inten is not None else np.full(ix.size, 0.5)

        # Z-buffer: keep the nearest surface per pixel.
        zbuf = np.full((h, w), -1e9, np.float32)
        np.maximum.at(zbuf, (iy, ix), z2)
        sel = z2 == zbuf[iy, ix]
        ix, iy, z2, inten = ix[sel], iy[sel], z2[sel], inten[sel]

        if ix.size == 0:
            self._qimg = QImage(img.data, w, h, w * 3, QImage.Format.Format_RGB888)
            return self._qimg

        # Depth shade: nearer = brighter. IR intensity tints cyan→white.
        zmin, zmax = float(z2.min()), float(z2.max())
        zn = (z2 - zmin) / max(zmax - zmin, 1e-6)
        shade = (0.35 + 0.65 * zn).astype(np.float32)
        t = np.clip(inten, 0.0, 1.0)
        r = ((20 + 60 * t) * shade).astype(np.uint8)
        g = ((140 + 90 * t) * shade).astype(np.uint8)
        b = np.clip((200 + 55 * t) * shade, 0, 255).astype(np.uint8)

        # Sparse clouds get 2 px points for visibility.
        if ix.size < 60000:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    jx = np.clip(ix + dx, 0, w - 1)
                    jy = np.clip(iy + dy, 0, h - 1)
                    img[jy, jx] = np.stack([r, g, b], axis=1)
        else:
            img[iy, ix] = np.stack([r, g, b], axis=1)

        self._qimg = QImage(img.data, w, h, w * 3, QImage.Format.Format_RGB888)
        return self._qimg

    def paintEvent(self, _e):
        p = QPainter(self)
        p.drawImage(0, 0, self._render_frame())
        # Frame border
        p.setPen(QPen(QColor(C_PRI_DIM)))
        p.drawRect(0, 0, self.width() - 1, self.height() - 1)
        p.end()


# ---------------------------------------------------------------------------
# Scan window
# ---------------------------------------------------------------------------

class KinectScanWindow(QWidget):
    """Small separate window hosting the guided scan + point-cloud viewer."""

    closed = pyqtSignal()
    saved = pyqtSignal(str, int)

    def __init__(self, scanner: KinectScanner,
                 parent: Optional[QWidget] = None,
                 on_status: Optional[Callable[[str], None]] = None) -> None:
        super().__init__(parent)
        self._scanner = scanner
        self._on_status = on_status or (lambda m: None)
        self._drag_pos = QPoint()
        self._dragging = False
        self._cloud: Optional[np.ndarray] = None
        self._scan_active = False

        make_frameless_overlay(self, parent)
        self.setFixedSize(540, 520)
        self.setStyleSheet(f"""
            KinectScanWindow {{ background: {C_PANEL}; border: 1px solid {C_PRI_DIM}; }}
            QLabel {{ color: {C_TEXT_MED}; font-family: Consolas; }}
        """)
        self._build_ui()

        self._tick = QTimer(self)
        self._tick.setInterval(TICK_MS)
        self._tick.timeout.connect(self._on_tick)
        self._tick.start()

    # -- UI construction ---------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 10)
        root.setSpacing(6)

        # Title bar (drag handle)
        bar = QHBoxLayout()
        title = QLabel("◈ KINECT // 3D SCANNER")
        title.setFont(_font(10, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {C_PRI};")
        bar.addWidget(title)
        bar.addStretch(1)
        self._phase_lbl = QLabel("SCAN")
        self._phase_lbl.setFont(_font(8))
        self._phase_lbl.setStyleSheet(f"color: {C_WARN};")
        bar.addWidget(self._phase_lbl)
        root.addLayout(bar)

        # Preview / cloud canvas
        self._preview = QLabel("initialising sensor…")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setFixedSize(520, 340)
        self._preview.setStyleSheet(f"background: {C_BG}; border: 1px solid {C_TEXT_DIM};")
        root.addWidget(self._preview, alignment=Qt.AlignmentFlag.AlignCenter)

        # Status line
        self._status_lbl = QLabel("—")
        self._status_lbl.setFont(_font(8))
        self._status_lbl.setWordWrap(True)
        root.addWidget(self._status_lbl)

        # Hint line
        self._hint_lbl = QLabel("Place the object on a flat surface inside the frame.")
        self._hint_lbl.setFont(_font(8))
        self._hint_lbl.setStyleSheet(f"color: {C_TEXT_DIM};")
        self._hint_lbl.setWordWrap(True)
        root.addWidget(self._hint_lbl)

        # Buttons
        btns = QHBoxLayout()
        self._finish_btn = QPushButton("Finish & View")
        self._finish_btn.setFont(_font(9, QFont.Weight.Bold))
        self._finish_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._finish_btn.setStyleSheet(_btn_style(C_PRI, "#000", C_PRI))
        self._finish_btn.clicked.connect(self._finish_scan)
        btns.addWidget(self._finish_btn)

        self._rescan_btn = QPushButton("Rescan")
        self._rescan_btn.setFont(_font(9))
        self._rescan_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._rescan_btn.setStyleSheet(_btn_style(C_PANEL, C_TEXT_MED, C_TEXT_DIM))
        self._rescan_btn.clicked.connect(self._rescan)
        self._rescan_btn.setVisible(False)
        btns.addWidget(self._rescan_btn)

        self._ply_btn = QPushButton("Save .PLY")
        self._ply_btn.setFont(_font(9))
        self._ply_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._ply_btn.setStyleSheet(_btn_style(C_PANEL, C_TEXT_MED, C_TEXT_DIM))
        self._ply_btn.clicked.connect(lambda: self._save("ply"))
        self._ply_btn.setVisible(False)
        btns.addWidget(self._ply_btn)

        self._obj_btn = QPushButton("Save .OBJ")
        self._obj_btn.setFont(_font(9))
        self._obj_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._obj_btn.setStyleSheet(_btn_style(C_PANEL, C_TEXT_MED, C_TEXT_DIM))
        self._obj_btn.clicked.connect(lambda: self._save("obj"))
        self._obj_btn.setVisible(False)
        btns.addWidget(self._obj_btn)

        self._close_btn = QPushButton("Close")
        self._close_btn.setFont(_font(9))
        self._close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._close_btn.setStyleSheet(_btn_style(C_PANEL, C_TEXT_MED, C_TEXT_DIM))
        self._close_btn.clicked.connect(self.close)
        btns.addWidget(self._close_btn)
        btns.addStretch(1)
        root.addLayout(btns)

        # Swap preview label for the cloud canvas lazily in view mode — it
        # sits directly under the preview so the layout doesn't jump.
        self._cloud_canvas = _CloudCanvas(self)
        self._cloud_canvas.setFixedSize(520, 340)
        self._cloud_canvas.setVisible(False)
        root.insertWidget(1, self._cloud_canvas, 0, Qt.AlignmentFlag.AlignCenter)

    # -- scan phase --------------------------------------------------------
    def begin_scan(self) -> None:
        self._scanner.begin_scan()
        self._scan_active = True
        self._set_phase("SCAN")

    def _set_phase(self, phase: str) -> None:
        scanning = phase == "SCAN"
        self._phase_lbl.setText(phase)
        self._preview.setVisible(scanning)
        self._cloud_canvas.setVisible(not scanning)
        self._finish_btn.setVisible(scanning)
        self._rescan_btn.setVisible(not scanning)
        self._ply_btn.setVisible(not scanning)
        self._obj_btn.setVisible(not scanning)
        if scanning:
            self._cloud_canvas.stop_spin()
        else:
            self._cloud_canvas.start_spin()

    def _on_tick(self) -> None:
        if self._scan_active:
            self._scanner.step()
            self._update_preview()
            st = self._scanner.status()
            self._status_lbl.setText(
                f"frames {st.frames}   points {st.points:,}   "
                f"IR contrast {st.ir_contrast:.2f}   obj px {st.object_pixels}"
            )
            self._finish_btn.setEnabled(st.points >= MIN_SCAN_POINTS)
            if st.last_error:
                self._hint_lbl.setText(st.last_error)
            elif st.plane_locked:
                self._hint_lbl.setText(
                    "Tabletop locked — rotate the object slowly so every side is seen."
                )
            else:
                self._hint_lbl.setText(
                    "Hold still… fitting the tabletop plane. Keep the object in frame."
                )
        else:
            self._update_view_status()

    def _update_preview(self) -> None:
        try:
            pv = self._scanner.preview()
            if pv is None:
                return
            ir, mask = pv.get("ir"), pv.get("mask")
            if ir is None:
                return
            irf = ir.astype(np.float32)
            hi = np.percentile(irf, 99) or 1.0
            base = np.clip(irf / hi, 0.0, 1.0)
            # Cyan-tinted IR intensity.
            rgb = np.stack([
                20 + 60 * base,
                80 + 160 * base,
                120 + 135 * base,
            ], axis=2).astype(np.uint8)
            if mask is not None and mask.shape == ir.shape and mask.any():
                m = mask.astype(bool)
                rgb[m] = (235, 250, 255)     # object region: hot white
            h, w = rgb.shape[:2]
            qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
            pm = QPixmap.fromImage(qimg).scaled(
                520, 340, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation)
            self._preview.setPixmap(pm)
        except Exception as e:
            log.debug("[kinect] preview render skipped: %s", e)

    # -- view phase --------------------------------------------------------
    def _finish_scan(self) -> None:
        cloud, st = self._scanner.finish()
        if cloud.shape[0] < MIN_SCAN_POINTS:
            self._hint_lbl.setText(
                f"Scan too sparse ({cloud.shape[0]:,} points) — keep the object in frame and rotate it."
            )
            self._scanner.begin_scan()
            return
        self._cloud = cloud
        store_cloud(cloud)
        self._scan_active = False
        self._cloud_canvas.set_cloud(cloud)
        self._set_phase("VIEW")
        self._status_lbl.setText(
            f"scan complete — {cloud.shape[0]:,} points, {st.frames} frames"
        )
        self._hint_lbl.setText("Drag to orbit · wheel-free inspection · export below.")
        self._on_status(f"3D scan complete: {cloud.shape[0]:,} points captured.")

    def _update_view_status(self) -> None:
        n = self._cloud_canvas.point_count()
        if n:
            self._status_lbl.setText(f"{n:,} points — drag to rotate")

    def _rescan(self) -> None:
        self._cloud = None
        self._cloud_canvas.set_cloud(np.empty((0, 4), np.float32))
        self._scan_active = True
        self.begin_scan()

    # -- export ------------------------------------------------------------
    def _save(self, fmt: str) -> None:
        if self._cloud is None or self._cloud.shape[0] == 0:
            return
        SCAN_DIR.mkdir(parents=True, exist_ok=True)
        default = str(SCAN_DIR / f"kinect_scan_{time.strftime('%Y%m%d_%H%M%S')}.{fmt}")
        path, _ = QFileDialog.getSaveFileName(
            self, f"Save {fmt.upper()} scan", default,
            f"{fmt.upper()} scan (*.{fmt});;All files (*)",
        )
        if not path:
            return
        try:
            n = geometry.export_ply(path, self._cloud) if fmt == "ply" \
                else geometry.export_obj(path, self._cloud)
            self._status_lbl.setText(f"saved {n:,} points → {path}")
            self._on_status(f"Scan exported to {path}.")
            self.saved.emit(path, n)
        except Exception as e:
            log.warning("[kinect] export failed: %s", e)
            self._status_lbl.setText(f"export failed: {e}")

    # -- window plumbing ---------------------------------------------------
    def show_centered(self, parent: Optional[QWidget]) -> None:
        try:
            if parent is not None:
                gp = parent.mapToGlobal(QPoint(0, 0))
                self.move(
                    gp.x() + (parent.width() - self.width()) // 2,
                    gp.y() + (parent.height() - self.height()) // 2,
                )
            self.show()
            raise_overlay(self, parent)
            self.activateWindow()
        except Exception:
            self.move(120, 120)
            self.show()

    def hideEvent(self, e):
        self._cloud_canvas.stop_spin()
        super().hideEvent(e)

    def showEvent(self, e):
        if not self._scan_active:
            self._cloud_canvas.start_spin()
        super().showEvent(e)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_pos = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if self._dragging and e.buttons() == Qt.MouseButton.LeftButton:
            self.move(e.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, e):
        self._dragging = False

    def closeEvent(self, e):
        try:
            if self._scan_active:
                self._scanner.discard()
            self._tick.stop()
            self._cloud_canvas.stop_spin()
        except Exception:
            pass
        self.closed.emit()
        super().closeEvent(e)
