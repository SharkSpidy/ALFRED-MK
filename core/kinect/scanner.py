"""
core/kinect/scanner.py — Guided 3D scan session.

KinectScanner owns a scan lifecycle on top of a shared KinectDevice:

    begin_scan()  — start accumulating; the tabletop plane is fitted from the
                    first clean frames and then held (refit every
                    PLANE_REFIT_EVERY frames to compensate for drift).
    step()        — called ~30x/sec by the viewer: grabs the newest frame,
                    extracts object-only points (table subtracted by the plane
                    fit) and merges them into a voxel accumulator. IR intensity
                    is carried through as per-point shading.
    finish()      — outlier-trims and returns the cloud (N,4: x,y,z,intensity).
    discard()     — throws the session away.

The user places the object on a flat surface and rotates it; every view adds
new voxels until the accumulated cloud covers all sides. The plane is fitted
once per session so the tabletop itself never enters the cloud.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from . import geometry
from .drivers import KinectDevice, KinectFrame, open_device, DEPTH_MIN_MM, DEPTH_MAX_MM

log = logging.getLogger(__name__)

# Min frames between plane refits is in geometry.PLANE_REFIT_EVERY; the plane
# is only refit while it still explains the scene (drift compensation).
PLANE_HOLD_FRAMES = geometry.PLANE_REFIT_EVERY
STEP_STRIDE = 2          # every 2nd pixel — density comes from frame overlap


@dataclass
class ScanStatus:
    "Snapshot of the running scan, safe to read from the GUI thread."""
    active: bool = False
    frames: int = 0
    points: int = 0
    points_seen: int = 0
    plane_locked: bool = False
    last_error: str = ""
    ir_contrast: float = 0.0
    object_pixels: int = 0
    started_at: float = 0.0


class KinectScanner:
    """One scan session at a time. NOT thread-safe for step(); the viewer's
    QTimer calls step() on the GUI thread while the capture thread in
    KinectDevice fills latest_frame()."""

    def __init__(self, device: KinectDevice,
                 on_status: Optional[Callable[[str], None]] = None) -> None:
        self.device = device
        self.on_status = on_status or (lambda m: None)
        self._acc = geometry.ScanAccumulator()
        self._plane: Optional[Tuple[np.ndarray, float]] = None
        self._plane_age = 0
        self._lock = threading.Lock()
        self._status = ScanStatus()
        self._last_frame_stamp = 0.0
        self._rng = np.random.default_rng()

    # -- lifecycle ---------------------------------------------------------
    def begin_scan(self) -> None:
        with self._lock:
            self._acc.reset()
            self._plane = None
            self._plane_age = 0
            self._status = ScanStatus(active=True, started_at=time.monotonic())
        self.on_status("Scan started — place the object on the flat surface and rotate it slowly.")

    def discard(self) -> None:
        with self._lock:
            self._acc.reset()
            self._plane = None
            self._status = ScanStatus()

    def is_active(self) -> bool:
        return self._status.active

    # -- per-frame step ----------------------------------------------------
    def step(self) -> Optional[Dict[str, Any]]:
        """Consume the newest frame into the accumulator. Returns debug info."""
        frame = self.device.latest_frame()
        if frame is None or frame.depth is None:
            return None
        if frame.stamp == self._last_frame_stamp:
            return None                      # same frame as last step
        self._last_frame_stamp = frame.stamp

        intr = geometry.intrinsics_for(frame.backend, frame.width, frame.height)

        # Fit (or refit) the tabletop plane from a strided cloud.
        if self._plane is None or self._plane_age >= PLANE_HOLD_FRAMES:
            pts_all, _ = geometry.depth_to_points(frame.depth, intr, step=STEP_STRIDE * 2)
            if pts_all.shape[0] >= geometry.RANSAC_MIN_INLIERS:
                fit = geometry.fit_plane(pts_all, rng=self._rng)
                if fit is not None:
                    self._plane = fit[0:2]
                    self._plane_age = 0
        self._plane_age += 1

        if self._plane is None:
            with self._lock:
                self._status.last_error = "waiting for tabletop plane…"
            return {"waiting": "plane"}

        mask = geometry.object_mask(frame.depth, self._plane, intr)
        n_px = int(mask.sum())
        if n_px < geometry.MIN_OBJECT_PIXELS:
            with self._lock:
                self._status.object_pixels = n_px
            return {"object_pixels": n_px}

        # Object points + per-point IR intensity as shading.
        obj_depth = np.where(mask, frame.depth, 0)
        pts, pix = geometry.depth_to_points(obj_depth, intr, step=STEP_STRIDE)
        if pts.shape[0] == 0:
            return None

        inten = None
        if frame.ir is not None and frame.ir.shape == frame.depth.shape:
            ir_flat = frame.ir.reshape(-1).astype(np.float32)
            lo, hi = np.percentile(ir_flat[ir_flat > 0], [5, 95]) if (ir_flat > 0).any() else (0, 1)
            span = max(hi - lo, 1.0)
            inten = np.clip((ir_flat[pix] - lo) / span, 0.0, 1.0)

        added = self._acc.add(pts, inten)

        _, _, contrast = geometry.ir_stats(frame.ir, mask)
        with self._lock:
            self._status.frames += 1
            self._status.points = self._acc.count()
            self._status.points_seen = self._acc.points_seen
            self._status.plane_locked = True
            self._status.ir_contrast = round(contrast, 3)
            self._status.object_pixels = n_px
            self._status.last_error = ""
        return {"added_voxels": added, "points": self._acc.count()}

    def preview(self) -> Optional[Dict[str, Any]]:
        """Latest frame + current object mask for the viewer's live preview.

        Does NOT accumulate — safe to call every paint tick.
        """
        frame = self.device.latest_frame()
        if frame is None or frame.depth is None:
            return None
        intr = geometry.intrinsics_for(frame.backend, frame.width, frame.height)
        if self._plane is not None:
            mask = geometry.object_mask(frame.depth, self._plane, intr)
        else:
            mask = np.zeros(frame.depth.shape, dtype=bool)
        return {"ir": frame.ir, "depth": frame.depth, "mask": mask,
                "backend": frame.backend}

    # -- finish ------------------------------------------------------------
    def finish(self) -> Tuple[np.ndarray, ScanStatus]:
        with self._lock:
            cloud = self._acc.cloud()
            status = ScanStatus(**self._status.__dict__)
            status.active = False
            self._status.active = False
        if cloud.shape[0] == 0:
            return cloud, status
        cloud = geometry.trim_statistical(cloud)
        return cloud, status

    def status(self) -> ScanStatus:
        with self._lock:
            return ScanStatus(**self._status.__dict__)

    def reset_for_rescan(self) -> None:
        self.begin_scan()


# ── Session singleton (the viewer owns the active scan) ──────────────────────

_lock = threading.Lock()
_scanner: Optional[KinectScanner] = None


def get_scanner(on_status: Optional[Callable[[str], None]] = None) -> KinectScanner:
    """Return the shared scanner, opening the Kinect device if needed."""
    global _scanner
    with _lock:
        if _scanner is None:
            dev = open_device()
            _scanner = KinectScanner(dev, on_status=on_status)
        elif on_status is not None:
            _scanner.on_status = on_status
        return _scanner


def last_cloud() -> Optional[np.ndarray]:
    """Most recently finished cloud, kept for the 'view scan' re-open path."""
    return _LAST_CLOUD[0]


_LAST_CLOUD: list = [None]


def store_cloud(cloud: np.ndarray) -> None:
    _LAST_CLOUD[0] = cloud
