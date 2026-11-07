"""
core/kinect/drivers.py — Kinect hardware abstraction.

Supported backends (first match wins):

    pykinect2  — Kinect for Windows v2 sensor (Model 1517). Requires the
                 "Kinect for Windows SDK 2.0" runtime plus `pip install
                 pykinect2`. Provides 512x424 depth (mm), 512x424 IR and
                 1080p colour. This is the primary path.

    PyKinect   — Kinect for Windows v1 / Xbox 360 Kinect. Requires "Kinect for
                 Windows SDK 1.8" plus `pip install PyKinect`. Provides
                 640x480 depth + IR. Depth arrives as raw disparity and is
                 converted to mm with the standard approximation.

    freenect   — libfreenect bindings (`pip install freenect`), mostly useful
                 on Linux/macOS with an Xbox 360 Kinect. Experimental.

Every backend import is guarded: importing this module NEVER fails. Call
open_device() to get a KinectDevice; it raises KinectUnavailable (with setup
instructions) when no sensor/SDK is present.

Frame data is exchanged through a latest-frame slot guarded by a lock, so the
GUI thread can poll at any rate without blocking the capture thread.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

log = logging.getLogger(__name__)

# ── Named constants ──────────────────────────────────────────────────────────
DEPTH_MIN_MM = 400          # ignore anything closer than this (lens flare zone)
DEPTH_MAX_MM = 2500         # ignore anything beyond reliable Kinect range
FRAME_STALE_S = 1.0         # a frame older than this is treated as "no frame"


class KinectUnavailable(RuntimeError):
    """Raised when no Kinect backend/sensor is available."""


@dataclass
class KinectFrame:
    """One captured frame bundle. Arrays may be None when a stream is absent."""
    depth: Optional[np.ndarray] = None   # uint16, millimetres (0 = invalid)
    ir: Optional[np.ndarray] = None      # uint16, IR intensity (sensor units)
    color: Optional[np.ndarray] = None   # uint8 BGR/BGRA, optional
    stamp: float = field(default_factory=time.monotonic)
    backend: str = ""
    width: int = 0
    height: int = 0

    @property
    def age(self) -> float:
        return time.monotonic() - self.stamp


# ── Backend capability probes ────────────────────────────────────────────────

def _probe_pykinect2() -> bool:
    try:
        import pykinect2  # noqa: F401
        return True
    except Exception:
        return False


def _probe_pykinect() -> bool:
    try:
        import PyKinect  # noqa: F401
        return True
    except Exception:
        return False


def _probe_freenect() -> bool:
    try:
        import freenect  # noqa: F401
        return True
    except Exception:
        return False


def backend_status() -> dict[str, Any]:
    """Report which backends are importable (does not touch hardware)."""
    return {
        "pykinect2": _probe_pykinect2(),
        "PyKinect": _probe_pykinect(),
        "freenect": _probe_freenect(),
    }


# ── pykinect2 backend (Kinect v2) ────────────────────────────────────────────

class _Pykinect2Backend:
    name = "pykinect2"

    def __init__(self) -> None:
        from pykinect2 import PyKinectRuntime, PyKinectSDKError  # type: ignore
        self._PyKinectSDKError = PyKinectSDKError
        self._rt = PyKinectRuntime(1)   # 1-based sensor index
        self._rt.open()
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self._rt.close()
            except Exception:
                pass

    def _as_gray(self, frame: Any, h: int, w: int) -> Optional[np.ndarray]:
        if frame is None:
            return None
        arr = np.asarray(frame)
        if arr.size == 0:
            return None
        arr = arr.reshape(-1)
        if arr.size < h * w:
            return None
        arr = arr[: h * w].reshape(h, w)
        return arr.astype(np.uint16, copy=False)

    def read(self) -> KinectFrame:
        rt = self._rt
        # pykinect2 exposes has_new_*_frame() getters; some builds also need an
        # explicit update()/process() pump — call it when present.
        pump = getattr(rt, "update", None) or getattr(rt, "process", None)
        if callable(pump):
            try:
                pump()
            except Exception:
                pass

        depth = ir = color = None
        try:
            if getattr(rt, "has_new_depth_frame", lambda: True)():
                depth = self._as_gray(rt.get_last_depth_frame(), 424, 512)
        except Exception:
            pass
        try:
            if getattr(rt, "has_new_ir_frame", lambda: True)():
                ir = self._as_gray(rt.get_last_ir_frame(), 424, 512)
        except Exception:
            pass
        try:
            if getattr(rt, "has_new_color_frame", lambda: False)():
                c = np.asarray(rt.get_last_color_frame())
                if c.size:
                    color = c.reshape(-1, c.shape[-1] if c.ndim > 1 else 4)[:1080 * 1920]
        except Exception:
            pass

        h, w = (depth.shape if depth is not None else (424, 512))
        return KinectFrame(depth=depth, ir=ir, color=color,
                           backend=self.name, width=w, height=h)


# ── PyKinect backend (Kinect v1 / SDK 1.8) ───────────────────────────────────

class _PyKinectBackend:
    name = "PyKinect"

    def __init__(self) -> None:
        from PyKinect import PyKinectRuntime  # type: ignore
        self._rt = PyKinectRuntime(PyKinectRuntime.FrameSource_Type.eDepth | PyKinectRuntime.FrameSource_Type.eIR)
        self._rt.open()
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self._rt.close()
            except Exception:
                pass

    def read(self) -> KinectFrame:
        rt = self._rt
        try:
            rt.update()
        except Exception:
            pass

        depth = ir = None
        try:
            if rt.has_new_depth_frame():
                d = np.asarray(rt.get_last_depth_frame())
                if d.size:
                    # v1 depth is 11-bit disparity; approximate mm = 8000 / raw.
                    d = d.reshape(d.shape[-2], d.shape[-1]) if d.ndim > 2 else d
                    raw = d.astype(np.float32)
                    depth = np.where(raw > 0, np.clip(8000.0 / np.maximum(raw, 1.0), 0, 8000), 0).astype(np.uint16)
        except Exception:
            pass
        try:
            if rt.has_new_ir_frame():
                i = np.asarray(rt.get_last_ir_frame())
                if i.size:
                    ir = i.reshape(i.shape[-2], i.shape[-1]) if i.ndim > 2 else i
                    ir = ir.astype(np.uint16, copy=False)
        except Exception:
            pass

        h, w = (depth.shape if depth is not None else (480, 640))
        return KinectFrame(depth=depth, ir=ir, backend=self.name, width=w, height=h)


# ── freenect backend (experimental) ──────────────────────────────────────────

class _FreenectBackend:
    name = "freenect"

    def __init__(self) -> None:
        import freenect  # type: ignore
        self._freenect = freenect
        self._dev = freenect.init()
        freenect.set_depth_mode(self._dev, freenect.RES_640x480)
        freenect.set_ir_mode(self._dev, freenect.RES_640x480)
        freenect.start_depth(self._dev)
        freenect.start_ir(self._dev)
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self._freenect.stop_depth(self._dev)
                self._freenect.stop_ir(self._dev)
                self._freenect.close_device(self._dev)
            except Exception:
                pass

    def read(self) -> KinectFrame:
        import freenect  # type: ignore
        depth_raw = freenect.sync_get_depth(self._dev, 0)
        ir_raw = freenect.sync_get_ir(self._dev, 0)
        depth = np.asarray(depth_raw).reshape(480, 640).astype(np.uint16)
        ir = np.asarray(ir_raw).reshape(480, 640).astype(np.uint16)
        return KinectFrame(depth=depth, ir=ir, backend=self.name, width=640, height=480)


# ── Device wrapper: capture thread + latest-frame slot ───────────────────────

_BACKENDS = (
    ("pykinect2", _Pykinect2Backend),
    ("PyKinect", _PyKinectBackend),
    ("freenect", _FreenectBackend),
)

_SETUP_HELP = (
    "No Kinect backend available. Install one of:\n"
    "  • Kinect v2:  Kinect for Windows SDK 2.0 + `pip install pykinect2`\n"
    "  • Kinect v1:  Kinect for Windows SDK 1.8  + `pip install PyKinect`\n"
    "  • Xbox 360:   libfreenect + `pip install freenect` (experimental)"
)


class KinectDevice:
    """Thread-safe Kinect capture device. Poll latest_frame() from any thread."""

    def __init__(self, backend: Any) -> None:
        self._backend = backend
        self.backend_name = backend.name
        self._lock = threading.Lock()
        self._frame: Optional[KinectFrame] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.last_error: str = ""

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, name="kinect-capture", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        try:
            self._backend.close()
        except Exception:
            pass

    def __enter__(self) -> "KinectDevice":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- capture loop ------------------------------------------------------
    def _capture_loop(self) -> None:
        while self._running:
            try:
                frame = self._backend.read()
                if frame is not None and (frame.depth is not None or frame.ir is not None):
                    with self._lock:
                        self._frame = frame
            except Exception as e:
                self.last_error = str(e)
                log.warning("[kinect] capture error: %s", e)
                time.sleep(0.25)

    # -- accessors ---------------------------------------------------------
    def latest_frame(self, max_age_s: float = FRAME_STALE_S) -> Optional[KinectFrame]:
        with self._lock:
            f = self._frame
        if f is None or f.age > max_age_s:
            return None
        return f

    def wait_frame(self, timeout_s: float = 5.0) -> Optional[KinectFrame]:
        """Block until a frame arrives or timeout (used by one-shot detect)."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            f = self.latest_frame(max_age_s=timeout_s)
            if f is not None:
                return f
            time.sleep(0.02)
        return None


_device_lock = threading.Lock()
_active_device: Optional[KinectDevice] = None


def open_device() -> KinectDevice:
    """Open the first available Kinect backend. Raises KinectUnavailable."""
    global _active_device
    with _device_lock:
        if _active_device is not None:
            return _active_device

        errors = []
        for name, cls in _BACKENDS:
            try:
                probe = _probe_pykinect2 if name == "pykinect2" else \
                        _probe_pykinect if name == "PyKinect" else _probe_freenect
                if not probe():
                    errors.append(f"{name}: python package not installed")
                    continue
                backend = cls()
                dev = KinectDevice(backend)
                dev.start()
                log.info("[kinect] opened via %s", name)
                _active_device = dev
                return dev
            except Exception as e:
                errors.append(f"{name}: {e}")
                log.info("[kinect] backend %s unavailable: %s", name, e)

        raise KinectUnavailable(_SETUP_HELP + "\n\nDetails: " + "; ".join(errors))


def close_device() -> None:
    """Release the shared Kinect device (safe to call repeatedly)."""
    global _active_device
    with _device_lock:
        dev, _active_device = _active_device, None
    if dev is not None:
        dev.close()


def shared_device() -> Optional[KinectDevice]:
    """Return the active device without opening one (None if not started)."""
    return _active_device
