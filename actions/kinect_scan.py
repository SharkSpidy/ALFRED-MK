# actions/kinect_scan.py
"""
Kinect integration: IR-based object detection + guided 3D scanning.

Modes:
    scan    (default) — open the 3D scanner window. The user places the object
                        on a flat surface, the system locks onto the tabletop
                        plane (depth) while the IR stream drives object
                        detection, and the user rotates the object so all
                        sides are captured. Finish & View shows the cloud in a
                        small separate window with .ply/.obj export.
    detect           — one-shot check: is there an object on the surface in
                        front of the Kinect right now? Reports size estimate
                        and confidence.
    status           — report which Kinect backend/driver is available.
    view             — reopen the most recent scan cloud in the viewer.

Requires a Kinect sensor plus one backend SDK (see core/kinect/drivers.py).
Everything degrades gracefully — with no sensor the tool explains what to
install instead of crashing.
"""

from __future__ import annotations

from typing import Any, Optional


def _central_widget(player) -> Optional[Any]:
    try:
        if player is not None and hasattr(player, "_win") and hasattr(player._win, "centralWidget"):
            return player._win.centralWidget()
        if player is not None and hasattr(player, "centralWidget"):
            return player.centralWidget()
    except Exception:
        pass
    return None


def _log(player, msg: str) -> None:
    try:
        if player is not None and hasattr(player, "write_log"):
            player.write_log(f"ALFRED: {msg}")
    except Exception:
        pass


def _open_window(scanner, player, speak_msg: str) -> str:
    """Create/show the scan window on the GUI thread."""
    from core.gui_thread import run_on_gui_thread

    def _show():
        try:
            from core.kinect.viewer import KinectScanWindow
            parent = _central_widget(player)
            win = KinectScanWindow(scanner, parent=parent,
                                   on_status=lambda m: _log(player, m))
            win.begin_scan()
            win.show_centered(parent)
            if player is not None:
                player._kinect_scan_window = win   # keep a reference (no GC)
        except Exception as e:
            _log(player, f"Scanner window failed: {e}")

    run_on_gui_thread(_show)
    return speak_msg


def _status_text() -> str:
    try:
        from core.kinect import drivers
        st = drivers.backend_status()
        installed = [k for k, v in st.items() if v]
        if not installed:
            return ("No Kinect Python backend is installed, sir. Install "
                    "pykinect2 (Kinect v2) or PyKinect (Kinect v1) plus the "
                    "matching Kinect SDK.")
        dev = drivers.shared_device()
        if dev is not None:
            return f"Kinect active via {dev.backend_name}, sir. Ready to scan."
        return f"Kinect backend available ({', '.join(installed)}), sir. Say '3D scan' to begin."
    except Exception as e:
        return f"Kinect status check failed: {e}"


def _detect(player) -> str:
    try:
        from core.kinect import drivers, geometry
        dev = drivers.open_device()
        frame = dev.wait_frame(timeout_s=6.0)
        if frame is None:
            return "I can't see any Kinect frames, sir. Check the sensor connection."
        # Average a few frames for a stable detection.
        best = None
        for _ in range(5):
            f = dev.latest_frame()
            if f is not None:
                r = geometry.detect_object(f)
                if best is None or r.get("confidence", 0) > best.get("confidence", 0):
                    best = r
        if best is None or not best.get("detected"):
            reason = (best or {}).get("reason", "no object found")
            return f"No object detected, sir — {reason}."
        size = best.get("size_m", [0, 0, 0])
        return (f"Object detected, sir. Confidence {int(best['confidence'] * 100)} percent, "
                f"roughly {size[0] * 100:.1f} by {size[1] * 100:.1f} by {size[2] * 100:.1f} centimetres. "
                f"Say '3D scan' to capture it.")
    except drivers.KinectUnavailable as e:
        return f"Kinect unavailable, sir. {e}"
    except Exception as e:
        return f"Detection failed, sir: {e}"


def _view_last(player) -> str:
    try:
        from core.kinect import scanner as kscanner
        cloud = kscanner.last_cloud()
        if cloud is None or cloud.shape[0] == 0:
            return "No previous scan to display, sir. Run a 3D scan first."
        parent = _central_widget(player)

        def _show():
            from core.kinect.viewer import KinectScanWindow
            win = KinectScanWindow(kscanner.get_scanner(), parent=parent,
                                   on_status=lambda m: _log(player, m))
            win._cloud = cloud
            win._cloud_canvas.set_cloud(cloud)
            win._scan_active = False
            win._set_phase("VIEW")
            win.show_centered(parent)
            if player is not None:
                player._kinect_scan_window = win

        from core.gui_thread import run_on_gui_thread
        run_on_gui_thread(_show)
        return f"Displaying your last scan — {cloud.shape[0]:,} points, sir."
    except Exception as e:
        return f"Could not open the scan viewer, sir: {e}"


def action(parameters: dict, player=None, speak=None, response=None, session_memory=None) -> str:
    mode = str(parameters.get("mode") or "scan").strip().lower()
    try:
        if mode in ("status", "info"):
            return _status_text()
        if mode in ("detect", "detect_object", "check"):
            return _detect(player)
        if mode in ("view", "show", "open"):
            return _view_last(player)

        # Default: start a guided scan session.
        from core.kinect import scanner as kscanner
        from core.kinect import drivers
        try:
            drivers.open_device()          # fail fast with setup instructions
        except drivers.KinectUnavailable as e:
            return f"Kinect unavailable, sir. {e}"
        scanner = kscanner.get_scanner(
            on_status=lambda m: _log(player, m))
        return _open_window(
            scanner, player,
            "3D scanner ready, sir. Place the object on a flat surface in front of the sensor, "
            "then rotate it slowly so I can see every side.",
        )
    except Exception as e:
        return f"Kinect scan failed, sir: {e}"


TOOL = {
    "name": "kinect_scan",
    "description": (
        "3D scan a physical object with the Kinect sensor, or detect whether an "
        "object is present on a flat surface using the IR/depth streams. Use for "
        "'3D scan', 'scan this object', 'scan object', 'is there an object', "
        "'kinect status'. NOT for screen scanning or document scanning."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "mode": {
                "type": "STRING",
                "description": (
                    "scan (default, opens the 3D scanner window), detect "
                    "(one-shot object-presence check), status (driver info), "
                    "view (reopen last scan)."
                ),
            }
        },
        "required": [],
    },
    "behavior": "NON_BLOCKING",
    "scheduling": "WHEN_IDLE",
    "handler": action,
}
