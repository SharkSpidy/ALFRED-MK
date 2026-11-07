"""
core/kinect/geometry.py — Depth→geometry pipeline for Kinect scanning.

Pipeline per frame:
    1. depth (mm) → metric point cloud via factory depth-camera intrinsics
       (overridable through config/kinect.json for unit-to-unit calibration).
    2. RANSAC plane fit finds the dominant surface — the flat table the object
       sits on. Points whose signed distance to that plane exceeds MARGIN_M on
       the camera side are "the object"; everything else (table, wall, floor)
       is discarded. This is what makes the scanner read ONLY the object.
    3. The Kinect IR stream refines detection: IR intensity contrast between the
       masked object region and its surroundings feeds the detection
       confidence, and per-point IR intensity becomes the point cloud's
       shading channel for the viewer.

Units: all geometry is in metres. Depth arrives in millimetres and is divided
by 1000 at projection time.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .drivers import DEPTH_MIN_MM, DEPTH_MAX_MM, KinectFrame

log = logging.getLogger(__name__)

# ── Named constants ──────────────────────────────────────────────────────────
PLANE_INLIER_M = 0.008        # RANSAC inlier band around the tabletop (8 mm)
OBJECT_MARGIN_M = 0.012       # min height above the plane to count as object
RANSAC_ITERS = 240
RANSAC_MIN_INLIERS = 800      # below this the "plane" isn't a table — no scan
PLANE_REFIT_EVERY = 30        # frames between plane refits during a scan
MIN_OBJECT_PIXELS = 150       # smaller blobs are noise, not an object
MAX_CLOUD_POINTS = 600_000    # hard cap on accumulated scan density
VOXEL_SIZE_M = 0.004          # 4 mm voxel grid for accumulation/downsampling

# Factory depth-camera intrinsics (fx, fy, cx, cy). Kinect v2 depth is
# 512x424, Kinect v1 depth is 640x480. Values are the commonly published
# factory defaults — override per-unit in config/kinect.json after a
# calibration scan if proportions look stretched.
_INTRINSICS: Dict[str, Tuple[float, float, float, float]] = {
    "pykinect2": (365.0, 365.0, 255.0, 205.0),     # 512 x 424
    "PyKinect": (525.0, 525.0, 320.0, 240.0),      # 640 x 480 (approximate)
    "freenect": (525.0, 525.0, 320.0, 240.0),      # 640 x 480 (approximate)
}

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "kinect.json"
_intrinsics_cache: Optional[Dict[str, Any]] = None


def _load_intrinsics_override() -> Dict[str, Any]:
    global _intrinsics_cache
    if _intrinsics_cache is not None:
        return _intrinsics_cache
    _intrinsics_cache = {}
    try:
        if _CONFIG_PATH.exists():
            _intrinsics_cache = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("[kinect] could not read %s: %s", _CONFIG_PATH, e)
    return _intrinsics_cache


def intrinsics_for(backend: str, width: int, height: int) -> Tuple[float, float, float, float]:
    """Return (fx, fy, cx, cy) for the backend, honouring config overrides."""
    ov = _load_intrinsics_override()
    per_backend = ov.get(backend) or ov.get("default") or {}
    fx = float(per_backend.get("fx", 0) or 0)
    fy = float(per_backend.get("fy", 0) or 0)
    cx = float(per_backend.get("cx", 0) or 0)
    cy = float(per_backend.get("cy", 0) or 0)
    if not (fx > 0 and fy > 0):
        fx, fy, cx, cy = _INTRINSICS.get(backend, _INTRINSICS["pykinect2"])
    # Scale principal point if the frame size differs from the reference size.
    ref = {"pykinect2": (512, 424), "PyKinect": (640, 480), "freenect": (640, 480)}
    rw, rh = ref.get(backend, (width, height))
    if width != rw or height != rh:
        sx, sy = width / rw, height / rh
        cx, cy = cx * sx, cy * sy
        fx, fy = fx * sx, fy * sy
    return fx, fy, cx, cy


# ── Depth → points ───────────────────────────────────────────────────────────

def valid_depth_mask(depth: np.ndarray) -> np.ndarray:
    return (depth > DEPTH_MIN_MM) & (depth < DEPTH_MAX_MM)


def depth_to_points(depth_mm: np.ndarray, intr: Tuple[float, float, float, float],
                    step: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Project a depth frame to camera-space points (metres).

    Returns (points Nx3 float32, flat pixel indices into the depth frame).
    """
    fx, fy, cx, cy = intr
    valid = valid_depth_mask(depth_mm)
    if step > 1:
        valid[: step - 1, :] = False
        valid[:, : step - 1] = False
    v_idx, u_idx = np.nonzero(valid)
    if v_idx.size == 0:
        return np.empty((0, 3), np.float32), np.empty(0, np.int64)

    z = depth_mm[v_idx, u_idx].astype(np.float32) / 1000.0
    x = (u_idx.astype(np.float32) - cx) * z / fx
    y = (v_idx.astype(np.float32) - cy) * z / fy
    pts = np.stack([x, y, z], axis=1).astype(np.float32)
    pix = (v_idx.astype(np.int64) * depth_mm.shape[1] + u_idx.astype(np.int64))
    return pts, pix


def pixel_residuals(depth_mm: np.ndarray, plane: Tuple[np.ndarray, float],
                    intr: Tuple[float, float, float, float]) -> np.ndarray:
    """Signed distance (m) of every valid pixel's 3D point to the plane.

    Positive = on the camera side of the plane (i.e. ABOVE the table).
    """
    n, d = plane
    fx, fy, cx, cy = intr
    h, w = depth_mm.shape
    v, u = np.mgrid[0:h, 0:w]
    z = np.where(valid_depth_mask(depth_mm), depth_mm.astype(np.float32) / 1000.0, 0.0)
    x = (u.astype(np.float32) - cx) * z / fx
    y = (v.astype(np.float32) - cy) * z / fy
    return n[0] * x + n[1] * y + n[2] * z + d


def object_mask(depth_mm: np.ndarray, plane: Tuple[np.ndarray, float],
                intr: Tuple[float, float, float, float],
                margin_m: float = OBJECT_MARGIN_M) -> np.ndarray:
    """Boolean mask: pixels belonging to the object (above the table plane)."""
    if plane is None:
        return np.zeros(depth_mm.shape, dtype=bool)
    res = pixel_residuals(depth_mm, plane, intr)
    mask = (res > margin_m) & valid_depth_mask(depth_mm)
    # Restrict to a central region — the object is placed deliberately in view,
    # and this rejects ceiling/wall clutter at the frame edges.
    h, w = mask.shape
    y0, y1 = int(h * 0.08), int(h * 0.95)
    x0, x1 = int(w * 0.10), int(w * 0.90)
    out = np.zeros_like(mask)
    out[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    return out


# ── Plane fitting (RANSAC + least-squares refine) ────────────────────────────

def _plane_from_points(pts: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
    """Least-squares plane n·p + d = 0 through pts (Nx3)."""
    if pts.shape[0] < 3:
        return None
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[-1]
    n = n / (np.linalg.norm(n) + 1e-12)
    d = -float(n @ c)
    return n, d


def fit_plane(points: np.ndarray, rng: Optional[np.random.Generator] = None,
              iters: int = RANSAC_ITERS,
              thresh_m: float = PLANE_INLIER_M) -> Optional[Tuple[np.ndarray, float, int]]:
    """RANSAC-fit the dominant plane. Returns (normal, d, inlier_count) or None.

    The normal is oriented so it points TOWARD the camera (origin), making
    positive residuals mean 'above the surface'.
    """
    if points.shape[0] < RANSAC_MIN_INLIERS:
        return None
    rng = rng or np.random.default_rng()
    n_pts = points.shape[0]
    sample_n = min(n_pts, 20000)
    sub = points[rng.choice(n_pts, sample_n, replace=False)]

    best_inliers: Optional[np.ndarray] = None
    best_count = 0
    for _ in range(iters):
        idx = rng.choice(sample_n, 3, replace=False)
        p0, p1, p2 = sub[idx]
        n = np.cross(p1 - p0, p2 - p0)
        ln = np.linalg.norm(n)
        if ln < 1e-9:
            continue
        n /= ln
        d = -float(n @ p0)
        dist = np.abs(sub @ n + d)
        inl = dist < thresh_m
        count = int(inl.sum())
        if count > best_count:
            best_count = count
            best_inliers = inl

    if best_inliers is None or best_count < RANSAC_MIN_INLIERS:
        return None

    plane = _plane_from_points(sub[best_inliers])
    if plane is None:
        return None
    n, d = plane
    # Orient so the camera (origin) is on the positive-residual side: the
    # residual at the origin is d, and the object sits between camera and
    # table, so object points then read positive in pixel_residuals().
    if d < 0:
        n, d = -n, -d
    return n, d, best_count


# ── IR-assisted object detection ─────────────────────────────────────────────

def ir_stats(ir: Optional[np.ndarray], mask: np.ndarray) -> Tuple[float, float, float]:
    """Return (object_mean, background_mean, contrast) of the IR stream.

    IR is active illumination: the object interrupts the tabletop's return, so
    a real object produces a measurable intensity step against its surround.
    """
    if ir is None or not mask.any():
        return 0.0, 0.0, 0.0
    irf = ir.astype(np.float32)
    obj_mean = float(irf[mask].mean())
    dil = mask.copy()
    # Cheap 5x5-ish surround via roll-free dilation using maximum_filter-free
    # approach: pad + shifts.
    bg = np.zeros_like(mask)
    for dy in (-3, 0, 3):
        for dx in (-3, 0, 3):
            if dy == 0 and dx == 0:
                continue
            bg |= np.roll(np.roll(mask, dy, 0), dx, 1)
    bg &= ~mask
    if not bg.any():
        return obj_mean, obj_mean, 0.0
    bg_mean = float(irf[bg].mean())
    denom = max(abs(obj_mean), abs(bg_mean), 1.0)
    contrast = abs(obj_mean - bg_mean) / denom
    return obj_mean, bg_mean, contrast


def detect_object(frame: KinectFrame, plane: Optional[Tuple[np.ndarray, float]] = None,
                  fit_plane_if_needed: bool = True) -> Dict[str, Any]:
    """One-shot object detection from a frame. Returns a result dict."""
    if frame.depth is None:
        return {"detected": False, "reason": "no depth stream", "confidence": 0.0}
    intr = intrinsics_for(frame.backend, frame.width, frame.height)

    pts, pix = depth_to_points(frame.depth, intr, step=2)
    if pts.shape[0] < RANSAC_MIN_INLIERS:
        return {"detected": False, "reason": "not enough valid depth", "confidence": 0.0}

    pl = plane or (fit_plane(pts)[0:2] if fit_plane_if_needed else None)
    if pl is None:
        return {"detected": False, "reason": "no tabletop plane found — is the object on a flat surface in view?",
                "confidence": 0.0}

    mask = object_mask(frame.depth, pl, intr)
    n_px = int(mask.sum())
    if n_px < MIN_OBJECT_PIXELS:
        return {"detected": False, "reason": "nothing above the surface detected",
                "confidence": 0.0, "plane": pl}

    # Object extent in metres from masked pixels.
    sub_pts, _ = depth_to_points(np.where(mask, frame.depth, 0), intr, step=1)
    if sub_pts.shape[0] == 0:
        return {"detected": False, "reason": "mask produced no points", "confidence": 0.0}
    extent = sub_pts.max(axis=0) - sub_pts.min(axis=0)

    obj_ir, bg_ir, contrast = ir_stats(frame.ir, mask)
    # Confidence blends mask area (object presence) and IR contrast.
    area_score = min(1.0, n_px / 4000.0)
    conf = round(min(1.0, 0.55 * area_score + 0.45 * min(1.0, contrast * 4.0)), 2)

    ys, xs = np.nonzero(mask)
    return {
        "detected": True,
        "confidence": conf,
        "object_pixels": n_px,
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "size_m": [round(float(extent[i]), 3) for i in range(3)],
        "ir_contrast": round(contrast, 3),
        "ir_object_mean": round(obj_ir, 1),
        "ir_background_mean": round(bg_ir, 1),
        "plane": pl,
    }


# ── Voxel accumulation ───────────────────────────────────────────────────────

class ScanAccumulator:
    """Accumulates object-only points into a fixed-size voxel grid.

    Each voxel keeps the point with the highest IR intensity, so overlapping
    views of the same surface merge instead of doubling density. Points are
    (x, y, z) metres + intensity 0..1.
    """

    def __init__(self, voxel_m: float = VOXEL_SIZE_M, max_points: int = MAX_CLOUD_POINTS) -> None:
        self.voxel_m = voxel_m
        self.max_points = max_points
        self._voxels: Dict[Tuple[int, int, int], Tuple[float, float, float, float]] = {}
        self.frames_added = 0
        self.points_seen = 0

    def add(self, points: np.ndarray, intensity: Optional[np.ndarray] = None) -> int:
        """Add an Nx3 (metres) point batch. Returns number of new voxels."""
        if points.shape[0] == 0:
            return 0
        inv = 1.0 / self.voxel_m
        keys = np.floor(points * inv).astype(np.int64)
        if intensity is None:
            inten = np.full(points.shape[0], 0.5, np.float32)
        else:
            inten = intensity.astype(np.float32)

        # Collapse intra-batch duplicates first: sort by voxel key then
        # intensity so the last row of each key group is its max-intensity
        # point. Keys are packed into one int64 (offset keeps negatives safe).
        OFF = np.int64(1 << 20)
        B = np.int64(1 << 21)
        comb = ((keys[:, 0] + OFF) * B + (keys[:, 1] + OFF)) * B + (keys[:, 2] + OFF)
        order = np.lexsort((inten, comb))
        scomb = comb[order]
        _, grp_start = np.unique(scomb, return_index=True)
        grp_last = np.r_[grp_start[1:] - 1, scomb.size - 1]
        chosen = order[grp_last]

        added = 0
        vox = self._voxels
        for i in chosen:
            kx, ky, kz = int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2])
            key = (kx, ky, kz)
            cur = vox.get(key)
            if cur is None:
                if len(vox) < self.max_points:
                    vox[key] = (float(points[i, 0]), float(points[i, 1]),
                                float(points[i, 2]), float(inten[i]))
                    added += 1
            elif inten[i] > cur[3]:
                vox[key] = (float(points[i, 0]), float(points[i, 1]),
                            float(points[i, 2]), float(inten[i]))
        self.frames_added += 1
        self.points_seen += points.shape[0]
        return added

    def count(self) -> int:
        return len(self._voxels)

    def cloud(self) -> np.ndarray:
        """Return (N,4) float32 array: x, y, z (m), intensity 0..1."""
        if not self._voxels:
            return np.empty((0, 4), np.float32)
        arr = np.asarray(list(self._voxels.values()), dtype=np.float32)
        return arr

    def reset(self) -> None:
        self._voxels.clear()
        self.frames_added = 0
        self.points_seen = 0


def trim_statistical(cloud: np.ndarray, k: int = 8, std_mult: float = 2.0) -> np.ndarray:
    """Drop sparse outlier points using a cheap grid-neighbour test."""
    if cloud.shape[0] < 64:
        return cloud
    xyz = cloud[:, :3]
    inv = 1.0 / VOXEL_SIZE_M
    keys = np.floor(xyz * inv).astype(np.int64)
    keyset = set(map(tuple, keys.tolist()))
    keep = np.zeros(cloud.shape[0], dtype=bool)
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
               if not (dx == 0 and dy == 0 and dz == 0)]
    for i, key in enumerate(map(tuple, keys.tolist())):
        nbrs = sum((key[0] + dx, key[1] + dy, key[2] + dz) in keyset for dx, dy, dz in offsets)
        if nbrs >= max(2, k // 4):
            keep[i] = True
    return cloud[keep] if keep.any() else cloud


# ── Export ───────────────────────────────────────────────────────────────────

def export_ply(path: str | Path, cloud: np.ndarray) -> int:
    """Write a binary PLY with vertex colours derived from IR intensity."""
    cloud = np.asarray(cloud, dtype=np.float32)
    n = cloud.shape[0]
    if n == 0:
        return 0
    # Centre the cloud so the model sits at the origin.
    centre = cloud[:, :3].mean(axis=0)
    pts = cloud[:, :3] - centre

    def shade(inten: np.ndarray) -> np.ndarray:
        t = np.clip(inten, 0.0, 1.0)
        # Cyan→white ramp matching the HUD palette.
        r = (30 + 225 * t).astype(np.uint8)
        g = (120 + 135 * t).astype(np.uint8)
        b = (160 + 95 * t).astype(np.uint8)
        return np.stack([r, g, b], axis=1)

    colors = shade(cloud[:, 3] if cloud.shape[1] > 3 else np.full(n, 0.5))
    with open(path, "wb") as f:
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        )
        f.write(header.encode("ascii"))
        for i in range(n):
            f.write(pts[i].astype("<f4").tobytes())
            f.write(colors[i].tobytes())
    return n


def export_obj(path: str | Path, cloud: np.ndarray) -> int:
    """Write a plain OBJ (v lines only) for import into any 3D tool."""
    cloud = np.asarray(cloud, dtype=np.float32)
    n = cloud.shape[0]
    if n == 0:
        return 0
    centre = cloud[:, :3].mean(axis=0)
    pts = cloud[:, :3] - centre
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# ALFRED Kinect scan — {n} points — {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write("# Units: metres, centred at origin\n")
        for p in pts:
            f.write(f"v {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")
    return n
