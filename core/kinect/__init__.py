"""
core/kinect — Kinect IR + depth sensing: object detection and 3D scanning.

Modules:
    drivers   — hardware abstraction over pykinect2 (Kinect v2) and PyKinect
                (Kinect v1 / Kinect for Windows SDK 1.8). Graceful degradation:
                every backend import is guarded, and open_device() raises
                KinectUnavailable with setup instructions when no SDK is found.
    geometry  — depth→point-cloud projection, tabletop plane fitting (RANSAC),
                object masking, IR-intensity object detection, voxel
                downsampling, PLY/OBJ export.
    scanner   — KinectScanner: owns the capture thread and a scan session that
                accumulates object-only points while the user rotates the object.
    viewer    — KinectScanWindow: small frameless window with a live IR/depth
                preview while scanning and a software-rendered rotating point
                cloud once the scan finishes.
"""
