"""Shared setup for the command line scripts.

Puts the project root on sys.path so ``import facetrack`` works when a script
is run directly, and collects the argument groups the scripts have in common.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from facetrack import config  # noqa: E402
from facetrack.detectors import DETECTOR_NAMES, get_detector  # noqa: E402
from facetrack.recognizers import RECOGNIZER_NAMES, get_recognizer  # noqa: E402


# ---------------------------------------------------------------- argument groups
def add_source_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("frame source")
    group.add_argument(
        "--source",
        default=None,
        help="video file or folder of images to use instead of a live camera",
    )
    group.add_argument("--camera-index", type=int, default=config.CAMERA_INDEX,
                       help="webcam index when no Pi camera is present (default: %(default)s)")
    group.add_argument("--width", type=int, default=config.FRAME_WIDTH,
                       help="capture width (default: %(default)s)")
    group.add_argument("--height", type=int, default=config.FRAME_HEIGHT,
                       help="capture height (default: %(default)s)")
    group.add_argument("--no-picamera", action="store_true",
                       help="ignore picamera2 even on the Raspberry Pi")


def add_display_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("display")
    group.add_argument("--no-display", action="store_true",
                       help="do not open a window (use this over SSH without X forwarding)")
    group.add_argument("--save-frames", default=None, metavar="DIR",
                       help="write annotated frames into DIR")
    group.add_argument("--save-every", type=int, default=1,
                       help="save one frame out of every N (default: %(default)s)")
    group.add_argument("--max-frames", type=int, default=0,
                       help="stop after N frames, 0 means run until q is pressed")


def add_detector_args(parser: argparse.ArgumentParser, default: str = "haar") -> None:
    group = parser.add_argument_group("detector")
    group.add_argument("--detector", choices=DETECTOR_NAMES, default=default,
                       help="face detector to use (default: %(default)s)")
    group.add_argument("--scale-factor", type=float, default=config.HAAR_SCALE_FACTOR,
                       help="Haar scaleFactor (default: %(default)s)")
    group.add_argument("--min-neighbors", type=int, default=config.HAAR_MIN_NEIGHBORS,
                       help="Haar minNeighbors (default: %(default)s)")
    group.add_argument("--min-size", type=int, default=config.HAAR_MIN_SIZE[0],
                       help="Haar minimum face side in pixels (default: %(default)s)")
    group.add_argument("--score-threshold", type=float, default=config.YUNET_SCORE_THRESHOLD,
                       help="YuNet score threshold (default: %(default)s)")


def add_recognizer_args(parser: argparse.ArgumentParser, default: str = "lbph") -> None:
    group = parser.add_argument_group("recognizer")
    group.add_argument("--recognizer", choices=RECOGNIZER_NAMES, default=default,
                       help="recognition method (default: %(default)s)")
    group.add_argument("--threshold", type=float, default=None,
                       help="decision threshold, overrides the value in config.py")
    group.add_argument("--model", default=None,
                       help="path to the trained model or gallery")


# ---------------------------------------------------------------- builders
def build_detector(args: argparse.Namespace, detector_name: str = None):
    """Create the detector named on the command line, passing only its own options."""
    name = detector_name or args.detector
    if name == "haar":
        return get_detector(
            "haar",
            scale_factor=getattr(args, "scale_factor", config.HAAR_SCALE_FACTOR),
            min_neighbors=getattr(args, "min_neighbors", config.HAAR_MIN_NEIGHBORS),
            min_size=(getattr(args, "min_size", config.HAAR_MIN_SIZE[0]),) * 2,
        )
    return get_detector(
        "yunet",
        score_threshold=getattr(args, "score_threshold", config.YUNET_SCORE_THRESHOLD),
        input_size=(getattr(args, "width", config.FRAME_WIDTH),
                    getattr(args, "height", config.FRAME_HEIGHT)),
    )


def build_recognizer(args: argparse.Namespace, recognizer_name: str = None):
    name = recognizer_name or args.recognizer
    kwargs = {}
    threshold = getattr(args, "threshold", None)
    if threshold is not None:
        kwargs["threshold"] = threshold
    return get_recognizer(name, **kwargs)


def build_camera(args: argparse.Namespace):
    from facetrack.camera import CameraSource

    return CameraSource(
        source=getattr(args, "source", None),
        width=getattr(args, "width", config.FRAME_WIDTH),
        height=getattr(args, "height", config.FRAME_HEIGHT),
        camera_index=getattr(args, "camera_index", config.CAMERA_INDEX),
        prefer_picamera=not getattr(args, "no_picamera", False),
    )


def build_display(args: argparse.Namespace, title: str):
    from facetrack.draw import Display

    return Display(
        title=title,
        enabled=not getattr(args, "no_display", False),
        save_dir=getattr(args, "save_frames", None),
        save_every=getattr(args, "save_every", 1),
    )


def banner(text: str) -> None:
    print("=" * 70)
    print(text)
    print("=" * 70)
