"""One camera layer for every script.

On the Raspberry Pi this uses picamera2 and the Camera Module v2. On a laptop
it falls back to cv2.VideoCapture and the built in webcam. It also accepts a
video file or a folder of images as the source, which is what makes the
benchmark runs repeatable: both detectors see exactly the same frames.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from . import config


def picamera_available() -> bool:
    try:
        import picamera2  # noqa: F401
        return True
    except Exception:
        return False


class CameraSource:
    """Frame source with a uniform ``read()`` regardless of what is behind it."""

    def __init__(
        self,
        source: Optional[str] = None,
        width: int = config.FRAME_WIDTH,
        height: int = config.FRAME_HEIGHT,
        camera_index: int = config.CAMERA_INDEX,
        prefer_picamera: bool = True,
    ) -> None:
        self.width = width
        self.height = height
        self.kind = "unknown"
        self._picam = None
        self._capture = None
        self._images: List[Path] = []
        self._image_pos = 0

        if source:
            self._open_path(Path(source))
        elif prefer_picamera and picamera_available():
            self._open_picamera()
        else:
            self._open_webcam(camera_index)

    # ------------------------------------------------------------ openers
    def _open_path(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"source not found: {path}")
        if path.is_dir():
            self._images = sorted(
                f for f in path.rglob("*")
                if f.suffix.lower() in config.IMAGE_EXTENSIONS
            )
            if not self._images:
                raise ValueError(f"no images found under {path}")
            self.kind = "images"
        elif path.suffix.lower() in config.IMAGE_EXTENSIONS:
            # A single photo. Handled as a one frame sequence rather than through
            # VideoCapture, so the GUI can tell a finished photo from a dead camera.
            self._images = [path]
            self.kind = "image"
        else:
            self._capture = cv2.VideoCapture(str(path))
            if not self._capture.isOpened():
                raise RuntimeError(f"could not open video file: {path}")
            self.kind = "video"

    def _open_picamera(self) -> None:
        from picamera2 import Picamera2

        self._picam = Picamera2()
        cfg = self._picam.create_preview_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"}
        )
        self._picam.configure(cfg)
        self._picam.start()
        self.kind = "picamera2"

    def _open_webcam(self, index: int) -> None:
        self._capture = cv2.VideoCapture(index, cv2.CAP_DSHOW if _is_windows() else 0)
        if not self._capture.isOpened():
            # CAP_DSHOW is a Windows only speed up; retry without it before giving up.
            self._capture = cv2.VideoCapture(index)
        if not self._capture.isOpened():
            raise RuntimeError(
                f"could not open camera index {index}. "
                "On the Raspberry Pi install picamera2, or pass --source with a video file."
            )
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.kind = "webcam"

    # ------------------------------------------------------------ reading
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._picam is not None:
            frame = self._picam.capture_array()
            # picamera2 hands back RGB; the rest of the pipeline works in BGR.
            return True, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if self._images:
            if self._image_pos >= len(self._images):
                return False, None
            path = self._images[self._image_pos]
            self._image_pos += 1
            frame = cv2.imread(str(path))
            if frame is None:
                return self.read()
            return True, frame

        ok, frame = self._capture.read()
        return ok, frame

    def __iter__(self):
        while True:
            ok, frame = self.read()
            if not ok or frame is None:
                break
            yield frame

    # ------------------------------------------------------------ teardown
    def release(self) -> None:
        if self._picam is not None:
            try:
                self._picam.stop()
                self._picam.close()
            except Exception:
                pass
            self._picam = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "CameraSource":
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    def __repr__(self) -> str:
        return f"CameraSource(kind={self.kind}, size={self.width}x{self.height})"


def _is_windows() -> bool:
    import sys
    return sys.platform.startswith("win")
