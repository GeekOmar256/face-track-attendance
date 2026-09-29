"""Publishing the most recent annotated frame to whatever is watching.

Both the recognition pipeline and an enrollment capture produce a live preview,
and the browser streams whichever one is running. Keeping that behaviour in one
small base class means the stream endpoint does not care which it is talking to.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import config


class FrameSource:
    """Holds the latest frame under a lock, with a counter so readers can tell
    a new frame from a repeat."""

    def __init__(self) -> None:
        self._frame_lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._frame_id = 0

    def publish(self, frame: np.ndarray) -> None:
        with self._frame_lock:
            self._frame = frame
            self._frame_id += 1

    def clear(self) -> None:
        with self._frame_lock:
            self._frame = None

    def latest_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            return None if self._frame is None else self._frame.copy()

    @property
    def frame_id(self) -> int:
        with self._frame_lock:
            return self._frame_id

    def latest_jpeg(self, quality: int = 80) -> Optional[bytes]:
        frame = self.latest_frame()
        if frame is None:
            return None
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buffer.tobytes() if ok else None

    def save_snapshot(self) -> Optional[Path]:
        frame = self.latest_frame()
        if frame is None:
            return None
        config.ensure_dirs()
        folder = config.OUTPUT_DIR / "snapshots"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(str(path), frame)
        return path
