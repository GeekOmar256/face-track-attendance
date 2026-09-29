"""Overlay drawing, plus a display wrapper that survives a headless Raspberry Pi.

The Pi is operated over SSH, so cv2.imshow may have no window system to talk to.
``Display`` tries once, and if the window cannot be created it switches to
saving annotated frames instead of letting the script die mid demo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

GREEN = (0, 200, 0)
RED = (0, 0, 220)
AMBER = (0, 180, 240)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_box(frame: np.ndarray, detection, color=GREEN, thickness: int = 2) -> None:
    cv2.rectangle(
        frame,
        (detection.x, detection.y),
        (detection.x + detection.w, detection.y + detection.h),
        color,
        thickness,
    )


def draw_label(frame: np.ndarray, detection, text: str, color=GREEN) -> None:
    """Label under the box, on a filled strip so it stays readable on any background."""
    (tw, th), baseline = cv2.getTextSize(text, FONT, 0.55, 1)
    x = detection.x
    y = detection.y + detection.h + th + baseline + 4
    height, width = frame.shape[:2]
    if y > height - 2:                      # not enough room below, put it above
        y = max(th + 4, detection.y - 6)
    x = min(x, max(0, width - tw - 6))
    cv2.rectangle(frame, (x, y - th - baseline - 2), (x + tw + 6, y + 2), color, -1)
    cv2.putText(frame, text, (x + 3, y - baseline + 1), FONT, 0.55, WHITE, 1, cv2.LINE_AA)


def draw_landmarks(frame: np.ndarray, detection, color=AMBER) -> None:
    """Mark the five landmark points, sized relative to the face.

    A fixed two pixel dot disappears on a large face and on a phone screen, so
    the radius follows the box width.
    """
    if detection.landmarks is None:
        return
    radius = max(2, int(round(detection.w * 0.012)))
    for (px, py) in detection.landmarks.astype(int):
        cv2.circle(frame, (int(px), int(py)), radius, color, -1)
        cv2.circle(frame, (int(px), int(py)), radius, BLACK, 1)


def draw_detections(
    frame: np.ndarray,
    detections: Sequence,
    labels: Optional[Sequence[str]] = None,
    colors: Optional[Sequence] = None,
    show_landmarks: bool = True,
) -> np.ndarray:
    for i, detection in enumerate(detections):
        color = colors[i] if colors else GREEN
        draw_box(frame, detection, color)
        if show_landmarks:
            draw_landmarks(frame, detection)
        if labels:
            draw_label(frame, detection, labels[i], color)
    return frame


def draw_banner(frame: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    """Small dark strip at the top left carrying FPS, detector name and counters."""
    if not lines:
        return frame
    pad = 6
    heights = []
    widest = 0
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, FONT, 0.5, 1)
        heights.append(th)
        widest = max(widest, tw)
    box_h = sum(heights) + pad * (len(lines) + 1)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (widest + pad * 2, box_h), BLACK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    y = pad
    for line, th in zip(lines, heights):
        y += th
        cv2.putText(frame, line, (pad, y), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
        y += pad
    return frame


class Display:
    """Show frames in a window when possible, save them to disk when not."""

    def __init__(
        self,
        title: str = "Face Track",
        enabled: bool = True,
        save_dir: Optional[str] = None,
        save_every: int = 1,
    ) -> None:
        self.title = title
        self.enabled = enabled
        self.save_dir = Path(save_dir) if save_dir else None
        self.save_every = max(1, save_every)
        self.frame_index = 0
        self.saved = 0
        self._window_failed = False
        self._warned = False
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def show(self, frame: np.ndarray) -> int:
        """Display and/or save one frame. Returns the pressed key, or -1."""
        self.frame_index += 1

        if self.save_dir and self.frame_index % self.save_every == 0:
            path = self.save_dir / f"frame_{self.frame_index:05d}.jpg"
            cv2.imwrite(str(path), frame)
            self.saved += 1

        if not self.enabled or self._window_failed:
            return -1

        try:
            cv2.imshow(self.title, frame)
            return cv2.waitKey(1) & 0xFF
        except cv2.error as exc:
            self._window_failed = True
            if not self._warned:
                print(
                    "\nno display available, continuing without a window "
                    f"({exc.err.strip() if hasattr(exc, 'err') else exc})"
                )
                if not self.save_dir:
                    print("tip: pass --save-frames DIR to keep the annotated frames instead.")
                self._warned = True
            return -1

    def close(self) -> None:
        if self.enabled and not self._window_failed:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        if self.save_dir and self.saved:
            print(f"saved {self.saved} annotated frames to {self.save_dir}")

    def __enter__(self) -> "Display":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
