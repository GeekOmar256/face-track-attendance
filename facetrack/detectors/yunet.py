"""YuNet face detector: the deep neural network the advisor suggested comparing.

YuNet is a small convolutional detector that runs inside OpenCV's own DNN
module, so the Raspberry Pi needs no TensorFlow and no PyTorch. The model file
is about 230 KB. Run scripts/download_models.py once to fetch it.

Unlike Haar it returns a confidence score and five facial landmarks, and those
landmarks are what SFace needs to align a face before computing its embedding.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .. import config
from .base import Detection, FaceDetector


class YuNetDetector(FaceDetector):
    name = "yunet"

    def __init__(
        self,
        model_path: Optional[Path] = None,
        score_threshold: float = config.YUNET_SCORE_THRESHOLD,
        nms_threshold: float = config.YUNET_NMS_THRESHOLD,
        top_k: int = config.YUNET_TOP_K,
        input_size: Tuple[int, int] = (config.FRAME_WIDTH, config.FRAME_HEIGHT),
    ) -> None:
        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError(
                "this OpenCV build has no FaceDetectorYN. Install opencv-contrib-python 4.7 or newer."
            )

        self.model_path = Path(model_path or config.YUNET_MODEL)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"YuNet model not found at {self.model_path}.\n"
                "Run:  python scripts/download_models.py"
            )

        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self._input_size = tuple(input_size)

        self.detector = cv2.FaceDetectorYN.create(
            str(self.model_path),
            "",                      # ONNX needs no separate config file
            self._input_size,
            score_threshold,
            nms_threshold,
            top_k,
        )

    def _sync_input_size(self, frame: np.ndarray) -> None:
        """YuNet must be told the frame size, and told again whenever it changes."""
        height, width = frame.shape[:2]
        if (width, height) != self._input_size:
            self.detector.setInputSize((width, height))
            self._input_size = (width, height)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        self._sync_input_size(frame)
        _, faces = self.detector.detect(frame)
        if faces is None:
            return []

        detections: List[Detection] = []
        for row in faces:
            x, y, w, h = row[0:4]
            landmarks = row[4:14].reshape(5, 2)
            detections.append(
                Detection(
                    int(round(x)),
                    int(round(y)),
                    int(round(w)),
                    int(round(h)),
                    score=float(row[14]),
                    landmarks=landmarks.astype(np.float32),
                    raw=np.asarray(row, dtype=np.float32),
                )
            )
        return detections

    def describe(self) -> str:
        return (
            f"YuNet DNN ({self.model_path.name}, scoreThreshold={self.score_threshold}, "
            f"nmsThreshold={self.nms_threshold})"
        )
