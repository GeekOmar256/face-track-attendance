"""LBPH recognizer: the simple baseline.

Local Binary Pattern Histograms come with opencv-contrib, need no model
download, and train on ten people in a couple of seconds. The FYP1 literature
review already notes that this family is sensitive to lighting, which is the
point of having the SFace embedding recognizer to compare it against.

LBPH returns a distance, so a LOWER score is a better match.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .. import config
from ..detectors.base import Detection
from .base import FaceRecognizer, RecognitionResult, TrainingSample


class LBPHRecognizer(FaceRecognizer):
    name = "lbph"
    higher_is_better = False

    def __init__(
        self,
        threshold: float = config.LBPH_THRESHOLD,
        face_size: Tuple[int, int] = config.LBPH_FACE_SIZE,
        radius: int = config.LBPH_RADIUS,
        neighbors: int = config.LBPH_NEIGHBORS,
        grid_x: int = config.LBPH_GRID_X,
        grid_y: int = config.LBPH_GRID_Y,
        margin: float = config.CAPTURE_MARGIN,
    ) -> None:
        super().__init__(threshold)
        if not hasattr(cv2, "face"):
            raise RuntimeError(
                "this OpenCV build has no cv2.face module. Install opencv-contrib-python."
            )
        self.face_size = tuple(face_size)
        self.margin = margin
        self.model = cv2.face.LBPHFaceRecognizer_create(
            radius=radius, neighbors=neighbors, grid_x=grid_x, grid_y=grid_y
        )
        self._trained = False

    # ------------------------------------------------------------ features
    def _prepare(self, frame: np.ndarray, detection: Detection) -> Optional[np.ndarray]:
        """Grey, equalised, fixed size crop. Returns None if the crop is empty."""
        crop = detection.crop(frame, margin=self.margin)
        if crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        gray = cv2.resize(gray, self.face_size, interpolation=cv2.INTER_AREA)
        return cv2.equalizeHist(gray)

    # ------------------------------------------------------------ training
    def fit(self, samples: Sequence[TrainingSample]) -> None:
        images: List[np.ndarray] = []
        label_ids: List[int] = []
        self._labels = sorted({s.label for s in samples})
        index = {label: i for i, label in enumerate(self._labels)}

        skipped = 0
        for sample in samples:
            prepared = self._prepare(sample.image, sample.detection)
            if prepared is None:
                skipped += 1
                continue
            images.append(prepared)
            label_ids.append(index[sample.label])

        if not images:
            raise ValueError("no usable training crops. Check the dataset and the detector.")
        if len(self._labels) < 2:
            raise ValueError(
                "LBPH needs at least two enrolled people to be meaningful, "
                f"found {len(self._labels)}."
            )

        self.model.train(images, np.array(label_ids, dtype=np.int32))
        self._trained = True
        if skipped:
            print(f"  note: {skipped} sample(s) skipped, the crop came out empty")

    # ------------------------------------------------------------ persistence
    def save(self, path: Optional[Path] = None) -> None:
        if not self._trained:
            raise RuntimeError("nothing to save, the recognizer has not been trained")
        model_path = Path(path or config.LBPH_MODEL)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.write(str(model_path))

        labels_path = model_path.with_name(model_path.stem + "_labels.json")
        labels_path.write_text(
            json.dumps(
                {
                    "labels": self._labels,
                    "face_size": list(self.face_size),
                    "margin": self.margin,
                    "threshold": self.threshold,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"saved LBPH model to {model_path}")
        print(f"saved label map to  {labels_path}")

    def load(self, path: Optional[Path] = None) -> None:
        model_path = Path(path or config.LBPH_MODEL)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"no LBPH model at {model_path}. Run:  python scripts/train.py --recognizer lbph"
            )
        self.model.read(str(model_path))

        labels_path = model_path.with_name(model_path.stem + "_labels.json")
        if not labels_path.is_file():
            raise FileNotFoundError(f"label map missing at {labels_path}")
        meta = json.loads(labels_path.read_text(encoding="utf-8"))
        self._labels = list(meta["labels"])
        self.face_size = tuple(meta.get("face_size", self.face_size))
        self.margin = float(meta.get("margin", self.margin))
        self._trained = True

    # ------------------------------------------------------------ inference
    def predict(self, frame: np.ndarray, detection: Detection) -> RecognitionResult:
        if not self._trained:
            raise RuntimeError("recognizer not trained or loaded")
        prepared = self._prepare(frame, detection)
        if prepared is None:
            return self._unknown(float("inf"))

        label_id, distance = self.model.predict(prepared)
        if label_id < 0 or label_id >= len(self._labels):
            return self._unknown(float(distance))

        label = self._labels[label_id]
        return RecognitionResult(
            label=label, score=float(distance), is_known=self.passes(distance)
        )
