"""SFace recognizer: the lightweight CNN embedding model.

SFace turns an aligned face into a 128 value embedding, and two faces are
compared by the cosine similarity of their embeddings. This is the
"lightweight CNN" the FYP1 plan committed to, and it runs through OpenCV's own
DNN module, so the Raspberry Pi needs no TensorFlow. Enrolling a new student
means appending embeddings to the gallery, with no retraining at all.

Cosine similarity, so a HIGHER score is a better match.

One honest limitation, worth reporting rather than hiding: ``alignCrop`` needs
the five facial landmarks, and only YuNet produces them. Behind Haar Cascade
this class falls back to a plain resized crop, and the pairing is expected to
score lower than YuNet with SFace. That gap is a result for the comparison
table, not a bug.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .. import config
from ..detectors.base import Detection
from .base import FaceRecognizer, RecognitionResult, TrainingSample


class SFaceRecognizer(FaceRecognizer):
    name = "sface"
    higher_is_better = True

    def __init__(
        self,
        threshold: float = config.SFACE_COSINE_THRESHOLD,
        model_path: Optional[Path] = None,
        input_size: Tuple[int, int] = config.SFACE_INPUT_SIZE,
        margin: float = config.CAPTURE_MARGIN,
    ) -> None:
        super().__init__(threshold)
        if not hasattr(cv2, "FaceRecognizerSF"):
            raise RuntimeError(
                "this OpenCV build has no FaceRecognizerSF. Install opencv-contrib-python 4.7 or newer."
            )

        self.model_path = Path(model_path or config.SFACE_MODEL)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"SFace model not found at {self.model_path}.\n"
                "Run:  python scripts/download_models.py"
            )

        self.input_size = tuple(input_size)
        self.margin = margin
        self.model = cv2.FaceRecognizerSF.create(str(self.model_path), "")

        self._features = np.empty((0, 128), dtype=np.float32)
        self._feature_labels: List[str] = []
        self._aligned_count = 0
        self._fallback_count = 0

    # ------------------------------------------------------------ features
    def _align(self, frame: np.ndarray, detection: Detection) -> Optional[np.ndarray]:
        """Produce the 112x112 input SFace expects."""
        if detection.raw is not None:
            # The proper path: landmark based alignment using YuNet's output row.
            try:
                aligned = self.model.alignCrop(frame, detection.raw.reshape(1, -1))
                if aligned is not None and aligned.size:
                    self._aligned_count += 1
                    return aligned
            except cv2.error:
                pass

        # Fallback for Haar, which reports no landmarks.
        crop = detection.crop(frame, margin=self.margin)
        if crop.size == 0:
            return None
        self._fallback_count += 1
        return cv2.resize(crop, self.input_size, interpolation=cv2.INTER_AREA)

    def embed(self, frame: np.ndarray, detection: Detection) -> Optional[np.ndarray]:
        """Return the 128 value embedding for one detected face."""
        aligned = self._align(frame, detection)
        if aligned is None:
            return None
        feature = self.model.feature(aligned)
        return np.asarray(feature, dtype=np.float32).reshape(-1)

    @staticmethod
    def _normalise(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    # ------------------------------------------------------------ training
    def fit(self, samples: Sequence[TrainingSample]) -> None:
        features: List[np.ndarray] = []
        labels: List[str] = []
        skipped = 0

        for sample in samples:
            feature = self.embed(sample.image, sample.detection)
            if feature is None:
                skipped += 1
                continue
            features.append(feature)
            labels.append(sample.label)

        if not features:
            raise ValueError("no usable embeddings. Check the dataset and the detector.")

        self._features = np.vstack(features).astype(np.float32)
        self._feature_labels = labels
        self._labels = sorted(set(labels))

        if skipped:
            print(f"  note: {skipped} sample(s) skipped, no embedding could be computed")
        if self._fallback_count:
            print(
                f"  note: {self._fallback_count} of {len(features)} crops used the unaligned "
                "fallback, which happens when the detector reports no landmarks (Haar)."
            )

    # ------------------------------------------------------------ persistence
    def save(self, path: Optional[Path] = None) -> None:
        if self._features.size == 0:
            raise RuntimeError("nothing to save, the gallery is empty")
        gallery_path = Path(path or config.SFACE_GALLERY)
        gallery_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            gallery_path,
            features=self._features,
            labels=np.array(self._feature_labels, dtype=object),
            threshold=np.array([self.threshold], dtype=np.float32),
            margin=np.array([self.margin], dtype=np.float32),
        )
        print(f"saved SFace gallery to {gallery_path} "
              f"({self._features.shape[0]} embeddings, {len(self._labels)} people)")

    def load(self, path: Optional[Path] = None) -> None:
        gallery_path = Path(path or config.SFACE_GALLERY)
        if not gallery_path.is_file():
            raise FileNotFoundError(
                f"no SFace gallery at {gallery_path}. "
                "Run:  python scripts/train.py --recognizer sface"
            )
        data = np.load(gallery_path, allow_pickle=True)
        self._features = data["features"].astype(np.float32)
        self._feature_labels = [str(v) for v in data["labels"]]
        self._labels = sorted(set(self._feature_labels))
        if "margin" in data:
            self.margin = float(data["margin"][0])

    # ------------------------------------------------------------ inference
    def predict(self, frame: np.ndarray, detection: Detection) -> RecognitionResult:
        if self._features.size == 0:
            raise RuntimeError("recognizer not trained or loaded")

        feature = self.embed(frame, detection)
        if feature is None:
            return self._unknown(-1.0)

        # cv2's FR_COSINE is the cosine of the angle between the two embeddings.
        # Computing it with numpy against the whole gallery at once is the same
        # measure and keeps the live loop fast enough for the Raspberry Pi.
        gallery = self._normalise(self._features)
        probe = feature / (np.linalg.norm(feature) or 1.0)
        similarities = gallery @ probe

        best_index = int(np.argmax(similarities))
        best_score = float(similarities[best_index])
        best_label = self._feature_labels[best_index]

        return RecognitionResult(
            label=best_label, score=best_score, is_known=self.passes(best_score)
        )

    def describe(self) -> str:
        return (
            f"SFace ({self.model_path.name}, {self._features.shape[0]} embeddings, "
            f"{len(self._labels)} enrolled, cosine threshold={self.threshold:g})"
        )
