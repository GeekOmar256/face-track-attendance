"""Face Track: embedded smart attendance system using face recognition.

FYP2, Team 10, Kuwait College of Science and Technology.

The package is deliberately split so that the detector and the recognizer are
each chosen by name at runtime. That is what makes the baseline against
advanced comparison the advisor asked for a command line flag instead of a
second version of the pipeline.
"""

__version__ = "0.1.0"

from . import config  # noqa: F401
from .camera import CameraSource  # noqa: F401
from .detectors import DETECTOR_NAMES, Detection, get_detector  # noqa: F401
from .recognizers import (  # noqa: F401
    RECOGNIZER_NAMES,
    RecognitionResult,
    TrainingSample,
    get_recognizer,
)

__all__ = [
    "config",
    "CameraSource",
    "Detection",
    "get_detector",
    "DETECTOR_NAMES",
    "RecognitionResult",
    "TrainingSample",
    "get_recognizer",
    "RECOGNIZER_NAMES",
]
