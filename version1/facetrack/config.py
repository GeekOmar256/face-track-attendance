"""Central configuration for the Face Track pipeline.

Every tunable number the report may need to discuss lives here rather than
being buried in the code, so a parameter sweep means editing one file or
passing one flag.
"""

from pathlib import Path

# ---------------------------------------------------------------- paths
CODE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CODE_DIR / "data"
DATASET_DIR = DATA_DIR / "dataset"
MODELS_DIR = DATA_DIR / "models"
OUTPUT_DIR = DATA_DIR / "output"

YUNET_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

LBPH_MODEL = MODELS_DIR / "lbph_model.yml"
LBPH_LABELS = MODELS_DIR / "lbph_labels.json"
SFACE_GALLERY = MODELS_DIR / "sface_gallery.npz"

# ---------------------------------------------------------------- camera
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_INDEX = 0

# ---------------------------------------------------------------- Haar cascade (baseline detector)
HAAR_CASCADE_NAME = "haarcascade_frontalface_default.xml"
HAAR_SCALE_FACTOR = 1.1
HAAR_MIN_NEIGHBORS = 5
HAAR_MIN_SIZE = (60, 60)

# ---------------------------------------------------------------- YuNet (DNN detector, the comparison)
YUNET_SCORE_THRESHOLD = 0.9
YUNET_NMS_THRESHOLD = 0.3
YUNET_TOP_K = 5000

# ---------------------------------------------------------------- LBPH recognizer (baseline)
LBPH_FACE_SIZE = (100, 100)
LBPH_RADIUS = 1
LBPH_NEIGHBORS = 8
LBPH_GRID_X = 8
LBPH_GRID_Y = 8
# LBPH returns a distance, so lower is better. This default is a starting point
# only: run scripts/evaluate_recognition.py and use the calibrated value.
LBPH_THRESHOLD = 70.0

# ---------------------------------------------------------------- SFace recognizer (embeddings)
SFACE_INPUT_SIZE = (112, 112)
# Cosine similarity, so higher is better. 0.363 is the threshold OpenCV documents
# for this model; calibrate it on the team's own images before quoting it.
SFACE_COSINE_THRESHOLD = 0.363

# ---------------------------------------------------------------- dataset capture
CAPTURE_DEFAULT_COUNT = 30
CAPTURE_BLUR_THRESHOLD = 60.0   # variance of the Laplacian; below this the crop is too soft
CAPTURE_MIN_FRAME_GAP = 5       # frames to skip between saves, so the images are not near duplicates
CAPTURE_MARGIN = 0.20           # fraction of the box added on each side when saving a crop

# ---------------------------------------------------------------- misc
UNKNOWN_LABEL = "Unknown"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def ensure_dirs() -> None:
    """Create the data folders if they are not there yet."""
    for path in (DATA_DIR, DATASET_DIR, MODELS_DIR, OUTPUT_DIR):
        path.mkdir(parents=True, exist_ok=True)
