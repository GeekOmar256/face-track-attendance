import argparse
import bz2
import hashlib
import shutil
import urllib.request

from vision import MODELS


FILES = {
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        200000),
    "face_recognition_sface_2021dec.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        30000000),
    "shape_predictor_5_face_landmarks.dat": (
        "https://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2", 5000000),
    "dlib_face_recognition_resnet_model_v1.dat": (
        "https://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2", 20000000),
}


def download(name, url, minimum):
    destination = MODELS / name
    if destination.exists() and destination.stat().st_size >= minimum:
        print(f"Already present: {name}")
        return
    compressed = url.endswith(".bz2")
    partial = MODELS / (name + (".bz2" if compressed else "") + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
    with urllib.request.urlopen(request, timeout=120) as response:
        mode = "ab" if offset and response.status == 206 else "wb"
        with partial.open(mode) as output:
            shutil.copyfileobj(response, output, length=1024*1024)
    extracted = MODELS / (name + ".ready")
    if compressed:
        with bz2.open(partial, "rb") as source, extracted.open("wb") as target:
            shutil.copyfileobj(source, target)
    else:
        shutil.copyfile(partial, extracted)
    if extracted.stat().st_size < minimum:
        raise RuntimeError(f"Download too small: {name}; possibly a Git LFS pointer or error page")
    extracted.replace(destination)
    partial.unlink()
    print(f"Downloaded {name}; SHA256 {hashlib.sha256(destination.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dlib", action="store_true", help="Also download dlib's predictor and embedding model")
    args = parser.parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)
    for name, (url, minimum) in FILES.items():
        if name.endswith(".dat") and not args.dlib:
            continue
        download(name, url, minimum)
