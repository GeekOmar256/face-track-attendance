"""Fetch the two ONNX models from the OpenCV model zoo.

    python scripts/download_models.py

Only needed for the YuNet detector and the SFace recognizer. The Haar Cascade
baseline and the LBPH recognizer need nothing downloaded, so Stage A and the
LBPH pipeline still work with no network at all.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request

from _bootstrap import banner, config

ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"

MODELS = [
    {
        "name": "YuNet face detector",
        "url": f"{ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "path": config.YUNET_MODEL,
        "min_bytes": 100_000,          # the real file is around 230 KB
    },
    {
        "name": "SFace face recognizer",
        "url": f"{ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "path": config.SFACE_MODEL,
        "min_bytes": 30_000_000,       # the real file is around 37 MB
    },
]


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    done = min(block_num * block_size, total_size)
    percent = done * 100 // total_size
    bar = "#" * (percent // 4)
    sys.stdout.write(f"\r    [{bar:<25}] {percent:3d}%  {done / 1e6:5.1f} / {total_size / 1e6:.1f} MB")
    sys.stdout.flush()


def download(model: dict, force: bool = False) -> bool:
    path = model["path"]
    if path.is_file() and path.stat().st_size >= model["min_bytes"] and not force:
        print(f"  already present: {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
        return True

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    print(f"  downloading {model['name']}")
    print(f"    from {model['url']}")
    try:
        urllib.request.urlretrieve(model["url"], temp, _progress)
        sys.stdout.write("\n")
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        sys.stdout.write("\n")
        temp.unlink(missing_ok=True)
        print(f"    FAILED: {exc}")
        print("    check the network connection, or copy the file across manually.")
        return False

    size = temp.stat().st_size
    if size < model["min_bytes"]:
        temp.unlink(missing_ok=True)
        print(f"    FAILED: downloaded file is only {size} bytes, it looks truncated.")
        return False

    temp.replace(path)
    print(f"    saved to {path} ({size / 1e6:.1f} MB)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="download again even if present")
    args = parser.parse_args()

    config.ensure_dirs()
    banner("Face Track: downloading detection and recognition models")

    results = [download(model, args.force) for model in MODELS]

    print()
    if all(results):
        print("all models ready.")
        return 0
    print("some models are missing. Haar Cascade and LBPH still work without them.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
