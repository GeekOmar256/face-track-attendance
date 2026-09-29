"""Stage C: train a recognizer from the captured dataset.

    python scripts/train.py --recognizer lbph
    python scripts/train.py --recognizer sface --detector yunet

Every dataset image is run through the detector, the largest face is taken as
the subject, and the recognizer builds its model or gallery from those faces.
Images where no face is found are reported, because a person with very few
usable images will drag the accuracy down and it is better to know before the
evaluation than after.

Version 2 note: the work happens in facetrack/training.py, so this script and
the Train button in the browser train identically.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import (
    add_detector_args,
    add_recognizer_args,
    banner,
    build_detector,
    config,
)
from facetrack.recognizers import default_model_path
from facetrack.training import train_recognizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_recognizer_args(parser, default="lbph")
    add_detector_args(parser, default="haar")
    parser.add_argument("--dataset", default=None, help="dataset root (default: data/dataset)")
    parser.add_argument("--width", type=int, default=config.FRAME_WIDTH, help=argparse.SUPPRESS)
    parser.add_argument("--height", type=int, default=config.FRAME_HEIGHT, help=argparse.SUPPRESS)
    args = parser.parse_args()

    config.ensure_dirs()
    dataset_dir = Path(args.dataset) if args.dataset else config.DATASET_DIR
    detector = build_detector(args)
    model_path = Path(args.model) if args.model else default_model_path(args.recognizer)

    banner(f"Face Track: training {args.recognizer} using {detector.name} detections")
    print(f"dataset   : {dataset_dir}")
    print(f"detector  : {detector.describe()}")
    print()

    def show(entry: dict) -> None:
        flag = "" if entry["rate"] >= 80 else "   <-- low, check these images"
        print(f"  {entry['name']:<28} {entry['id']:<8} "
              f"{entry['faces_found']:>3}/{entry['images']:<3} faces "
              f"({entry['rate']:5.1f}%){flag}")

    try:
        result = train_recognizer(args.recognizer, detector, dataset_dir, model_path, show)
    except ValueError as exc:
        raise SystemExit(f"\n{exc}\n")

    print(f"\ntotal training faces: {result['faces']} across {result['people']} people")
    if result["people"] < 10:
        print(f"note: the advisor asked for about 10 people, this dataset has "
              f"{result['people']}.")

    print("\nnext:")
    print(f"  python scripts/recognize_live.py --recognizer {args.recognizer} "
          f"--detector {detector.name}")
    print("  python scripts/evaluate_recognition.py   # to calibrate the threshold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
