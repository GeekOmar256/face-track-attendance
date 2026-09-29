"""Stage C: train a recognizer from the captured dataset.

    python scripts/train.py --recognizer lbph
    python scripts/train.py --recognizer sface --detector yunet

Every dataset image is run through the detector, the largest face is taken as
the subject, and the recognizer builds its model or gallery from those faces.
Images where no face is found are reported, because a person with very few
usable images will drag the accuracy down and it is better to know before the
evaluation than after.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import (
    add_detector_args,
    add_recognizer_args,
    banner,
    build_detector,
    build_recognizer,
    config,
)
from facetrack.recognizers import TrainingSample, default_model_path
from facetrack.utils import iter_images, list_dataset, split_person_dir


def collect_samples(dataset_dir: Path, detector, verbose: bool = True):
    """Detect one face per dataset image and return the training samples."""
    people = list_dataset(dataset_dir)
    if not people:
        raise SystemExit(
            f"no people found in {dataset_dir}.\n"
            'Run:  python scripts/capture_dataset.py --id 210001 --name "First Student"'
        )

    samples = []
    report = []
    for person, paths in people.items():
        found = 0
        for _, image in iter_images(paths):
            detection = detector.detect_largest(image)
            if detection is None:
                continue
            samples.append(TrainingSample(label=person, image=image, detection=detection))
            found += 1
        report.append({"person": person, "images": len(paths), "faces_found": found})
        if verbose:
            student_id, name = split_person_dir(person)
            rate = 100 * found / len(paths) if paths else 0
            flag = "" if rate >= 80 else "   <-- low, check these images"
            print(f"  {name:<28} {student_id:<8} {found:>3}/{len(paths):<3} faces ({rate:5.1f}%){flag}")

    return samples, report


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
    recognizer = build_recognizer(args)

    banner(f"Face Track: training {recognizer.name} using {detector.name} detections")
    print(f"dataset   : {dataset_dir}")
    print(f"detector  : {detector.describe()}")
    print()

    samples, report = collect_samples(dataset_dir, detector)
    people = sorted({s.label for s in samples})

    print(f"\ntotal training faces: {len(samples)} across {len(people)} people")
    if len(people) < 2:
        raise SystemExit("at least two people are needed before training is meaningful.")
    if len(people) < 10:
        print(f"note: the advisor asked for about 10 people, this dataset has {len(people)}.")

    print("\ntraining...")
    recognizer.fit(samples)

    model_path = Path(args.model) if args.model else default_model_path(recognizer.name)
    recognizer.save(model_path)

    print(f"\n{recognizer.describe()}")
    print("\nnext:")
    print(f"  python scripts/recognize_live.py --recognizer {recognizer.name} "
          f"--detector {detector.name}")
    print(f"  python scripts/evaluate_recognition.py   # to calibrate the threshold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
