"""Stage D: Haar Cascade against the YuNet DNN detector, on identical images.

    python scripts/benchmark_detectors.py

This is the comparison the advisor described as extra work that strengthens
both the project and the paper. Both detectors see exactly the same frames.

Ground truth without hand labelling: every dataset image was captured with one
person in front of the camera and saved only when exactly one face was visible,
so each image contains exactly one face. That gives two honest measures:

  detection rate   share of images where at least one face was found
  extra detections boxes beyond the first, which are false positives

Run it on the Raspberry Pi as well as the laptop. The timing column is what
answers NFReq-3, and Pi numbers are the ones that belong in the report.
"""

from __future__ import annotations

import argparse
import platform
from pathlib import Path

from _bootstrap import banner, build_detector, config
from facetrack.detectors import DETECTOR_NAMES
from facetrack.utils import Stopwatch, iter_images, list_dataset, save_table


def benchmark(detector, images, warmup: int = 3):
    """Run one detector over every image and collect timings and counts."""
    # A few untimed frames first, so one-off allocation does not land in the mean.
    for _, image in iter_images(images[:warmup]):
        detector.detect(image)

    timings = []
    found = 0
    extra = 0
    missed = []

    for path, image in iter_images(images):
        with Stopwatch() as watch:
            detections = detector.detect(image)
        timings.append(watch.ms)
        if detections:
            found += 1
            extra += len(detections) - 1
        else:
            missed.append(path)

    total = len(timings)
    mean_ms = sum(timings) / total if total else 0.0
    sorted_ms = sorted(timings)
    median_ms = sorted_ms[total // 2] if total else 0.0
    return {
        "images": total,
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "max_ms": max(timings) if timings else 0.0,
        "fps": 1000 / mean_ms if mean_ms else 0.0,
        "found": found,
        "detection_rate": 100 * found / total if total else 0.0,
        "extra": extra,
        "missed": missed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=None, help="dataset root (default: data/dataset)")
    parser.add_argument("--detectors", nargs="+", choices=DETECTOR_NAMES,
                        default=list(DETECTOR_NAMES),
                        help="which detectors to compare (default: all)")
    parser.add_argument("--limit", type=int, default=0,
                        help="use only the first N images per person, 0 means all")
    parser.add_argument("--output", default=None,
                        help="output file stem (default: data/output/detector_benchmark)")
    parser.add_argument("--list-missed", action="store_true",
                        help="print the images where a detector found no face")
    parser.add_argument("--width", type=int, default=config.FRAME_WIDTH, help=argparse.SUPPRESS)
    parser.add_argument("--height", type=int, default=config.FRAME_HEIGHT, help=argparse.SUPPRESS)
    parser.add_argument("--scale-factor", type=float, default=config.HAAR_SCALE_FACTOR)
    parser.add_argument("--min-neighbors", type=int, default=config.HAAR_MIN_NEIGHBORS)
    parser.add_argument("--min-size", type=int, default=config.HAAR_MIN_SIZE[0])
    parser.add_argument("--score-threshold", type=float, default=config.YUNET_SCORE_THRESHOLD)
    args = parser.parse_args()

    config.ensure_dirs()
    dataset_dir = Path(args.dataset) if args.dataset else config.DATASET_DIR
    people = list_dataset(dataset_dir)
    if not people:
        raise SystemExit(f"no images found under {dataset_dir}. Capture a dataset first.")

    images = []
    for paths in people.values():
        images.extend(paths[: args.limit] if args.limit else paths)

    banner("Face Track: detector comparison")
    print(f"machine : {platform.machine()} on {platform.system()} {platform.release()}")
    print(f"dataset : {dataset_dir}")
    print(f"images  : {len(images)} from {len(people)} people")
    print("note    : each image holds exactly one face, so extra boxes are false positives\n")

    rows = []
    for name in args.detectors:
        try:
            detector = build_detector(args, detector_name=name)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"skipping {name}: {exc}\n")
            continue

        print(f"running {detector.describe()}")
        stats = benchmark(detector, images)
        print(f"  {stats['mean_ms']:.1f} ms/image, detection rate {stats['detection_rate']:.1f}%, "
              f"{stats['extra']} extra box(es)\n")

        if args.list_missed and stats["missed"]:
            print(f"  images with no face found by {name}:")
            for path in stats["missed"]:
                print(f"    {path}")
            print()

        rows.append({
            "Detector": "Haar Cascade" if name == "haar" else "YuNet (DNN)",
            "Images": stats["images"],
            "Detection rate (%)": f"{stats['detection_rate']:.1f}",
            "Faces found": stats["found"],
            "Extra detections": stats["extra"],
            "Mean time (ms)": f"{stats['mean_ms']:.1f}",
            "Median time (ms)": f"{stats['median_ms']:.1f}",
            "Max time (ms)": f"{stats['max_ms']:.1f}",
            "FPS": f"{stats['fps']:.1f}",
            "NFReq-3 (<=1000 ms)": "PASS" if stats["mean_ms"] <= 1000 else "FAIL",
        })

    if not rows:
        print("no detector could be run.")
        return 1

    stem = Path(args.output) if args.output else config.OUTPUT_DIR / "detector_benchmark"
    save_table(stem, rows)
    print(f"\nplatform recorded: {platform.machine()} / {platform.system()} {platform.release()}")
    print("run this on the Raspberry Pi too, those are the timings the report needs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
