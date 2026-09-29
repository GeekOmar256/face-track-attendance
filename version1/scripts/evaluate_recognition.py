"""Stage D: accuracy for every detector and recognizer combination, with a
calibrated decision threshold.

    python scripts/evaluate_recognition.py
    python scripts/evaluate_recognition.py --impostors 2

Each person's images are split into a training part and a testing part, the
recognizer is built from the training part only, and the testing part is then
identified. All four combinations of detector and recognizer are reported side
by side.

The threshold is not assumed. Every probe's raw score is recorded and the
threshold is swept afterwards, so the value that goes in the report is the one
calibrated on the team's own images rather than a number copied from a
tutorial.

``--impostors N`` holds N people out of the gallery completely and uses their
images as unknown probes. Without that there is no way to measure the false
acceptance rate, and a face recognition attendance system that cannot say
Unknown will mark strangers present.
"""

from __future__ import annotations

import argparse
import platform
import random
from pathlib import Path
from typing import Dict, List, Sequence

import cv2

from _bootstrap import banner, build_detector, config
from facetrack.detectors import DETECTOR_NAMES
from facetrack.recognizers import RECOGNIZER_NAMES, TrainingSample, get_recognizer
from facetrack.utils import (
    Stopwatch,
    list_dataset,
    save_table,
    split_person_dir,
    write_csv,
)

DETECTOR_TITLES = {"haar": "Haar Cascade", "yunet": "YuNet (DNN)"}
RECOGNIZER_TITLES = {"lbph": "LBPH", "sface": "SFace"}


# ---------------------------------------------------------------- splitting
def split_people(people: Dict[str, List[Path]], test_ratio: float, seed: int,
                 impostor_count: int):
    rng = random.Random(seed)
    names = sorted(people)
    impostor_names = names[len(names) - impostor_count:] if impostor_count else []
    gallery_names = [n for n in names if n not in impostor_names]

    train: Dict[str, List[Path]] = {}
    test: Dict[str, List[Path]] = {}
    for name in gallery_names:
        paths = list(people[name])
        rng.shuffle(paths)
        n_test = max(1, int(round(len(paths) * test_ratio)))
        n_test = min(n_test, len(paths) - 1)       # always leave something to train on
        test[name] = paths[:n_test]
        train[name] = paths[n_test:]

    impostor_paths = [p for name in impostor_names for p in people[name]]
    return train, test, impostor_paths, impostor_names


# ---------------------------------------------------------------- detection cache
def detect_all(detector, paths: Sequence[Path]):
    """Run the detector once per image and remember the largest face."""
    cache = {}
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            cache[path] = None
            continue
        cache[path] = detector.detect_largest(image)
    return cache


# ---------------------------------------------------------------- threshold sweep
def sweep_thresholds(genuine, impostor_scores, higher_is_better, max_points=200):
    """Walk the threshold across the observed scores and score each position.

    ``genuine`` holds (true_label, predicted_label, score) for enrolled probes.
    """
    all_scores = sorted({s for _, _, s in genuine} | set(impostor_scores))
    all_scores = [s for s in all_scores if s not in (float("inf"), float("-inf"))]
    if not all_scores:
        return []

    if len(all_scores) > max_points:
        step = len(all_scores) / max_points
        all_scores = [all_scores[int(i * step)] for i in range(max_points)]

    def accepted(score, threshold):
        return score >= threshold if higher_is_better else score <= threshold

    rows = []
    n_genuine = len(genuine)
    n_impostor = len(impostor_scores)
    for threshold in all_scores:
        correct = sum(
            1 for true, pred, score in genuine
            if pred == true and accepted(score, threshold)
        )
        rejected = sum(1 for _, _, score in genuine if not accepted(score, threshold))
        false_accepts = sum(1 for score in impostor_scores if accepted(score, threshold))
        rows.append({
            "threshold": threshold,
            "accuracy": correct / n_genuine if n_genuine else 0.0,
            "frr": rejected / n_genuine if n_genuine else 0.0,
            "far": false_accepts / n_impostor if n_impostor else None,
        })
    return rows


def pick_best(sweep_rows, has_impostors: bool):
    if not sweep_rows:
        return None
    if has_impostors:
        return max(sweep_rows, key=lambda r: (r["accuracy"] + (1 - r["far"])) / 2)
    return max(sweep_rows, key=lambda r: r["accuracy"])


# ---------------------------------------------------------------- one combination
def evaluate(recognizer_name, detections, train_paths, test_paths, impostor_paths):
    recognizer = get_recognizer(recognizer_name)

    samples = []
    for label, paths in train_paths.items():
        for path in paths:
            detection = detections.get(path)
            if detection is None:
                continue
            image = cv2.imread(str(path))
            if image is None:
                continue
            samples.append(TrainingSample(label=label, image=image, detection=detection))

    if len({s.label for s in samples}) < 2:
        raise ValueError("fewer than two people have usable training faces")
    recognizer.fit(samples)

    genuine = []
    undetected = 0
    total_ms = 0.0
    predictions = 0

    for label, paths in test_paths.items():
        for path in paths:
            detection = detections.get(path)
            if detection is None:
                undetected += 1
                continue
            image = cv2.imread(str(path))
            if image is None:
                undetected += 1
                continue
            with Stopwatch() as watch:
                result = recognizer.predict(image, detection)
            total_ms += watch.ms
            predictions += 1
            genuine.append((label, result.label, result.score))

    impostor_scores = []
    impostor_undetected = 0
    for path in impostor_paths:
        detection = detections.get(path)
        if detection is None:
            impostor_undetected += 1
            continue
        image = cv2.imread(str(path))
        if image is None:
            impostor_undetected += 1
            continue
        result = recognizer.predict(image, detection)
        impostor_scores.append(result.score)

    closed_set = (
        sum(1 for true, pred, _ in genuine if pred == true) / len(genuine)
        if genuine else 0.0
    )
    sweep_rows = sweep_thresholds(genuine, impostor_scores, recognizer.higher_is_better)
    best = pick_best(sweep_rows, bool(impostor_scores))

    default_accepted = [
        (true, pred, score) for true, pred, score in genuine
        if recognizer.passes(score)
    ]
    default_accuracy = (
        sum(1 for true, pred, _ in default_accepted if pred == true) / len(genuine)
        if genuine else 0.0
    )

    return {
        "recognizer": recognizer,
        "genuine": genuine,
        "impostor_scores": impostor_scores,
        "undetected": undetected,
        "impostor_undetected": impostor_undetected,
        "closed_set": closed_set,
        "default_accuracy": default_accuracy,
        "sweep": sweep_rows,
        "best": best,
        "mean_ms": total_ms / predictions if predictions else 0.0,
        "train_faces": len(samples),
    }


# ---------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=None, help="dataset root (default: data/dataset)")
    parser.add_argument("--detectors", nargs="+", choices=DETECTOR_NAMES,
                        default=list(DETECTOR_NAMES))
    parser.add_argument("--recognizers", nargs="+", choices=RECOGNIZER_NAMES,
                        default=list(RECOGNIZER_NAMES))
    parser.add_argument("--test-ratio", type=float, default=0.3,
                        help="share of each person's images used for testing (default: %(default)s)")
    parser.add_argument("--impostors", type=int, default=0,
                        help="people held out of the gallery entirely, to measure false accepts")
    parser.add_argument("--seed", type=int, default=42, help="split seed (default: %(default)s)")
    parser.add_argument("--output", default=None,
                        help="output file stem (default: data/output/recognition_evaluation)")
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
    if len(people) < 2:
        raise SystemExit(f"need at least two people under {dataset_dir}, found {len(people)}.")
    if args.impostors >= len(people) - 1:
        raise SystemExit(
            f"--impostors {args.impostors} leaves fewer than two people in the gallery."
        )

    train_paths, test_paths, impostor_paths, impostor_names = split_people(
        people, args.test_ratio, args.seed, args.impostors
    )
    all_paths = [p for paths in people.values() for p in paths]

    banner("Face Track: recognition evaluation")
    print(f"machine   : {platform.machine()} on {platform.system()} {platform.release()}")
    print(f"dataset   : {dataset_dir}")
    print(f"people    : {len(people)} total, {len(train_paths)} enrolled, "
          f"{len(impostor_names)} held out as unknown")
    print(f"images    : {sum(len(v) for v in train_paths.values())} train, "
          f"{sum(len(v) for v in test_paths.values())} test, {len(impostor_paths)} impostor")
    if impostor_names:
        print("held out  : " + ", ".join(split_person_dir(n)[1] for n in impostor_names))
    else:
        print("note      : no impostors held out, so the false acceptance rate cannot be")
        print("            measured, and the sweep will simply pick the loosest threshold")
        print("            because nothing penalises accepting everyone. Use --impostors 2")
        print("            once there are ten people, otherwise the calibrated threshold")
        print("            below is not a usable operating point.")
    print()
    print("reading the numbers: 'accuracy, no threshold' ignores rejection and is the")
    print("upper bound. The calibrated figure is lower because rejecting a genuine face")
    print("costs accuracy but buys a lower false acceptance rate.")
    print()

    # The per combination sweeps land next to the summary table, so --output keeps
    # every file from one run together.
    summary_stem = Path(args.output) if args.output else config.OUTPUT_DIR / "recognition_evaluation"
    output_dir = summary_stem.parent

    rows = []
    for detector_name in args.detectors:
        try:
            detector = build_detector(args, detector_name=detector_name)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"skipping detector {detector_name}: {exc}\n")
            continue

        print(f"detecting faces with {detector.describe()}")
        detections = detect_all(detector, all_paths)
        found = sum(1 for d in detections.values() if d is not None)
        print(f"  {found}/{len(all_paths)} images produced a face\n")

        for recognizer_name in args.recognizers:
            combo = f"{detector_name} + {recognizer_name}"
            print(f"evaluating {combo}")
            try:
                outcome = evaluate(recognizer_name, detections,
                                   train_paths, test_paths, impostor_paths)
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                print(f"  skipped: {exc}\n")
                continue

            best = outcome["best"]
            recognizer = outcome["recognizer"]
            print(f"  closed set accuracy (threshold ignored): {100 * outcome['closed_set']:.1f}%")
            print(f"  at the default threshold {recognizer.threshold:g}: "
                  f"{100 * outcome['default_accuracy']:.1f}%")
            if best:
                far_text = "n/a" if best["far"] is None else f"{100 * best['far']:.1f}%"
                print(f"  calibrated threshold {best['threshold']:.3f}: "
                      f"{100 * best['accuracy']:.1f}% accurate, "
                      f"FRR {100 * best['frr']:.1f}%, FAR {far_text}")
            print(f"  mean recognition time: {outcome['mean_ms']:.1f} ms\n")

            if outcome["sweep"]:
                stem = output_dir / f"threshold_sweep_{detector_name}_{recognizer_name}"
                save_rows = [
                    {
                        "threshold": f"{r['threshold']:.4f}",
                        "accuracy_%": f"{100 * r['accuracy']:.1f}",
                        "frr_%": f"{100 * r['frr']:.1f}",
                        "far_%": "" if r["far"] is None else f"{100 * r['far']:.1f}",
                    }
                    for r in outcome["sweep"]
                ]
                write_csv(stem.with_suffix(".csv"), save_rows)

            rows.append({
                "Detector": DETECTOR_TITLES[detector_name],
                "Recognizer": RECOGNIZER_TITLES[recognizer_name],
                "Enrolled people": len(train_paths),
                "Test images": len(outcome["genuine"]),
                "No face found": outcome["undetected"],
                "Accuracy, no threshold (%)": f"{100 * outcome['closed_set']:.1f}",
                "Accuracy, calibrated (%)": f"{100 * best['accuracy']:.1f}" if best else "",
                "Calibrated threshold": f"{best['threshold']:.3f}" if best else "",
                "FRR (%)": f"{100 * best['frr']:.1f}" if best else "",
                "FAR (%)": ("" if not best or best["far"] is None
                            else f"{100 * best['far']:.1f}"),
                "Recognition time (ms)": f"{outcome['mean_ms']:.1f}",
            })

    if not rows:
        print("no combination could be evaluated.")
        return 1

    save_table(summary_stem, rows)

    print("\nreminder: put the calibrated threshold into facetrack/config.py, or pass it")
    print("with --threshold, before quoting any accuracy figure in the report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
