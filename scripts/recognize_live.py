"""Stage C: the full pipeline, detect then identify, running live.

    python scripts/recognize_live.py --recognizer lbph
    python scripts/recognize_live.py --recognizer sface --detector yunet

A face whose score does not clear the threshold is labelled Unknown rather than
being forced onto the nearest enrolled student. An attendance system that never
says Unknown will mark strangers present, so this behaviour is deliberate and
the false acceptance rate it controls is reported by evaluate_recognition.py.

Press q to stop.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from _bootstrap import (
    add_detector_args,
    add_display_args,
    add_recognizer_args,
    add_source_args,
    banner,
    build_camera,
    build_detector,
    build_display,
    build_recognizer,
    config,
)
from facetrack.draw import AMBER, GREEN, draw_banner, draw_detections
from facetrack.recognizers import default_model_path
from facetrack.utils import FPSMeter, Stopwatch, split_person_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_recognizer_args(parser, default="lbph")
    add_detector_args(parser, default="haar")
    add_source_args(parser)
    add_display_args(parser)
    parser.add_argument("--show-score", action="store_true",
                        help="print the raw match score next to each name")
    args = parser.parse_args()

    detector = build_detector(args)
    recognizer = build_recognizer(args)
    model_path = Path(args.model) if args.model else default_model_path(recognizer.name)
    recognizer.load(model_path)

    banner("Face Track: live recognition")
    print(f"detector   : {detector.describe()}")
    print(f"recognizer : {recognizer.describe()}")
    print(f"model      : {model_path}")
    if recognizer.name == "sface" and detector.name == "haar":
        print("note: SFace behind Haar has no landmarks to align with, so accuracy is")
        print("      expected to be lower than SFace behind YuNet. See the README.")

    fps = FPSMeter()
    frames = 0
    total_ms = 0.0
    seen = Counter()

    with build_camera(args) as camera, build_display(args, "Face Track - recognition") as display:
        print(f"source     : {camera!r}")
        print("\npress q to stop\n")

        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                break

            with Stopwatch() as watch:
                detections = detector.detect(frame)
                results = [recognizer.predict(frame, d) for d in detections]

            frames += 1
            total_ms += watch.ms
            fps.tick()

            labels = []
            colours = []
            for result in results:
                _, name = split_person_dir(result.label)
                if result.is_known:
                    seen[result.label] += 1
                    text = name
                    colours.append(GREEN)
                else:
                    text = config.UNKNOWN_LABEL
                    colours.append(AMBER)
                if args.show_score:
                    text = f"{text} ({result.score:.2f})"
                labels.append(text)

            draw_detections(frame, detections, labels=labels, colors=colours)
            draw_banner(frame, [
                f"{detector.name} + {recognizer.name}",
                f"faces: {len(detections)}",
                f"pipeline: {watch.ms:.1f} ms",
                f"fps: {fps.fps:.1f}",
            ])

            if display.show(frame) == ord("q"):
                break
            if args.max_frames and frames >= args.max_frames:
                break

    if frames == 0:
        print("no frames were read from the source.")
        return 1

    mean_ms = total_ms / frames
    print("\n--- summary ---")
    print(f"frames processed        : {frames}")
    print(f"mean pipeline time      : {mean_ms:.1f} ms  ({1000 / mean_ms if mean_ms else 0:.1f} fps)")
    print(f"NFReq-3 (1000 ms/frame) : {'PASS' if mean_ms <= 1000 else 'FAIL'}")
    if seen:
        print("\nidentified during the run:")
        for label, count in seen.most_common():
            student_id, name = split_person_dir(label)
            print(f"  {name:<28} {student_id:<8} {count} frame(s)")
    else:
        print("\nno enrolled face cleared the threshold during this run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
