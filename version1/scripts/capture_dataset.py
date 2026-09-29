"""Stage B: build the enrolment dataset, one person at a time.

    python scripts/capture_dataset.py --id 210001 --name "First Student"

Images land in data/dataset/<id>_<name>/. The folder carries the student ID so
enrolment maps straight onto the student table planned for the attendance
database.

A frame is only saved when exactly one face is visible and the crop is sharp
enough. Without that gate a run produces thirty near identical blurred frames
in a few seconds, and any accuracy measured on them means nothing. Follow the
on screen prompt and change pose and expression while it captures.

The FULL frame is saved, not the detected crop. That matters: if the dataset
held crops cut out by Haar Cascade, then benchmarking YuNet against Haar on
those same images would be circular, because every image would already be one
Haar succeeded on. Full frames keep the detector comparison honest and let the
crop margin be changed later without capturing again.

Press q to stop early, or space to pause.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from _bootstrap import (
    add_detector_args,
    add_display_args,
    add_source_args,
    banner,
    build_camera,
    build_detector,
    build_display,
    config,
)
from facetrack.draw import AMBER, GREEN, RED, draw_banner, draw_detections
from facetrack.utils import blur_score, person_dir_name

POSE_PROMPTS = [
    "look straight at the camera",
    "turn slightly left",
    "turn slightly right",
    "tilt your head a little",
    "smile",
    "neutral expression again",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--id", required=True, help="student ID, for example 210001")
    parser.add_argument("--name", required=True, help='full name, for example "First Student"')
    parser.add_argument("--count", type=int, default=config.CAPTURE_DEFAULT_COUNT,
                        help="how many images to save (default: %(default)s)")
    parser.add_argument("--blur-threshold", type=float, default=config.CAPTURE_BLUR_THRESHOLD,
                        help="minimum sharpness, variance of the Laplacian (default: %(default)s)")
    parser.add_argument("--gap", type=int, default=config.CAPTURE_MIN_FRAME_GAP,
                        help="frames to skip between saves (default: %(default)s)")
    parser.add_argument("--dataset", default=None, help="dataset root (default: data/dataset)")
    parser.add_argument("--overwrite", action="store_true",
                        help="delete any existing images for this person first")
    add_detector_args(parser, default="haar")
    add_source_args(parser)
    add_display_args(parser)
    args = parser.parse_args()

    config.ensure_dirs()
    root = Path(args.dataset) if args.dataset else config.DATASET_DIR
    person_dir = root / person_dir_name(args.id, args.name)
    person_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(p for p in person_dir.glob("*.jpg"))
    if existing and args.overwrite:
        for path in existing:
            path.unlink()
        existing = []
        print(f"cleared previous images in {person_dir}")

    detector = build_detector(args)
    banner(f"Face Track: capturing {args.count} images for {args.name} ({args.id})")
    print(f"detector : {detector.describe()}")
    print(f"folder   : {person_dir}")
    if existing:
        print(f"note     : {len(existing)} image(s) already there, new ones are added after them")

    saved = len(existing)
    target = saved + args.count
    frames_since_save = 0
    paused = False
    rejected_blur = 0
    rejected_count = 0

    with build_camera(args) as camera, build_display(args, "Face Track - capture") as display:
        print(f"source   : {camera!r}")
        print("\nq to stop, space to pause\n")

        while saved < target:
            ok, frame = camera.read()
            if not ok or frame is None:
                break

            detections = detector.detect(frame)
            frames_since_save += 1
            status = ""
            colour = RED

            if paused:
                status = "paused"
                colour = AMBER
            elif len(detections) == 0:
                status = "no face visible"
                rejected_count += 1
            elif len(detections) > 1:
                status = f"{len(detections)} faces, only one person at a time"
                rejected_count += 1
            else:
                detection = detections[0]
                crop = detection.crop(frame, margin=config.CAPTURE_MARGIN)
                sharpness = blur_score(crop)
                if sharpness < args.blur_threshold:
                    status = f"too blurred ({sharpness:.0f} < {args.blur_threshold:.0f})"
                    colour = AMBER
                    rejected_blur += 1
                elif frames_since_save < args.gap:
                    status = "waiting for the next frame gap"
                    colour = AMBER
                else:
                    # The full frame is written, not the crop. See the module docstring.
                    path = person_dir / f"{args.id}_{saved + 1:03d}.jpg"
                    cv2.imwrite(str(path), frame)
                    saved += 1
                    frames_since_save = 0
                    status = f"saved {path.name} (sharpness {sharpness:.0f})"
                    colour = GREEN

            prompt = POSE_PROMPTS[(saved // 5) % len(POSE_PROMPTS)]
            draw_detections(frame, detections, colors=[colour] * len(detections))
            draw_banner(frame, [
                f"{args.name} ({args.id})",
                f"saved: {saved - len(existing)} / {args.count}",
                prompt,
                status,
            ])

            key = display.show(frame)
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused

    print("\n--- summary ---")
    print(f"images saved this run : {saved - len(existing)}")
    print(f"total in folder       : {len(list(person_dir.glob('*.jpg')))}")
    print(f"rejected, blurred     : {rejected_blur}")
    print(f"rejected, face count  : {rejected_count}")
    print(f"folder                : {person_dir}")
    if saved < target:
        print("\nstopped before reaching the target. Run the same command again to add more.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
