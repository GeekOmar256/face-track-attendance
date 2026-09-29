"""Stage A: live face detection. This is the advisor's first implementation task.

    python scripts/detect_live.py                      # Haar Cascade, the baseline
    python scripts/detect_live.py --detector yunet     # the DNN detector, for comparison

On the Raspberry Pi over SSH, when no window can be opened:

    python scripts/detect_live.py --no-display --save-frames data/output/haar_check

Press q to stop.
"""

from __future__ import annotations

import argparse

from _bootstrap import (
    add_detector_args,
    add_display_args,
    add_source_args,
    banner,
    build_camera,
    build_detector,
    build_display,
)
from facetrack.draw import GREEN, draw_banner, draw_detections
from facetrack.utils import FPSMeter, Stopwatch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_detector_args(parser, default="haar")
    add_source_args(parser)
    add_display_args(parser)
    args = parser.parse_args()

    detector = build_detector(args)
    banner(f"Face Track: live detection with {detector.describe()}")

    fps = FPSMeter()
    frames = 0
    frames_with_face = 0
    total_faces = 0
    total_ms = 0.0

    with build_camera(args) as camera, build_display(args, "Face Track - detection") as display:
        print(f"source: {camera!r}")
        print("press q to stop\n")

        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                break

            with Stopwatch() as watch:
                detections = detector.detect(frame)

            frames += 1
            total_ms += watch.ms
            total_faces += len(detections)
            if detections:
                frames_with_face += 1

            fps.tick()
            draw_detections(frame, detections,
                            labels=[f"{d.score:.2f}" for d in detections],
                            colors=[GREEN] * len(detections))
            draw_banner(frame, [
                f"detector: {detector.name}",
                f"faces: {len(detections)}",
                f"detect: {watch.ms:.1f} ms",
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
    print(f"frames processed      : {frames}")
    print(f"frames with a face    : {frames_with_face} ({100 * frames_with_face / frames:.1f}%)")
    print(f"total faces detected  : {total_faces}")
    print(f"mean detection time   : {mean_ms:.1f} ms  ({1000 / mean_ms if mean_ms else 0:.1f} fps)")
    # NFReq-3 in the FYP1 plan allows one second per frame for processing.
    print(f"NFReq-3 (1000 ms/frame): {'PASS' if mean_ms <= 1000 else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
