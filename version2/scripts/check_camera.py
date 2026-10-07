"""Diagnose the Raspberry Pi camera, separating hardware faults from this code.

    python3 scripts/check_camera.py

Runs the checks in the order that narrows the problem fastest, and says what
each result means. If this script cannot capture a frame, nothing in the
project can either, and the fault is below the application.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import time

from _bootstrap import banner  # noqa: F401  (puts the project root on sys.path)


def run(cmd, timeout=20):
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                             timeout=timeout)
        return (out.stdout + out.returncode * "" + out.stderr).strip(), out.returncode
    except subprocess.TimeoutExpired:
        return "(timed out)", 124
    except FileNotFoundError:
        return "(command not found)", 127


def section(title):
    print()
    print(title)
    print("-" * len(title))


def main() -> int:
    banner("Face Track: camera check")

    # ---------------------------------------------------------- 1. detection
    section("1. Is the sensor detected?")
    try:
        from picamera2 import Picamera2
    except ImportError:
        print("  picamera2 is not installed.")
        print("  Fix:  sudo apt install -y python3-picamera2")
        return 1

    try:
        info = Picamera2.global_camera_info()
    except Exception as exc:
        print(f"  could not query cameras: {exc}")
        return 1

    if not info:
        print("  NO CAMERA DETECTED.")
        print("  The ribbon is not making contact at all, or the camera is disabled.")
        print("  Power the Pi down fully, reseat the ribbon at both ends, and retry.")
        return 1
    for cam in info:
        print(f"  found: {cam.get('Model', '?')}   {cam.get('Id', '')}")
    print("  Detection only proves the control lines work, not the data lanes.")

    # ------------------------------------------------- 2. is anything holding it
    section("2. Is another process holding the camera?")
    mine = os.getpid()
    holders = []
    for node in sorted(glob.glob("/dev/video*")):
        out, code = run(f"fuser {node} 2>/dev/null")
        if code == 0 and out:
            pids = [p for p in out.split() if p.isdigit() and int(p) != mine]
            if pids:
                holders.append((node, pids))
    if holders:
        for node, pids in holders:
            names = []
            for pid in pids:
                name, _ = run(f"ps -p {pid} -o comm=")
                names.append(f"{pid} ({name})")
            print(f"  {node} is held by: {', '.join(names)}")
        print("\n  A running Face Track server will hold the camera. Stop it first:")
        print("    pkill -f run_web.py")
    else:
        print("  nothing else is holding the camera.")

    # ------------------------------------------------------------ 3. power
    section("3. Power and thermals")
    out, code = run("vcgencmd get_throttled")
    if code == 0 and "throttled" in out:
        print(f"  {out}")
        try:
            value = int(out.split("=")[1], 16)
        except (IndexError, ValueError):
            value = 0
        if value == 0:
            print("  no under-voltage recorded.")
        else:
            if value & 0x1:
                print("  UNDER-VOLTAGE RIGHT NOW. The supply cannot hold 5 V.")
            if value & 0x10000:
                print("  under-voltage has occurred since boot.")
            print("  A weak supply is a known cause of camera frontend timeouts.")
            print("  Use the official 5 V 3 A supply and avoid long or thin USB cables.")
    else:
        print("  vcgencmd unavailable, skipping.")

    # ------------------------------------------------------ 4. actual capture
    section("4. Can it actually deliver frames?")
    print("  starting the camera and grabbing 5 frames ...")
    picam = None
    try:
        picam = Picamera2()
        cfg = picam.create_preview_configuration(main={"size": (640, 480),
                                                       "format": "RGB888"})
        picam.configure(cfg)
        picam.start()
        times = []
        for n in range(5):
            start = time.perf_counter()
            frame = picam.capture_array()
            times.append((time.perf_counter() - start) * 1000)
            print(f"    frame {n + 1}: {frame.shape}  {times[-1]:.0f} ms")
        print(f"\n  CAPTURE WORKS. mean {sum(times) / len(times):.0f} ms per frame.")
        print("  The camera is fine. If the application still fails, the fault is in")
        print("  the application, so send this output and the server log.")
        return 0
    except Exception as exc:
        print(f"\n  CAPTURE FAILED: {type(exc).__name__}: {exc}")
        print()
        print("  The sensor is detected but will not stream. This is below the")
        print("  application, so no change to this project can fix it. In order:")
        print("    1. Power the Pi down fully, then reseat the ribbon at BOTH ends.")
        print("       Contacts face away from the Ethernet port on the Pi.")
        print("    2. Test without this project at all:   rpicam-hello -t 5000 -n")
        print("       If that times out too, it is hardware or the system, not us.")
        print("    3. Try a different ribbon cable. They fail more often than sensors.")
        print("    4. Check the power supply, see section 3 above.")
        return 1
    finally:
        if picam is not None:
            for close in (getattr(picam, "stop", None), getattr(picam, "close", None)):
                try:
                    close and close()
                except Exception:
                    pass


if __name__ == "__main__":
    raise SystemExit(main())
