"""Stop this installation's duplicate app processes, then start only its systemd service."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def belongs_to_installation(arguments, cwd, root):
    if not arguments or not Path(arguments[0]).name.startswith("python"):
        return False
    root = Path(root).resolve()
    for argument in arguments[1:]:
        if Path(argument).name != "app.py":
            continue
        path = Path(argument)
        if not path.is_absolute():
            if cwd is None:
                continue
            path = Path(cwd) / path
        if path.resolve() == root / "app.py":
            return True
    return False


def matching_processes(root):
    matches = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            if process.stat().st_uid != os.getuid():
                continue
            arguments = (process / "cmdline").read_bytes().decode(errors="replace").strip("\0").split("\0")
            cwd = (process / "cwd").resolve(strict=True)
            if belongs_to_installation(arguments, cwd, root):
                matches.append(int(process.name))
        except (OSError, ValueError):
            continue  # Process exited while inspecting it, or belongs to another user.
    return matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Only list matching processes; do not stop anything")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Run this command on the Raspberry Pi")
    if os.geteuid() == 0:
        parser.error("Run as the Pi login user without sudo; service control will request sudo")
    root = Path(__file__).resolve().parent
    if args.check:
        print("App processes for this installation:", matching_processes(root))
        return
    # Stop automatic restarts first, then terminate only exact app.py paths owned by this user.
    subprocess.run(["sudo", "systemctl", "stop", "face-track.service"], check=True)
    for pid in matching_processes(root):
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"Stopped duplicate attendance process {pid}")
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 10
    while matching_processes(root) and time.monotonic() < deadline:
        time.sleep(.2)
    remaining = matching_processes(root)
    if remaining:
        parser.exit(1, f"App processes still stopping: {remaining}. Service left stopped; inspect before restarting.\n")
    subprocess.run(["sudo", "systemctl", "start", "face-track.service"], check=True)
    subprocess.run(["systemctl", "is-active", "face-track.service"], check=True)
    print("Only the systemd service was started. Open the teacher website; allow a few seconds for status.")


if __name__ == "__main__":
    main()
