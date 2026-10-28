# Prompt for the AI connected to the Raspberry Pi

Deploy and verify my Face Track attendance platform on the Raspberry Pi connected
to this computer. Perform the work; do not only describe commands. Use the existing
SSH connection if available. Discover the actual Pi address/login user from the
connection; ask me only if required connection information is missing. Run all Linux
installation commands on the Pi, not this Windows/macOS computer.

Project details:

- Public repo: https://github.com/GeekOmar256/face-track-attendance
- Branch: `version-2` (important: the default main branch contains older work).
- App folder: `attendance-platform`.
- Teacher web app: https://face-track-bio-attendance.web.app/
- Teacher Firebase login: `admin@gmail.com`; use the password supplied privately
  by the owner. Do not publish it in this repository.
- Firebase project: `bio-attendance-9zeket`.
- Firebase device ID: `classroom-pi`.
- Local web port: `8010`.
- Expected hardware: Raspberry Pi 4, 8 GB, CSI Pi camera; let actual discovery
  establish the sensor model. Support a USB webcam fallback.
- Expected OS: 64-bit Raspberry Pi OS / Debian 13 Trixie with OS Python 3.13.
  Verify the actual versions instead of assuming.

Private Firebase credential handling:

The service-account JSON was previously downloaded to the Pi. Locate it without
printing its contents or private key. Look in the login user's private folders such
as `~/secrets`, `~/face/secrets`, `~/.config/face-track`, and Downloads. Validate only
its type (`service_account`), project ID (`bio-attendance-9zeket`), and required field
presence. Do not include it in chat, git, logs, frontend files or screenshots.
If found, retain the original and put a private copy at
`~/.config/face-track/service-account.json`; directory mode 700, file mode 600, owned
by the Pi login user. If no copy exists on the Pi but one exists on this computer,
transfer that file privately by SFTP/SCP. If it exists nowhere accessible, ask me
to supply the file privately and continue independent setup in the meantime. Do not
invent a key, disable security rules, or provision an unrelated Firebase project.
The browser already has its public Firebase config in the repository; that is not
a substitute for the private device credential.

1. Inspect and preserve the current installation.

Check `/etc/os-release`, `uname -m`, `/usr/bin/python3 --version`, available disk,
camera processes, and existing Face Track services. The usual folder is
`~/face-track-attendance/attendance-platform`. Stop the identified old platform
service/process cleanly before changing files or opening the camera. Do not kill
unrelated Python processes. Back up its `data/` folder, including SQLite/WAL/SHM
files while the app is stopped, outside the checkout. Preserve students, photos,
attendance, models, existing credentials and working camera configuration. Inspect
local Git changes and preserve them; never use `git reset --hard` or delete the folder.

For a new install:

```bash
sudo apt update
sudo apt install -y git
cd ~
git clone --branch version-2 --single-branch https://github.com/GeekOmar256/face-track-attendance.git
cd ~/face-track-attendance/attendance-platform
```

For an existing checkout, fetch the repo, use `version-2`, and update with
`git pull --ff-only origin version-2` after preserving local work. Resolve divergence
without discarding data. Read `attendance-platform/PI_DEPLOYMENT.md` and
`TEACHER_SETUP.md`. Run this platform, not the older `version1` or `version2` app.

2. Install libraries with the actual OS Python.

```bash
sudo apt update
sudo apt install -y git python3-venv python3-dev python3-opencv opencv-data python3-numpy python3-picamera2 v4l-utils ca-certificates
cd ~/face-track-attendance/attendance-platform
/usr/bin/python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --timeout 120 --retries 10 -r requirements-pi.txt
```

Before reusing an existing `.venv`, verify that it uses the system interpreter and
includes system site packages. If unsuitable, preserve it under another name before
creating the correct venv. Do not install pip OpenCV/NumPy/Picamera2 over working apt
packages, do not use `--break-system-packages`, and do not replace system Python.
Keep TLS verification enabled; retry interrupted downloads with longer timeout.

Install all available model support. Try `sudo apt install python3-dlib` if available,
then run `.venv/bin/python doctor.py` to check actual dlib inference. If apt dlib is
missing or fails the NumPy/Python health check, use:

```bash
sudo apt install -y build-essential cmake pkg-config libopenblas-dev liblapack-dev
.venv/bin/python -m pip install --upgrade pip setuptools wheel
CMAKE_BUILD_PARALLEL_LEVEL=1 MAKEFLAGS=-j1 .venv/bin/python -m pip install --ignore-installed --no-cache-dir --no-build-isolation --no-binary=dlib -r requirements-dlib.txt
.venv/bin/python download_models.py --dlib
.venv/bin/python doctor.py
```

Allow the dlib source build enough time; inspect failures and fix dependencies.
Do not use Windows/x86 or wrong-Python wheels, and do not downgrade Python 3.13 to
NumPy 1.x. This app uses dlib directly; installing the separate `face-recognition`
wrapper is unnecessary. Confirm Haar/YuNet/dlib detectors and LBPH/SFace/dlib
recognizers are usable; clearly identify any that genuinely cannot be installed.

3. Verify real camera capture and automatic selection.

Stop all identified attendance/preview instances first. Run:

```bash
.venv/bin/python camera_diagnostic.py --list
timeout --kill-after=3s 30s .venv/bin/python camera_probe.py --seconds 3
```

Inspect `data/camera-probe.json`: require `ok: true`, positive frames/FPS and the
selected camera type/index. This probe uses the same Camera class as the application.
Automatic mode discovers Pi CSI cameras, checks a real frame, then falls back to
other Pi/USB candidates. Set the saved camera setting to Automatic while stopped:

```bash
.venv/bin/python -c 'from pathlib import Path; from store import Store; from engine import Engine; Engine(Store(Path("data/attendance.sqlite3"))).save_settings({"camera":"auto"})'
```

Do not assume a listed IMX219/IMX708 proves streaming works. If no frames arrive,
run the independent diagnostic with `--camera auto --no-preview --seconds 3 --timeout 8`.
Compare with `rpicam-still --nopreview --timeout 3000 -o /tmp/camera-check.jpg` while
the service is stopped. Diagnose busy devices, permissions, cable/power and driver
problems from evidence. Do not blindly change boot overlays or claim software fixed
broken hardware. Never reseat a ribbon cable while powered. If hardware remains
blocked, finish service/web setup and explicitly report the camera blocker.

4. Install and start the persistent service.

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
.venv/bin/python install_pi_service.py --credentials "$HOME/.config/face-track/service-account.json" --device classroom-pi --port 8010
systemctl is-enabled face-track
systemctl is-active face-track
systemctl status face-track --no-pager
journalctl -u face-track -n 80 --no-pager
hostname -I
```

Run the installer as the Pi login user, without a leading sudo. It invokes sudo
only to install/enable/restart `face-track.service`. The service runs as that user,
loads its private environment file, joins available camera groups, restarts on exit,
and starts at boot without an SSH terminal. It starts the platform; the teacher
activates an attendance session separately. Do not leave a second manual app process
running. Read logs privately and redact any local administrator password in reports.

5. Verify Firebase, the web app, and reboot persistence.

Check the local page at `http://PI_ADDRESS:8010/`. The local password is separate
from the hosted teacher login and is generated/preserved by the service installer.
Use the teacher website with the supplied login. Confirm a fresh `classroom-pi`
heartbeat, **Online**, and **Teacher controls ready**. Do not expose credentials to
browser JavaScript or loosen Firestore rules to fix connection errors. If denied,
inspect the private service account's actual Firestore permissions; account approval
also requires Firebase Auth administration permission. Keep project/device IDs exact.

With my consenting test student/photos, verify adding/editing/removing profiles,
model selection, one-shot and live session start/stop, automatic attendance updates,
one attendance row per person per session, a fresh row in a new session, history
and reports. Avoid modifying unrelated real student records. Sleep means software
standby: it stops attendance and releases the camera while keeping Linux/Firebase
online. Wake clears standby; it cannot power on a physically off Pi.

After setup, save work, reboot the Pi once and reconnect to prove the app starts
without a manual login. If this would interrupt another active user or process,
request a reboot window first. Verify service enabled/active, local HTTP response,
fresh Firebase heartbeat and successful camera capture after boot (stop the service
for an independent probe, then restart it). Do not report success based only on
unit tests or a running process.

Finish with a concise deployment report: Pi hostname/IP, OS/Python versions, commit
and branch, repo location, service state after reboot, camera model/type/index and
actual frame result, available detector/recognizer pairs, Firebase connectivity,
local/hosted URLs, and any remaining blockers. Do not include private keys or tokens.
