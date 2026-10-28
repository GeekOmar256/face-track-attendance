# Raspberry Pi deployment

Teacher website: https://face-track-bio-attendance.web.app/

Use the public repository's **version-2** branch. The application to run is
`attendance-platform/app.py`, not the older version1/version2 applications.

```bash
git clone --branch version-2 --single-branch https://github.com/GeekOmar256/face-track-attendance.git
cd face-track-attendance/attendance-platform
sudo apt update
sudo apt install -y git python3-venv python3-dev python3-opencv opencv-data python3-numpy python3-picamera2 v4l-utils ca-certificates
/usr/bin/python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --timeout 120 --retries 10 -r requirements-pi.txt
.venv/bin/python download_models.py --dlib
```

If this folder already exists, stop its running app/service, back up `data/`, review
local changes, then update the existing checkout instead of overwriting it or cloning
inside it. Do not replace a working Pi OS OpenCV/NumPy/Picamera2 installation with pip
wheels. The venv must use the OS interpreter, normally Python 3.13 on Pi OS Trixie.

## dlib

Try `sudo apt install python3-dlib` if available. Then run `.venv/bin/python doctor.py`
and inspect the dlib health check, not only the import. If absent or incompatible,
build in the venv against its actual OS Python and NumPy:

```bash
sudo apt install -y build-essential cmake pkg-config libopenblas-dev liblapack-dev
.venv/bin/python -m pip install --upgrade pip setuptools wheel
CMAKE_BUILD_PARALLEL_LEVEL=1 MAKEFLAGS=-j1 .venv/bin/python -m pip install --ignore-installed --no-cache-dir --no-build-isolation --no-binary=dlib -r requirements-dlib.txt
.venv/bin/python doctor.py
```

Compilation takes time. Do not install a Windows wheel, an x86 wheel, a wheel for a
different Python, or NumPy 1.x on Python 3.13. Report compiler failures precisely.
Model files are downloaded by the model downloader, not stored in Git.

## Credentials and automatic startup

Use the existing private JSON for project `bio-attendance-9zeket`. Keep it outside
the repository, for example `/home/face/.config/face-track/service-account.json`.
Never print its private key or put it in the public prompt/repository/browser config.
If only the deployment computer has it, transfer the file privately with SFTP/SCP.
The Pi service account needs Firestore access; teacher approvals also need Firebase
Authentication administration permission. Do not broaden permissions unnecessarily.

Stop any existing attendance instance before camera tests:

```bash
.venv/bin/python camera_diagnostic.py --list
timeout --kill-after=3s 30s .venv/bin/python camera_probe.py --seconds 3
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
.venv/bin/python install_pi_service.py --credentials /home/face/.config/face-track/service-account.json
systemctl is-enabled face-track
systemctl is-active face-track
journalctl -u face-track -n 80 --no-pager
```

Adjust the credentials path and home directory to the actual login user. Run the
installer as that user, **without sudo**; it invokes sudo only for systemd setup.
It creates `~/.config/face-track/service.env` (600) and installs
`/etc/systemd/system/face-track.service`, enables boot startup, and starts the app.
The environment contains the credentials file path, device ID `classroom-pi` and a
generated separate local administrator password. The JSON itself stays private.

The service uses port 8010, runs without root, gets available video/render groups,
restarts on process exit, and shuts down its camera processes when stopped. It
starts the platform, not an attendance session; the teacher starts sessions in the app.
It does not depend on an SSH terminal or `.bashrc`. With the user's approval, reboot
and reconnect to verify that `systemctl is-active face-track` is active after boot.

## Automatic camera selection

In Settings choose **Automatic**, or set `camera` to `auto` using the engine while
the service is stopped. Each new capture/preview/session discovers Pi CSI cameras
using Picamera2, prefers the configured index, then tries the others and USB cameras.
On Linux USB candidates come from physical USB video nodes, not Pi codec nodes.
A candidate must deliver a frame to be selected; the first frame is kept for processing.
Pi frame waits time out after five seconds. Explicit Pi/USB modes use their configured
index. Auto mode searches again at the next capture/session; it does not promise
seamless hot-swapping during an existing session.

`camera_probe.py` tests this same application camera class, reports its selected kind,
index, frame count and FPS to `data/camera-probe.json`, and closes the camera afterward.
Use an external `timeout` for the probe: a blocked USB kernel/driver read cannot always
be interrupted by an OpenCV property. The separate diagnostic has subprocess watchdogs.
If no working camera exists, the UI stays available and a capture attempt returns an
error; never claim success based only on a listed sensor. Compare CSI failures with
`rpicam-still --nopreview --timeout 3000 -o /tmp/camera-check.jpg` while the service is stopped.

## Verify the complete path

Open `http://PI_ADDRESS:8010/` for the local page. Its login password is separate from
Firebase; the app prints it in its private service log. On the hosted teacher page,
sign in with the supplied teacher account and check Online / Teacher controls ready.
With a consenting test student, verify enrolment, selected models, one-shot/live
attendance, one entry per person per session, history, edit/remove and reports. A new
session permits a fresh attendance record. Sleep releases the camera but keeps the
OS and bridge running; Wake resumes availability. It cannot power on an off Pi.

Local unit tests and mocked camera fallback tests do not verify physical Pi hardware.
Record the actual target's Python/OpenCV/dlib versions, camera probe result, model
capabilities, service boot state and Firebase heartbeat before reporting deployment done.

References: [Raspberry Pi camera software](https://www.raspberrypi.com/documentation/accessories/camera-software.html),
[Picamera2 manual](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf),
[systemd network startup guidance](https://github.com/systemd/systemd/blob/main/docs/NETWORK_ONLINE.md).
