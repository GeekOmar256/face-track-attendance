# Face Track attendance platform

A separate application alongside versions 1 and 2. SQLite stores people, enrollment images,
sessions, every processed frame, every recognition event, image evidence, attendance and
a durable Firebase synchronization queue. No existing versions are changed.

## Teacher web app

The mobile-style teacher companion is hosted at
https://face-track-bio-attendance.web.app/. It provides student profiles, independent
model selection, live/one-shot sessions, attendance history, reports and Pi software
standby controls. See [TEACHER_SETUP.md](TEACHER_SETUP.md) for the Pi update and setup.
For a complete Pi installation, automatic camera discovery and boot startup, follow
[PI_DEPLOYMENT.md](PI_DEPLOYMENT.md). The combined Pi dependencies are in
`requirements-pi.txt`; `install_pi_service.py` installs the systemd service.

## Models and comparison

Select a **detector** (Haar Cascade, YuNet, dlib HOG) and an **identity recognizer**
(LBPH, SFace, dlib ResNet) independently in Settings. Haar and YuNet do not identify
students themselves. dlib HOG is a CPU face detector; the separate dlib ResNet model
creates 128-dimensional identity embeddings. This implementation uses dlib directly,
so the older face-recognition wrapper and its installation dependencies are unnecessary.

The optional simple Haar gate checks for a face before the selected detector runs.
The selected detector then supplies bounding boxes to the identity recognizer.
Disable the gate for standalone detector comparisons: Haar gate misses otherwise affect
all models. SFace with Haar/dlib uses an unaligned crop; YuNet supplies alignment landmarks.
Use the same source, class, resolution, gate setting and recognizer for detector comparisons.

Thresholds are starting values, not calibrated accuracy claims. Enrollment images and
test images must come from different captures. FPS, counts and scores do not establish
accuracy without labeled ground truth. No liveness/spoof prevention is implemented.

## Windows laptop (Python 3.10 or 3.12)

```powershell
cd "E:\Fall 2026\FYP2\Face Recognition\attendance-platform"
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-desktop.txt
python download_models.py
python doctor.py
python app.py
```

Open http://localhost:8000. A generated admin password is printed in the terminal.
To keep the password across restarts, set `FACETRACK_ADMIN_PASSWORD` in the environment.
The session cookie changes after a server restart; sign in again. Only the Python 3.10
runtime available during development has been exercised; run doctor and tests on 3.12/3.13.

## Raspberry Pi OS, 64 bit, Python 3.13

Use the OS-provided Python that matches Picamera2/libcamera. Do not replace system Python
or install another OpenCV wheel over the working apt build. These commands keep the apt
OpenCV, NumPy and Picamera2 visible inside a virtual environment:

```bash
sudo apt update
sudo apt install -y python3-venv python3-dev python3-opencv opencv-data python3-numpy python3-picamera2
cd ~/face-track-attendance/attendance-platform  # adjust to where this folder was copied
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python doctor.py
python download_models.py
python app.py --host 0.0.0.0 --port 8000
```

Find the address with `hostname -I`; open `http://PI_ADDRESS:8000` on the same network.
requirements.txt intentionally excludes OpenCV, NumPy and Picamera2: apt supplies them
on the Pi. The server has admin login and CSRF protection; use a trusted local network
or HTTPS reverse proxy for access beyond it. Use `tmux` if it must survive SSH disconnection.

### dlib on the Pi / Python 3.13

First try the distribution's package if available for your OS:

```bash
apt-cache policy python3-dlib
sudo apt install -y python3-dlib
python -c "import dlib; print(dlib.__version__)"
```

If unavailable, build against the actual Python 3.13 interpreter inside the venv:

```bash
sudo apt install -y build-essential cmake pkg-config python3-dev libopenblas-dev liblapack-dev
python -m pip install --upgrade pip setuptools wheel
CMAKE_BUILD_PARALLEL_LEVEL=1 python -m pip install --no-cache-dir -r requirements-dlib.txt
python download_models.py --dlib
python doctor.py
```

dlib wheels may be unavailable for ARM64/Python 3.13, triggering a C++ build. A failed
build is not fixed by installing an unrelated Windows wheel or a wheel for Python 3.10.
Use limited parallelism to reduce RAM consumption, active cooling and free disk space.
Compilation can take significant time; failure output from the compiler is needed to
diagnose a specific build. Do not assume Python 3.13 compatibility without the actual
import/inference checks. On Windows a source build requires CMake and MSVC C++ build tools.
Haar/LBPH remains usable when dlib is unavailable; unavailable models are disabled.

## Local workflow

### Standalone camera diagnostic

Stop attendance, its enrollment preview and other camera programs first. This diagnostic
does not use SQLite, recognition or Firebase. It discovers libcamera/Pi cameras and Linux
USB capture devices (requires `sudo apt install v4l-utils`), checks OS/configuration/power
and available kernel logs, and tests cameras sequentially. Windows webcam indexes 0–3 are
only candidates until capture succeeds. A separate capture process lets a watchdog stop
blocked camera reads, including the IMX219 frontend-timeout case.

```bash
python camera_diagnostic.py --list
python camera_diagnostic.py --camera pi --host 0.0.0.0 --seconds 60
```

Open `http://PI_IP:8011` from a computer on the same trusted LAN for live preview, FPS and
results. No desktop/X forwarding is required over SSH. The preview server is read-only
but has no login: do not expose port 8011 to the internet. Default binding is localhost.
After tests finish the server stays open for reviewing results; Ctrl+C closes it.

```bash
python camera_diagnostic.py --no-preview
python camera_diagnostic.py --camera usb --index 0 --host 0.0.0.0
```

Each run saves `data/camera-diagnostic-TIMESTAMP/report.json`, worker logs, and a last-frame
JPEG per tested camera. Failed Pi tests also attempt a bounded native `rpicam-still`
capture when installed. Discovery and successful imports do not prove frames work.
Diagnostic suggestions are hypotheses, not proof of a defective camera, cable or Pi.
Kernel logs unavailable without extra permissions are reported, not silently treated as
healthy. No automatic driver edits, privilege escalation or package changes are performed.

1. Settings: choose detector, recognizer, camera (Auto/Pi/USB), rotation and thresholds.
2. Add a user: ID, name and class. Browse multiple photos, or capture from the device
   camera: the button opens a live preview to adjust positioning, then **Capture photo**
   saves the current frame. Capture repeatedly for different angles; **Close camera**
   releases the device. The preview is local only. Photos need a single subject: one detection, or a dominant detection
   at least twice the area of the next box. Comparable-size faces are rejected. If Haar produces
   spurious boxes, try YuNet and a clear upright portrait. The gallery rebuilds at session start.
3. Attendance: choose session/class and live or one-shot mode. Optionally upload an image/video
   for testing instead of opening the camera. Upload limit is 30 MB.
4. Each known person is marked once per session.
   Every Start/Enable creates a new session, including each
   one-shot click. Stop followed by Enable can record the same person again. The page
   automatically selects the new session so its attendance list is shown.
   Repeated appearances and unknowns remain
   in recognition events. No-face frames also remain in frame statistics. Removal deletes
   enrollment photos and disables the user; earlier attendance retains its name/class/images.
5. Choose a historical session and export Excel (embedded thumbnails and annotated frame images),
   a printable HTML report (browser Print/Save as PDF), or frame-statistics CSV.
   **Export all sessions (Excel)** combines all sessions in one workbook, including attendance
   images, session IDs/names, session summaries, configurations, frame statistics and recognition
   events. The same student is retained separately for each session. Sessions with no attendance
   still appear in the summary and configuration sheets.
   The Excel Frame statistics sheet and local statistics CSV include one row per person
   per session, taken from their first accepted recognition. ID and name identify each row.
   Full frame measurements remain in SQLite and session summaries.

Session configuration snapshots include thresholds, selected algorithms, gate, dimensions,
camera, rotation, OS, Python and OpenCV version. Frame records include gate/detection/recognition
times, pipeline and database-write times, processing FPS, dimensions and face counts.
Face events record detection/identity scores, box, accepted/unknown state, face thumbnail and
annotated frame. Scores differ: LBPH/dlib distance is lower-is-better; SFace cosine is higher-is-better.
Haar has no calibrated confidence score, so its score is null.
Pipeline FPS excludes camera reads, intentional interval waits, database writes and cloud
sync. This is processing capacity, not live acquisition FPS. SQLite timestamps use UTC.
Camera failures set session error; a restart marks unfinished sessions interrupted.
SQLite and JPEG evidence accumulate: back up data/ and plan retention for prolonged runs.

## Firebase mobile admin web app

The mobile dashboard is in firebase/public. Firebase Hosting publishes it as a responsive
web page. It has email/password admin login, attendance images/history, event statistics,
users and remote start/stop/add/remove/settings commands. The Pi camera stream is never synced.
Cloud access requires an external Firebase project; local operation does not require Firebase.

This installation is configured for project `bio-attendance-9zeket`, with the dedicated
Hosting site https://face-track-bio-attendance.web.app and collection
`face_track_devices/classroom-pi`. The existing Bio-Attendance Hosting site and collections
remain separate. Mobile access uses `face_track_admin` and `face_track_device` custom claims.
Server credentials and the generated mobile-admin login are under
`C:/Users/omars/.facetrack/`, outside the source folder. The local
`data/firebase-settings.json` selects the credential file and device ID at startup.
For Pi deployment, securely copy the server credential to a private path on the Pi and
set the environment variables below (or create its own data/firebase-settings.json).

The mobile page has Attendance, People and Settings tabs. Live attendance changes update
the person cards automatically; device status, presence counts and performance use labels
instead of JSON. Reports/performance and advanced parameters are collapsed by default.
Create account registers a Firebase Auth login and a pending access request. An existing
admin approves or declines it under Settings → Account requests. Approval is executed by
the connected attendance server, which needs `roles/firebaseauth.admin` in addition to
`roles/datastore.user`; new accounts cannot grant themselves device access. The server must
be online to process approvals. The account-request rules currently target classroom-pi.

1. Create a Firebase project, a Firestore database and an Email/Password Authentication user.
2. Register a Web app and paste its public configuration into firebase/public/config.js.
   Set the matching DEVICE_ID (for example classroom-pi).
3. Download a service account JSON outside this project folder. Keep it on the Pi only;
   it must never appear in Hosting or the browser. Set the environment:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/home/face/secrets/service-account.json
export FACETRACK_DEVICE_ID=classroom-pi
python -m pip install -r requirements-firebase.txt
python firebase/grant_admin.py FIREBASE_AUTH_USER_UID classroom-pi
python app.py
```

4. Install the Firebase CLI on your deployment computer, then:

```bash
cd firebase
firebase login
firebase use --add
firebase deploy --only hosting,firestore
```

Rules grant access only to an admin claim assigned to that device. The browser can create
commands but cannot alter measurements or results. The privileged Pi writes the results.
Commands expire after five minutes. SQLite records command IDs before execution to avoid
replaying actions on network failures; an interrupted execution requires checking state and
issuing a new command. Commands queue while the Pi is offline, but cannot operate the camera
until it reconnects. Failed sync retries without blocking attendance. Check the cloud status
and queue depth locally. Per-frame cloud writes can incur considerable Firestore usage.

Cloud photos are resized to 640px in the browser. The mobile view offers CSV and printable
image-bearing reports; full XLSX exports are available locally. Mobile frame and event exports
paginate through all synced rows; the live frame subscription shows up to 10,000 records.
The local recognition table shows the first accepted detection per user ID per session;
unknown and repeated detections stay in SQLite, exports and the paginated event API.
Remote raw events displayed are capped at 100; all remain in SQLite and Excel,
with paginated local APIs for older rows. A deployment
must verify Firebase rules/indexes with its own project and run a device-to-phone test.

## Validation

```bash
python -m unittest discover -s tests -v
```

Tests exercise persistence, repeat attendance deduplication, unknown/no-face logging,
session settings, authenticated APIs, person updates/removal, and Excel evidence.
Physical CSI/USB cameras, actual identity accuracy, Python 3.13 builds and Firebase deployment
need validation on the target hardware/account.

References: [Picamera2 installation](https://github.com/raspberrypi/picamera2),
[dlib build instructions](https://www.dlib.net/compile.html),
[Firebase custom admin claims](https://firebase.google.com/docs/auth/admin/custom-claims).
### NumPy / dlib compatibility check

The installed Windows dlib 19.24.0 rejected even a uint8 grayscale image with NumPy 2
during testing. An isolated Python 3.10 environment with NumPy 1.26 is used for desktop
validation. Desktop requirements select NumPy 1.x for Python 3.10/3.12, and NumPy 2.x
for Python 3.13, which cannot use NumPy 1.26. On the Pi rebuild a recent dlib against
the actual OS Python/NumPy if its apt package fails. doctor.py and the settings page
exercise the dlib detector on an actual uint8 array; a failed check disables dlib.
An import succeeding alone is not sufficient evidence that the binary works.

