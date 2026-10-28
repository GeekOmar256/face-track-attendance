# Face Track teacher companion

Open https://face-track-bio-attendance.web.app/ on your phone or computer.
Sign in with the teacher account provided with this release. On your phone, use the
browser's Add to Home Screen option for an app-style shortcut.

## Pages

- **Overview:** choose a class and session name, select Live or One shot, start/stop,
  and watch attendance arrive automatically. One student appears once per session.
- **Students:** search, add a student with ID/name/class/photos, edit a profile,
  append face photos, or remove a student. Removal keeps historical attendance.
  A new enrolment needs photos. Uploaded photos are resized before sending.
  Optional Pi capture takes one photo at the device; position the student first.
- **History:** search sessions, load older sessions, and open their attendance.
  Session reports include attendance CSV and a printable report with saved evidence.
  Statistics and event CSV are available under Session reports & performance.
- **Models:** select Haar, YuNet or dlib HOG detection and LBPH, SFace or dlib
  ResNet recognition separately. Unsupported models are disabled by the Pi's
  capability report. Stop attendance before changing settings or student records.
- **Device:** review connectivity, use Sleep/Wake, sign out, or approve teacher
  account requests. New teacher accounts require approval through the connected Pi.

## Update an existing Raspberry Pi installation

This release needs the updated `app.py`, `cloud.py`, `engine.py` and `store.py` on
the Pi. The website reports **Update required** if its device software is older.
The ZIP supplied with this release contains these four files and this guide.

1. Stop the Python platform with Ctrl+C (or stop its service if configured).
2. Back up the existing attendance-platform folder, including `data/`, while stopped.
3. Copy the four Python files from the ZIP into
   `~/face-track-attendance/attendance-platform/`, replacing those four existing files.
   Keep all other modules, the existing virtual environment, models and data.
4. Restart using the commands below. The new SQLite standby table is created
   automatically; existing students, photos and attendance are retained.

```bash
cd ~/face-track-attendance/attendance-platform
source .venv/bin/activate
python -m pip install -r requirements-firebase.txt
export FACETRACK_DEVICE_ID=classroom-pi
# Keep GOOGLE_APPLICATION_CREDENTIALS set to your existing service-account JSON path.
test -r "$GOOGLE_APPLICATION_CREDENTIALS" && echo "Firebase credentials file found"
python app.py --host 0.0.0.0 --port 8010
```

Use the existing system-Python-compatible virtual environment on the Pi. This update
does not require reinstalling OpenCV, Picamera2 or dlib. If credentials are configured
instead in `data/firebase-settings.json`, retain that file; the bridge also supports it.
Never upload the service-account JSON to Hosting or commit it to Git.

Keep the Pi online with the Python app running. After startup, the teacher interface
should show **Online** and **Teacher controls ready**. The commands are acknowledged
by the Pi before the interface reports success. An offline Pi leaves history readable
but disables device actions. A request without a reply may still have executed: check
the Pi/history before repeating it. Remote commands expire after five minutes.

## What Sleep and Wake do

Sleep stops attendance, closes the camera and persists software standby in SQLite.
The operating system and Firebase bridge stay running. Wake clears standby; start
a session separately when ready. These controls do not cut power, shut down Linux,
or wake a physically powered-off Pi. Physical power control needs suitable external
hardware or local power-on.

## Hosting and teacher access administration

The Firebase project is `bio-attendance-9zeket`; the Hosting site is
`face-track-bio-attendance`. The site uses the existing `classroom-pi` device data.
Attendance and profile reads require an authenticated, device-approved teacher.
Passwords and service-account secrets are not embedded in the web files.

To provision another teacher administratively, use an authorized private credential
and run the following from the attendance-platform folder. The password is prompted
without being saved in source code; existing accounts are not reset unless requested.

```bash
python firebase/provision_teacher.py --email teacher@example.com --device classroom-pi --credentials /path/to/service-account.json
```

Change the requested demo password before using real student information. The local
Python page retains its separate existing administrator login.

To publish subsequent website changes:

```bash
cd firebase
npm ci
npm test
firebase deploy --only hosting,firestore:rules --project bio-attendance-9zeket
```

## Validation scope

Automated checks cover backend profile updates, historical record retention, persistent
standby, authentication/CSRF checks, command dispatch, web navigation, offline controls,
live attendance rendering, deduplication and acknowledgement handling. Live Pi camera
capture and physical wake hardware require testing on the actual device. No physical
power control is supplied by this release.
