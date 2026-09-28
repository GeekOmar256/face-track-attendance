# Face Track — detection and recognition pipeline

FYP2, Team 10, Kuwait College of Science and Technology.
*Face Track: An Embedded Smart Attendance System Using Face Recognition.*

This is the first implementation task: get Haar Cascade face detection running
as the baseline, then build the identification pipeline on top of it, then
compare the baseline against a deep neural network detector.

## Install

Laptop:

```bash
pip install -r requirements.txt
```

Raspberry Pi 4:

```bash
sudo apt install -y python3-opencv python3-picamera2
pip install -r requirements-pi.txt
```

The Haar baseline and the LBPH recognizer need nothing downloaded. For the
YuNet detector and the SFace recognizer, fetch the two ONNX models once:

```bash
python scripts/download_models.py
```

## The four steps

### 1. Detection (the baseline)

```bash
python scripts/detect_live.py                    # Haar Cascade
python scripts/detect_live.py --detector yunet   # the DNN detector
```

On the Pi over SSH with no window system:

```bash
python scripts/detect_live.py --no-display --save-frames data/output/haar_check
```

Every display script falls back to saving frames if it cannot open a window, so
a missing X connection does not kill the run.

### 2. Dataset, about ten people

```bash
python scripts/capture_dataset.py --id 210001 --name "First Student"
python scripts/capture_dataset.py --id 210002 --name "Second Student"
# ...and so on, roughly thirty images each
```

An image is saved only when exactly one face is visible and the crop is sharp
enough, with a gap between saves. Without that gate a run produces thirty
near identical blurred frames and any accuracy measured on them is meaningless.

The **full frame** is saved, not the detected crop. If the dataset held crops
cut out by Haar, then benchmarking YuNet against Haar on those images would be
circular, since every image would be one Haar had already succeeded on.

### 3. Recognition

```bash
python scripts/train.py --recognizer lbph
python scripts/recognize_live.py --recognizer lbph

python scripts/train.py --recognizer sface --detector yunet
python scripts/recognize_live.py --recognizer sface --detector yunet
```

A face whose score does not clear the threshold is labelled **Unknown** instead
of being forced onto the nearest enrolled student.

### 4. The comparison

```bash
python scripts/benchmark_detectors.py      # Haar vs YuNet on identical images
python scripts/evaluate_recognition.py --impostors 2
```

Both write a CSV and a markdown table into `data/output/`, ready to paste into
the report. Run them on the Raspberry Pi as well: those timings are the ones
that answer NFReq-3.

## What is in the box

| Part | Baseline | Advanced |
| --- | --- | --- |
| Detection | Haar Cascade, ships with OpenCV | YuNet, ~230 KB ONNX |
| Recognition | LBPH, ships with opencv-contrib | SFace, ~37 MB ONNX, 128-value embeddings |

Both sides sit behind one interface, so `--detector` and `--recognizer` pick
any of the four combinations without touching the code.

```
facetrack/
  config.py        every tunable number, in one place
  camera.py        picamera2 on the Pi, cv2.VideoCapture on a laptop, or a file
  draw.py          overlays, plus the headless-safe Display wrapper
  utils.py         timing, sharpness, dataset listing, table output
  detectors/       haar.py, yunet.py behind base.py
  recognizers/     lbph.py, sface.py behind base.py
scripts/           the seven command line entry points
data/
  dataset/<id>_<name>/*.jpg
  models/*.onnx
  output/          CSV and markdown result tables
```

## Two things worth knowing

**SFace behind Haar scores lower, by design.** SFace aligns a face using five
landmarks before computing its embedding, and only YuNet produces landmarks.
Haar reports a bounding box and nothing else, so behind Haar the code falls
back to a plain resized crop. The gap this creates is a result for the
comparison table, not a defect. `evaluate_recognition.py` measures all four
combinations precisely so the effect can be quoted with a number.

**Do not quote the default thresholds.** `config.py` ships with
`LBPH_THRESHOLD = 70.0` and `SFACE_COSINE_THRESHOLD = 0.363`. These are
starting points, not results. Run `evaluate_recognition.py`, which records
every probe's raw score and sweeps the threshold afterwards, and put the
calibrated value from the team's own images into the report.

## Not done yet

SQLite attendance logging, the lecturer interface and CSV export are the next
task, once the identification pipeline is confirmed working. Multi-face
capacity, the fifteen simultaneous faces in NFReq-5, is not tested at this
stage either: the ten-person single-face dataset comes first.
