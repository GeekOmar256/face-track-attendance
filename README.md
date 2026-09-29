# Face Track

FYP2, Team 10, Kuwait College of Science and Technology.
*Face Track: An Embedded Smart Attendance System Using Face Recognition.*

An embedded attendance prototype: a Raspberry Pi 4 with a camera detects and
recognises faces on the device itself, with no cloud and no internet.

The project is kept as two versions side by side, so the two can be compared in
the report.

| | [version1/](version1/) | [version2/](version2/) |
| --- | --- | --- |
| Detection | Haar Cascade, YuNet | same |
| Recognition | LBPH, SFace | same |
| Command line tools | 9 scripts | same |
| Browser interface | recognise only | recognise **and enrol** |
| Desktop window | yes | yes |
| Enrolling a new person | command line only | in the browser, with their details |
| Retraining | command line only | a button in the browser |
| Theme | dark | light |

## version1

The baseline. Detection, recognition, the benchmark and evaluation scripts, a
browser interface and a desktop window. Adding a person means running
`capture_dataset.py` and then `train.py` in a terminal, and the only thing
recorded about them is what the folder name encodes.

## version2

Everything in version 1, plus:

- **An "Add a face" page.** Enrol a person from the browser: their student ID,
  name, email, programme, section and notes, then capture their face from the
  camera with a live preview and a progress bar. The details are stored in a
  `person.json` beside their images, so the system holds a real record rather
  than a parsed folder name.
- **A list of enrolled people**, with image counts and a remove button.
- **A Train button**, so a newly enrolled person can be recognised without
  opening a terminal.
- **A light theme.**

## Getting started

Either version runs on its own. Pick one and work inside it:

```bash
cd version2
pip install -r requirements.txt
python scripts/download_models.py     # only for YuNet and SFace
python scripts/run_web.py             # then open the address it prints
```

Each version's own README has the full detail.

## What is not in the repository

Face images, trained models, results and uploads are all ignored by git. Student
face images are biometric data, and the FYP1 plan commits to keeping them on the
device; consent to be photographed for an attendance prototype is not consent to
be published. The two ONNX models are fetched by `scripts/download_models.py`
rather than committed.
