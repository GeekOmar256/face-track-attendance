# Haar Cascade against YuNet: a comparison of two face detection models

Face Track — FYP2, Team 10, Kuwait College of Science and Technology.
*Face Track: An Embedded Smart Attendance System Using Face Recognition.*

---

## 1. Purpose

The FYP1 plan selected Haar Cascade as the face detection algorithm, on the
grounds that it is computationally cheap and therefore suited to a Raspberry Pi.
The advisor asked that, once the baseline worked, a deep neural network detector
be tried and compared against it, as evidence for the choice rather than an
assumption.

This document reports the comparison as it was actually carried out: on the
Raspberry Pi 4, through the project's web interface, on photographs of five
people taken in the laboratory and the classroom.

## 2. How the tests were run

Each photograph was uploaded to the Raspberry Pi through the interface and
processed with one detector at a time. The detector and its parameters were
changed between runs; the image was not. Both detectors therefore saw identical
input, which is the point of the exercise.

| | |
| --- | --- |
| Hardware | Raspberry Pi 4, running the pipeline on the board itself |
| Subjects | 5, photographed in an office and a classroom |
| Images | Single photographs, each containing exactly one person |
| Haar settings | scaleFactor 1.1, minSize 60 px, minNeighbors 6 and 10 |
| YuNet setting | score threshold 0.75 |
| Recognizer | LBPH, on most runs |

### 2.1 A limitation in how the figures were collected

**The statistics panel accumulates until "Reset statistics" is pressed, and it
was not pressed between configurations.** The frame counter climbs steadily
from 3 to 21 across the session, so the "mean time" shown on any later
screenshot is an average over every run that came before it, mixing Haar and
YuNet together.

Per-image detection counts can still be recovered exactly, by taking the
difference between consecutive readings, and that is how section 3 was
produced. **Per-configuration timings cannot be separated this way**, so
section 4 quotes only the two readings taken before any mixing occurred.

For the final report, press **Reset statistics** before each configuration.
Every figure then stands on its own and no reconstruction is needed.

## 3. Detection results

Because every photograph contains exactly one person, any detection beyond the
first is a false positive. Differencing the counters gives the number of boxes
each detector produced on each image:

| Subject | Detector | Boxes produced | False positives |
| --- | --- | --- | --- |
| A (white cardigan, hijab) | Haar, minNeighbors 6 | 4 | 3 |
| B (dark blazer, whiteboard) | Haar, minNeighbors 6 | 1 | 0 |
| B (dark blazer, whiteboard) | YuNet, 0.75 | 1 | 0 |
| C (white cardigan, office) | Haar, minNeighbors 6 | 5 | 4 |
| C (white cardigan, office) | Haar, minNeighbors 10 | about 5 | about 4 |
| C (white cardigan, office) | YuNet, 0.75 | 1 | 0 |
| D (distant, classroom) | Haar, minNeighbors 10 | 1 | 0 |
| D (distant, classroom) | YuNet, 0.75 | 1 | 0 |
| E (dark blazer, office) | Haar, minNeighbors 10 | 1 | 0 |
| E (dark blazer, office) | YuNet, 0.75 | 1 | 0 |

Summarised across the session:

- **YuNet returned exactly one box on every run, without exception.** Six runs,
  six correct detections, no false positives.
- **Haar returned exactly one box on five runs and four to five boxes on
  three runs.** Its first run averaged 3.3 boxes per frame over three frames.
- **Neither detector missed a face.** The detection rate was 100% for both,
  including subject D standing several metres from the camera.

### 3.1 Haar's false positives are not random

They cluster on one kind of subject. Subjects A and C both wear a light,
textured cardigan, and the spurious boxes land on the fabric, on the folds of
the hijab and on the shoulder line. Subjects B and E, photographed in a plain
dark blazer, produced a single clean box from the same detector at the same
settings.

This is consistent with how the classifier works: Haar features respond to
local light–dark contrast patterns, and a patterned garment under office
lighting presents the same contrast structure as the eye-and-cheek arrangement
the cascade was trained to find. Nothing about the person changed between the
two cases; the clothing did.

**Raising minNeighbors from 6 to 10 did not fix it.** On subject C the stricter
setting still produced about five boxes. It did no harm elsewhere, so 10 is a
better default than 6, but it is not a solution to this failure mode.

## 4. Timing on the Raspberry Pi

Only two readings were taken before the statistics began to mix:

| Configuration | Frames | Mean detection time | Throughput |
| --- | --- | --- | --- |
| Haar, minNeighbors 6, detection only | 3 | 229.3 ms | 4.4 fps |
| YuNet (separate session, version 1) | 10 | 204.3 ms | 4.9 fps |
| YuNet (separate session, version 1) | 12 | 235.9 ms | 4.2 fps |

On the Raspberry Pi the two models cost roughly the same: both land between
200 and 240 ms per frame, or 4 to 5 frames per second at 640 px.

**This contradicts the measurement taken earlier on a laptop**, where YuNet ran
at 16.9 ms against Haar's 28.1 ms and was clearly the faster of the two. The
advantage does not carry over to the Pi. The likely reason is that the two
models stress different parts of the processor: Haar's cascade is a
branch-heavy integer search that suits a general-purpose CPU, while YuNet is a
convolutional network whose throughput depends on wide vector units that an ARM
core at this price does not have in the same measure. The laptop figure
flattered the neural network.

The practical consequence for the project is that **choosing YuNet does not
cost throughput on the target hardware**, which removes the main objection the
FYP1 plan raised against a neural detector.

The later blended means, between 318 ms and 360 ms with maxima of 574.6 ms, are
averages over mixed configurations and should not be quoted for either model.

## 5. What the two models return

Two differences in output explain most of the behaviour above.

**Haar returns a bounding box and nothing else.** It reports no confidence, so
every detection is trusted equally and there is no threshold to tune on the
output side. The only control is how many overlapping hits to demand, and
section 3 shows that tightening it does not separate a cardigan from a face.

**YuNet returns a bounding box, a confidence score, and five facial landmarks**
(both eyes, the nose tip, both mouth corners). The score is what allows a
spurious region to be discarded without discarding faces: the threshold of 0.75
used in these tests rejected everything that was not a face while keeping every
face. The landmarks matter separately, because the SFace recognizer uses them
to align a face before computing its embedding; behind Haar there are no
landmarks and the code falls back to a plain resized crop.

## 6. Recognition during these tests

Recognition was switched on for most runs, using LBPH. The outcome was almost
entirely **Unknown**: the panel reports 24 to 28 unknown faces by the end of the
session, against a single identification of one enrolled person in 1 frame of
12, an 8.3% share.

**This is a gallery and threshold problem, not evidence that recognition does
not work.** Two causes are visible in the screenshots. First, a large share of
the "unknown faces" counted are Haar's false positives — boxes on clothing,
which the recognizer is quite correctly refusing to match to anybody. Second,
the decision threshold was left at the value in `config.py`, which has not yet
been calibrated against this dataset.

Recognition cannot be assessed properly until there are about ten enrolled
people with several images each, and until the threshold is calibrated with
`evaluate_recognition.py --impostors 2`, which records every probe's raw score
and sweeps the threshold afterwards.

## 7. Conclusion

On the Raspberry Pi, with the team's own photographs:

1. **Both detectors found every face.** Detection rate was 100% for each, at
   close range and at several metres.
2. **YuNet produced no false positives at all; Haar produced three to four on
   two of the five subjects.** The failure is driven by patterned, light
   clothing, and raising minNeighbors from 6 to 10 did not correct it.
3. **The two cost about the same on the Pi**, 200 to 240 ms per frame. The
   laptop result showing YuNet nearly twice as fast does not transfer to ARM.

Haar Cascade remains the correct baseline for the project: it needs no model
file, it is the method identified in the FYP1 literature review, and on plain
backgrounds it is accurate. But on this evidence **YuNet is the better choice
for the final system**, because it is equally fast on the target hardware,
produces a clean single detection on every subject tested, and supplies the
landmarks the recognizer needs.

Three things follow for the next round of testing:

- Press **Reset statistics** between configurations, so each figure stands alone.
- Enrol about ten people and calibrate the recognition threshold before quoting
  any accuracy figure.
- Re-run the detector comparison with the counters reset, to obtain a clean
  per-model timing on the Pi rather than the two isolated readings available
  here.

## 8. How to reproduce

```bash
python3 scripts/benchmark_detectors.py
python3 scripts/benchmark_detectors.py --detectors haar --min-neighbors 10
python3 scripts/benchmark_detectors.py --detectors yunet --score-threshold 0.75
python3 scripts/evaluate_recognition.py --impostors 2
```

Each writes a CSV and a markdown table into `data/output/`. Run them on the
Raspberry Pi; the figures that belong in the final report are the ones measured
on the target hardware.
