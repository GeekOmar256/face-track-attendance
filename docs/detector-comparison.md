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

This document compares the two models on identical images and reports what was
measured. Both detectors run behind the same interface in the project code, so
they are selected with a flag and see exactly the same frames.

## 2. Test conditions, and what they do not cover

| | |
| --- | --- |
| Images | 30, each containing exactly one face |
| Identities | 2 |
| Frame size | 640 px wide |
| Platform | Windows laptop, AMD64 |
| Repetitions | Timing averaged over 3 passes after a warm-up pass |

**These conditions are a limitation and the figures must be read with it in
mind.** The 30 images are photometric and geometric variations derived from two
source photographs, not 30 independent captures of 30 people. They are adequate
for comparing two detectors on identical input, which is what this document
does, and they are *not* adequate for quoting an absolute accuracy figure for
the system. The measurements also come from a laptop, not from the Raspberry Pi
the system will run on, so the timings establish the *relative* cost of the two
models rather than the figure that answers the project's timing requirement.

Ground truth is available without hand-labelling because every image was
captured or selected to contain exactly one face. Two measures follow directly:
the **detection rate**, the share of images in which at least one face was
found, and the **false positive count**, the number of boxes beyond the first.

## 3. The two models

**Haar Cascade** is the Viola–Jones classifier from 2001, and is the method the
FYP1 literature review identified as the common choice for attendance systems.
It slides a window across a greyscale image and applies a cascade of simple
rectangular contrast tests, rejecting non-face regions as early as possible. It
is not a learned feature extractor in the modern sense: the features are
hand-designed and the training selects which to use. The cascade ships inside
OpenCV as a 908 KB XML file, so it needs no download.

**YuNet** is a small convolutional neural network released through the OpenCV
model zoo. It is a 227 KB ONNX file and runs through OpenCV's own DNN module,
which matters for this project: it needs no TensorFlow, no PyTorch and no dlib,
so the Raspberry Pi install stays at OpenCV plus NumPy.

Two differences in what they return shape the rest of this comparison:

- **Haar returns a bounding box and nothing else.** It reports no confidence, so
  every detection is equally trusted and there is no threshold to tune on the
  output side. Tuning happens only through the search parameters.
- **YuNet returns a bounding box, a confidence score, and five facial landmarks**
  (both eyes, the nose tip, both mouth corners).

## 4. Detection results

At the configuration each model was first run with, and then at the best
configuration found for each:

| Configuration | Detection rate | Faces found | False positives | Mean time | Max time |
| --- | --- | --- | --- | --- | --- |
| Haar, defaults (scaleFactor 1.1, minNeighbors 5) | 96.7% | 29 / 30 | 4 | 28.2 ms | 44.9 ms |
| Haar, tuned (scaleFactor 1.1, minNeighbors 7) | 96.7% | 29 / 30 | **0** | 28.1 ms | 42.9 ms |
| Haar, fast (scaleFactor 1.2, minNeighbors 7) | 90.0% | 27 / 30 | 0 | 16.8 ms | 25.6 ms |
| YuNet, default (score threshold 0.90) | **46.7%** | 14 / 30 | 0 | 16.0 ms | — |
| YuNet, tuned (score threshold 0.70) | **100%** | 30 / 30 | **0** | 16.9 ms | 25.7 ms |

Read on its own, the first and last rows say YuNet is better on every axis:
it found every face, produced no false detections, and did so in roughly 60% of
the time. That conclusion survives a fairer comparison, but only after two
things are said about the defaults.

## 5. Both default configurations are wrong for this data

This is the most useful finding, and it applies to each model in the opposite
direction.

### 5.1 Haar's default accepts false positives

Varying `minNeighbors`, which sets how many overlapping detections are required
before a region is accepted:

| minNeighbors | Detected | Missed | False positives |
| --- | --- | --- | --- |
| 1 | 30 | 0 | 19 |
| 3 | 30 | 0 | 7 |
| **5 (default)** | 29 | 1 | **4** |
| **7** | 29 | 1 | **0** |
| 9 | 29 | 1 | 0 |
| 12 | 27 | 3 | 0 |

Raising the value from 5 to 7 removed every false positive at no cost in
detection rate and no cost in time. The project's default should be 7.

`scaleFactor`, which sets how much the image shrinks between passes, trades
accuracy against speed as expected: 1.05 costs 53.0 ms per image, 1.1 costs
28.5 ms, 1.2 costs 16.7 ms, and 1.3 drops to 10.5 ms but misses 5 of 30 faces.

The practical consequence of Haar's false positives is not abstract. During
enrolment, a single photograph of one person was reported as **five faces**,
because the classifier fired on hair, beard and background texture. The
enrolment code had to be changed to accept the largest detection when it clearly
dominates the others, rather than refusing any image with more than one box.

### 5.2 YuNet's default rejects real faces

The confidence YuNet assigns to these faces is lower than its default threshold
expects:

- lowest 0.705, median 0.875, highest 0.921
- **16 of 30 images scored below the 0.90 default**

So at the shipped threshold YuNet found only 14 of 30 faces, a 46.7% detection
rate that would make it look far worse than Haar. Lowering the threshold to 0.70
found every face and still produced no false positives, because the gap between
genuine faces (0.705 and above) and spurious regions is wide on this data.

The clearest single case is the night-time photograph used for an early test:

| Detector | Result |
| --- | --- |
| Haar, defaults | 1 face |
| YuNet at 0.90 | **0 faces** |
| YuNet at 0.70 | 1 face, confidence 0.850 |

The image is a close-range selfie under harsh artificial light, and the face
scored 0.850 — a confident detection by any reasonable reading, rejected only
because the default sits at 0.90.

## 6. Why the two models behave differently

The results follow from how each model works.

**Haar's errors are false positives; YuNet's are false negatives.** Haar has no
notion of confidence, so its only defence against a wrong detection is requiring
more overlapping hits. Tightening that also discards marginal true faces, which
is why `minNeighbors` 12 lost 3 real faces. YuNet instead scores every candidate
and lets a threshold decide, which moves the error from "accepts rubbish" to
"rejects anything it is unsure about". Neither is inherently safer; they fail in
opposite directions, and for an attendance system a missed detection is more
visible to the user than a spurious box the recognizer will reject anyway.

**The CNN was not slower, which contradicts the FYP1 assumption.** The project
plan chose Haar partly on the expectation that a neural network would be too
expensive for a Raspberry Pi. On this hardware YuNet at 16.9 ms was faster than
Haar at its best accurate setting, 28.1 ms. Haar can be made as fast by raising
`scaleFactor` to 1.2, but that costs 3 of 30 faces. The reason is that Haar's
sliding-window search repeats over many scales in sequence, while YuNet performs
one pass over a fixed-size input. **This must be re-measured on the Raspberry
Pi before the claim is made in the final report**, since the balance between a
branch-heavy classical search and a convolutional network can differ on ARM.

**Only YuNet produces landmarks, and that affects recognition.** The SFace
recognizer aligns a face using the five landmark points before computing its
embedding. Behind Haar there are no landmarks, so the code falls back to a plain
resized crop. The pairing of SFace with Haar is therefore expected to be weaker
than SFace with YuNet, and the difference is a property of the detector, not of
the recognizer.

## 7. Recognition results, and why they are not yet usable

All four detector–recognizer combinations were evaluated on the same split:

| Detector | Recognizer | Test images | Accuracy | Calibrated threshold | Recognition time |
| --- | --- | --- | --- | --- | --- |
| Haar | LBPH | 8 | 100% | 59.8 | 2.1 ms |
| Haar | SFace | 8 | 100% | 0.910 | 7.8 ms |
| YuNet | LBPH | 8 | 100% | 65.0 | 2.1 ms |
| YuNet | SFace | 8 | 100% | 0.939 | 6.6 ms |

**These figures should not be quoted as an accuracy result.** With two enrolled
identities, a system that guesses would score 50%, and no impostors were held
out, so the false acceptance rate could not be measured at all. What the table
does establish is that the pipeline is correctly wired end to end and that the
recognition stage costs 2–8 ms, which is small beside detection.

A meaningful figure needs roughly ten enrolled people with genuinely independent
images and at least two people held out of the gallery as impostors. The
evaluation script supports this directly, and it records every probe's raw score
so the decision threshold is calibrated from the team's own data rather than
copied from documentation.

## 8. Conclusion and recommendation

On this test set YuNet matched or beat Haar Cascade on every measure: 100%
against 96.7% detection, zero false positives against four at Haar's default,
and 16.9 ms against 28.1 ms at Haar's best accurate configuration. It also
supplies confidence scores and landmarks, the second of which the SFace
recognizer needs. The cost is a 227 KB model file that must be downloaded once.

Against that, Haar Cascade remains the correct baseline for the project. It
requires no download, it is the method the literature review identified, and
tuned to `minNeighbors` 7 it reaches 96.7% with no false positives — close
enough that the comparison is a genuine engineering trade-off rather than a
foregone conclusion.

Three concrete changes follow from these measurements:

1. **Change Haar's default `minNeighbors` from 5 to 7.** It removes every false
   positive at no measured cost.
2. **Change YuNet's default score threshold from 0.90 to 0.70.** The shipped
   default rejected more than half the faces in this set.
3. **Repeat the whole comparison on the Raspberry Pi**, with about ten enrolled
   people and impostors held out. The timings here establish the relative cost
   of the two models; the absolute figures that belong in the final report must
   come from the target hardware.

## 9. How to reproduce

```bash
python3 scripts/benchmark_detectors.py                      # the table in section 4
python3 scripts/benchmark_detectors.py --detectors haar --min-neighbors 7
python3 scripts/benchmark_detectors.py --detectors yunet --score-threshold 0.7
python3 scripts/evaluate_recognition.py --impostors 2       # the table in section 7
```

Each writes a CSV and a markdown table into `data/output/`.
