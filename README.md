# Gesture Classifier — from webcam to on-device iOS model

An end-to-end supervised learning project: collect your own labeled data,
engineer features, train and evaluate a classifier, and export it to run
on-device on iOS. It applies the same concepts from Course 1 of the ML
specialization (feature scaling, feature engineering, gradient descent,
logistic regression, the sigmoid function, decision boundaries) to a real
pipeline instead of a fixed course dataset — and it's built the way this
kind of project actually gets built at work, not as a single notebook.

It's designed to plug into the hand-tracking web app you already built
(MediaPipe Hands + Web Audio): instead of hardcoded rules on landmark
angles/distances to recognize a chord shape, a model *learns* the mapping
from data you record.

## Why this structure (and why it's worth having on your CV)

Real ML work is rarely "train a model in a notebook and you're done." This
repo is organized to reflect the actual steps and the actual failure modes:

- **Data collection is its own script**, and it saves *raw* landmarks, not
  pre-computed features. Feature engineering can change later without
  re-recording anything — you never want your expensive-to-collect data
  locked to one version of your feature code.
- **Feature engineering is isolated and unit-tested** (`tests/test_features.py`).
  Joint angles and normalized distances were chosen deliberately over raw
  (x, y, z) coordinates, because angles don't change when your hand moves
  closer to the camera or shifts position in frame — raw coordinates would.
- **The train/test split is grouped by recording session, and stratified by
  class.** Splitting plain rows at random leaks information (consecutive
  frames from the same take are nearly identical), and splitting sessions
  without checking class balance can accidentally drop a whole class from
  the test set. Both are common, real mistakes.
- **Model selection uses cross-validation, not one lucky split**, and the
  final report includes a confusion matrix, not just accuracy — accuracy
  alone hides whether the model confuses two specific gestures.
- **Deployment is a separate, explicit step** (Core ML export), including
  the dependency-pinning problem you'll actually hit (coremltools' sklearn
  converter caps the scikit-learn version it accepts).

That's the difference between "I did the Coursera labs" and "I built and
shipped an ML pipeline" on a CV — this project is set up to let you
honestly say the second one.

## Project layout

```
data/
  make_sample_data.py   # generates synthetic data so you can run everything today
  raw/sample_landmarks.csv
src/
  features.py            # raw landmarks -> engineered feature vector
  collect_data.py         # record your OWN labeled data from a webcam
  train.py                 # split, cross-validate, evaluate, save the model
  predict.py                # batch predictions on a CSV, or a live webcam demo
  export_coreml.py           # convert the trained model to .mlmodel for iOS
tests/
  test_features.py           # unit tests for the feature engineering
models/                       # trained model + exported .mlmodel land here
```

## Quickstart (works right now, no camera needed)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest                 # run the unit tests
python src/train.py     # trains on the included synthetic dataset
python src/predict.py --input-csv data/raw/sample_landmarks.csv
```

The synthetic dataset (`data/make_sample_data.py`) is a simple
forward-kinematics hand model with noise — good enough to validate every
step of the pipeline, but it's not real data, so don't read much into the
(very high) accuracy it produces. The real project starts at the next step.

## Using your own data

```bash
pip install opencv-python mediapipe        # camera dependencies

python src/collect_data.py --label open_hand
python src/collect_data.py --label fist
python src/collect_data.py --label peace
# ...record 4-6 short sessions per gesture, moving your hand slightly
# between sessions (distance, angle, lighting) so the model doesn't
# overfit to one exact position

python src/train.py --data data/raw/landmarks.csv
python src/predict.py --live       # sanity-check it against your webcam
```

If accuracy on real data is much lower than on the synthetic set, that's
normal and useful information, not a bug: check the confusion matrix to see
which gestures actually get confused, and record a few more sessions for
those classes specifically.

## Exporting for iOS

```bash
python -m venv .venv-export && source .venv-export/bin/activate
pip install "scikit-learn==1.5.1" coremltools joblib
python src/export_coreml.py
```

This produces `models/GestureClassifier.mlmodel`. Drag it into Xcode; feed
it the same 20 engineered features in the same order (`src/features.py` has
the exact list) computed from landmarks you get via Vision's hand-pose
request or a ported version of the MediaPipe pipeline.

## Natural next steps

- Add more gesture classes (map them to actual chords for the music app).
- Try a small neural net instead of logistic regression/random forest and
  compare — same evaluation harness, just swap the model in `train.py`.
- Log experiments (which features, which model, what accuracy) instead of
  overwriting `models/gesture_classifier.joblib` each time — a simple CSV
  or a tool like MLflow, if you want to go further.
- Push this to GitHub as a portfolio project: the README, the tests, and
  the two-environment deployment step are exactly what a reviewer looks for
  beyond "the model works."
