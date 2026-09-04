"""
Run the trained model. Two modes:

1. Batch, on a CSV of raw landmarks (same schema collect_data.py writes):
     python src/predict.py --input-csv data/raw/sample_landmarks.csv

2. Live, from your webcam (sanity-check the model before touching iOS/web):
     python src/predict.py --live
   (needs: pip install opencv-python mediapipe)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import extract_features  # noqa: E402

LANDMARK_COLS = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]


def load_model(path: Path):
    bundle = joblib.load(path)
    return bundle["pipeline"], bundle["feature_names"], bundle["classes"]


def predict_from_landmarks(pipeline, landmarks) -> tuple[str, dict[str, float]]:
    features = extract_features(landmarks).reshape(1, -1)
    label = pipeline.predict(features)[0]
    probs = {}
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(features)[0]
        probs = dict(zip(pipeline.classes_, proba))
    return label, probs


def run_batch(pipeline, feature_names, input_csv: Path):
    df = pd.read_csv(input_csv)
    raw = df[LANDMARK_COLS].to_numpy().reshape(len(df), 21, 3)
    X = np.array([extract_features(sample) for sample in raw])
    preds = pipeline.predict(X)
    df["predicted_label"] = preds
    if "label" in df.columns:
        accuracy = (df["predicted_label"] == df["label"]).mean()
        print(f"Accuracy on {input_csv}: {accuracy:.3f}")
    print(df[["label", "predicted_label"]].head(20) if "label" in df.columns else df[["predicted_label"]].head(20))


def run_live(pipeline):
    try:
        import cv2
        import mediapipe as mp
    except ImportError as e:
        raise SystemExit("Run: pip install opencv-python mediapipe") from e

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            label_text = "no hand"
            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                label, probs = predict_from_landmarks(pipeline, hand_landmarks)
                conf = probs.get(label, 0.0)
                label_text = f"{label} ({conf:.2f})"

            cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.imshow("predict - q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--model", default=str(root / "models" / "gesture_classifier.joblib"))
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()

    pipeline, feature_names, classes = load_model(Path(args.model))
    print(f"Loaded model. Classes: {classes}")

    if args.live:
        run_live(pipeline)
    elif args.input_csv:
        run_batch(pipeline, feature_names, Path(args.input_csv))
    else:
        parser.error("Pass --input-csv <file> or --live")


if __name__ == "__main__":
    main()
