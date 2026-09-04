"""
Data collection: record labeled hand-landmark samples from your webcam.

Run this on your own machine (needs a camera) -- it won't run in a cloud
sandbox. Install the extras first:

    pip install opencv-python mediapipe

Usage:
    python src/collect_data.py --label open_hand
    python src/collect_data.py --label fist
    python src/collect_data.py --label chord_c   # name your own gesture classes

Controls while the window is focused:
    SPACE  - toggle recording on/off
    q      - quit

Design notes (why it works this way):
  - We save RAW landmarks (21 x,y,z), not engineered features. Feature
    engineering (features.py) can then change/improve later without
    re-collecting data -- store raw, derive features downstream. This is
    standard practice: keep the expensive-to-collect data as close to raw
    as possible.
  - Every run gets its own session_id (timestamp). When you later split
    into train/test, split BY SESSION, not by row. Consecutive frames from
    one recording session are highly correlated (same lighting, same exact
    hand position drifting slightly) -- if you split by row, your "test"
    set leaks information from the same session and your accuracy will
    look better than it really is. This is one of the most common real
    mistakes people make with sequential/sensor data.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

LANDMARK_COLS = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
CSV_HEADER = ["session_id", "label", "timestamp"] + LANDMARK_COLS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="Gesture class name for this session")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[1] / "data" / "raw" / "landmarks.csv"),
        help="CSV file to append to",
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    args = parser.parse_args()

    try:
        import cv2
        import mediapipe as mp
    except ImportError as e:
        raise SystemExit(
            "Missing camera dependencies. Run: pip install opencv-python mediapipe"
        ) from e

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()

    session_id = f"{args.label}_{int(time.time())}"
    recording = False
    n_saved = 0

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    with out_path.open("a", newline="") as f, mp_hands.Hands(
        max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7
    ) as hands:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(CSV_HEADER)

        print(f"Session '{session_id}' -- label '{args.label}'. SPACE to record, q to quit.")

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if recording:
                    row = [session_id, args.label, time.time()]
                    for lm in hand_landmarks.landmark:
                        row += [lm.x, lm.y, lm.z]
                    writer.writerow(row)
                    n_saved += 1

            status = f"REC ({n_saved})" if recording else "paused"
            cv2.putText(
                frame, f"label={args.label} [{status}]", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if recording else (0, 200, 0), 2,
            )
            cv2.imshow("collect_data - SPACE=record, q=quit", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                recording = not recording
            elif key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {n_saved} samples to {out_path} (session_id={session_id})")


if __name__ == "__main__":
    main()
