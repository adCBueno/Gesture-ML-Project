"""
Generates a SYNTHETIC raw-landmark dataset (data/raw/sample_landmarks.csv) so
you can run the whole pipeline (train.py, tests, export_coreml.py) before you
have real webcam data. It writes the same schema collect_data.py produces, so
swapping this for real data later is a no-op for the rest of the pipeline.

This is not meant to be biomechanically perfect -- it's a simple forward-
kinematics model (each finger = a chain of segments that bend at each joint)
parameterized by a "curl" angle per finger, per gesture class, with noise and
per-session rotation/scale/position jitter added so sessions don't look
identical (like real recordings wouldn't).

Run:
    python data/make_sample_data.py
"""

from __future__ import annotations

import csv
import math
import random
import time
from pathlib import Path

random.seed(42)

OUT_PATH = Path(__file__).resolve().parent / "raw" / "sample_landmarks.csv"

# Per-finger base direction (degrees from "straight up") and bone lengths
# (wrist->j1, j1->j2, j2->j3, j3->tip), in normalized [0,1] frame units.
FINGER_GEOMETRY = {
    "thumb":  {"base_angle": -55, "lengths": [0.055, 0.045, 0.040, 0.035]},
    "index":  {"base_angle": -18, "lengths": [0.095, 0.050, 0.032, 0.026]},
    "middle": {"base_angle":  -4, "lengths": [0.100, 0.055, 0.035, 0.030]},
    "ring":   {"base_angle":  12, "lengths": [0.093, 0.050, 0.033, 0.028]},
    "pinky":  {"base_angle":  28, "lengths": [0.078, 0.038, 0.026, 0.022]},
}
FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]

EXTENDED_CURL = 8      # degrees of bend per joint when finger is "open"
CURLED_CURL = 58       # degrees of bend per joint when finger is "closed"

# gesture class -> which fingers are extended
GESTURES = {
    "open_hand": {"thumb": True, "index": True, "middle": True, "ring": True, "pinky": True},
    "fist":      {"thumb": False, "index": False, "middle": False, "ring": False, "pinky": False},
    "peace":     {"thumb": False, "index": True, "middle": True, "ring": False, "pinky": False},
    "point":     {"thumb": False, "index": True, "middle": False, "ring": False, "pinky": False},
}

N_SESSIONS_PER_CLASS = 5
N_SAMPLES_PER_SESSION = 40
NOISE_CURL_DEG = 5.0
NOISE_ANGLE_DEG = 3.0
NOISE_LENGTH_PCT = 0.05
NOISE_Z = 0.01


def _dir_vector(angle_deg: float) -> tuple[float, float]:
    theta = math.radians(angle_deg)
    return math.sin(theta), -math.cos(theta)  # 0 deg = straight up


def make_hand(gesture: str, wrist_xy, session_rotation_deg, session_scale) -> list[float]:
    """Returns flattened [x0,y0,z0, x1,y1,z1, ..., x20,y20,z20]."""
    points = [None] * 21
    points[0] = (wrist_xy[0], wrist_xy[1], 0.0)

    finger_extended = GESTURES[gesture]
    idx_map = {
        "thumb": [1, 2, 3, 4],
        "index": [5, 6, 7, 8],
        "middle": [9, 10, 11, 12],
        "ring": [13, 14, 15, 16],
        "pinky": [17, 18, 19, 20],
    }

    for finger in FINGER_ORDER:
        geom = FINGER_GEOMETRY[finger]
        curl_base = EXTENDED_CURL if finger_extended[finger] else CURLED_CURL
        base_angle = geom["base_angle"] + session_rotation_deg + random.gauss(0, NOISE_ANGLE_DEG)

        direction = base_angle
        cur_point = wrist_xy
        joint_ids = idx_map[finger]

        for seg_i, length in enumerate(geom["lengths"]):
            length = length * session_scale * (1 + random.gauss(0, NOISE_LENGTH_PCT))
            if seg_i > 0:
                curl = curl_base + random.gauss(0, NOISE_CURL_DEG)
                direction += curl
            dx, dy = _dir_vector(direction)
            new_point = (cur_point[0] + dx * length, cur_point[1] + dy * length)
            z = random.gauss(0, NOISE_Z) - (0.01 if finger_extended[finger] else 0.0)
            points[joint_ids[seg_i]] = (new_point[0], new_point[1], z)
            cur_point = new_point

    flat = []
    for p in points:
        flat.extend(p)
    return flat


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = ["session_id", "label", "timestamp"] + [
        f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")
    ]

    rows = []
    t0 = time.time()
    for gesture in GESTURES:
        for s in range(N_SESSIONS_PER_CLASS):
            session_id = f"{gesture}_synthsession{s}"
            # Each "session" simulates a slightly different hand position,
            # rotation and distance from camera -- like a real recording run.
            wrist_xy = (0.5 + random.uniform(-0.1, 0.1), 0.85 + random.uniform(-0.1, 0.05))
            session_rotation = random.uniform(-12, 12)
            session_scale = random.uniform(0.85, 1.15)

            for f in range(N_SAMPLES_PER_SESSION):
                flat = make_hand(gesture, wrist_xy, session_rotation, session_scale)
                rows.append([session_id, gesture, t0 + f * 0.03] + flat)

    with OUT_PATH.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Wrote {len(rows)} synthetic samples ({len(GESTURES)} classes) to {OUT_PATH}")


if __name__ == "__main__":
    main()
