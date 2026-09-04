"""
Feature engineering: turn raw MediaPipe hand landmarks into a fixed-length,
rotation/translation/scale-invariant feature vector.

Why not just feed in raw (x, y, z) coordinates? Because those depend on
where your hand is in the frame and how close it is to the camera. Two
identical gestures shot at different distances would look like different
points to a model. Instead we compute joint angles (which don't care about
scale or position) plus a few normalized distances. This is the same idea
as the feature scaling / feature engineering labs you've already done in
the course -- we're just picking better features up front instead of
scaling bad ones.

Landmark indices follow MediaPipe Hands' 21-point model:
https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
    0  WRIST
    1-4    THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP
    5-8    INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP
    9-12   MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP
    13-16  RING_MCP, RING_PIP, RING_DIP, RING_TIP
    17-20  PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP
"""

from __future__ import annotations

import numpy as np

WRIST = 0

# Each finger as a chain of joint indices from wrist to tip.
FINGERS = {
    "thumb": [0, 1, 2, 3, 4],
    "index": [0, 5, 6, 7, 8],
    "middle": [0, 9, 10, 11, 12],
    "ring": [0, 13, 14, 15, 16],
    "pinky": [0, 17, 18, 19, 20],
}

FINGERTIPS = {name: chain[-1] for name, chain in FINGERS.items()}

# Reference distance used to normalize scale: wrist -> middle finger MCP.
# Roughly constant regardless of finger pose, so it's a stable proxy for
# "how big the hand looks", i.e. how close it is to the camera.
PALM_REFERENCE = (0, 9)


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex b, formed by points a-b-c, in degrees."""
    v1 = a - b
    v2 = c - b
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    cos_angle = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def build_feature_names() -> list[str]:
    names = []
    for finger, chain in FINGERS.items():
        for i in range(1, len(chain) - 1):
            names.append(f"{finger}_angle_{i}")
    for finger in FINGERS:
        names.append(f"{finger}_tip_dist")
    return names


FEATURE_NAMES = build_feature_names()


def landmarks_to_array(landmarks) -> np.ndarray:
    """
    Accepts either:
      - a MediaPipe NormalizedLandmarkList-like object (has .landmark, each
        with .x, .y, .z), or
      - an iterable of 21 (x, y, z) tuples / lists,
    and returns a (21, 3) numpy array.
    """
    if hasattr(landmarks, "landmark"):
        pts = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
    else:
        pts = list(landmarks)
    arr = np.array(pts, dtype=np.float64)
    if arr.shape != (21, 3):
        raise ValueError(f"Expected 21 (x,y,z) landmarks, got shape {arr.shape}")
    return arr


def extract_features(landmarks) -> np.ndarray:
    """
    landmarks: anything accepted by landmarks_to_array.
    Returns a 1D feature vector of length len(FEATURE_NAMES) (20 by default:
    15 joint angles + 5 normalized fingertip distances).
    """
    coords = landmarks_to_array(landmarks)

    ref_a, ref_b = PALM_REFERENCE
    palm_size = np.linalg.norm(coords[ref_a] - coords[ref_b])
    palm_size = palm_size if palm_size > 1e-6 else 1e-6

    features = []

    # Joint angles: rotation/translation/scale invariant by construction.
    for chain in FINGERS.values():
        for i in range(1, len(chain) - 1):
            a, b, c = coords[chain[i - 1]], coords[chain[i]], coords[chain[i + 1]]
            features.append(_angle(a, b, c))

    # Fingertip distance from wrist, normalized by palm size: captures how
    # extended each finger is, independent of hand distance from camera.
    wrist = coords[WRIST]
    for tip_idx in FINGERTIPS.values():
        dist = np.linalg.norm(coords[tip_idx] - wrist) / palm_size
        features.append(dist)

    return np.array(features, dtype=np.float64)
