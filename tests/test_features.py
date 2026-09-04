"""
Unit tests for feature engineering. Run with:  pytest

The point of testing feature engineering specifically (rather than only
testing "does train.py crash") is that a silent bug here -- e.g. swapped
landmark indices, or a normalization that isn't actually scale-invariant --
would poison every downstream model without ever raising an error. Catching
that requires deliberately constructed inputs, not just running real data
through it.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import FEATURE_NAMES, FINGERS, extract_features, landmarks_to_array  # noqa: E402


def make_straight_hand() -> np.ndarray:
    """
    A fully open hand: every finger's joints lie on a single straight line
    from the wrist, so every joint angle should be ~180 degrees.
    """
    coords = np.zeros((21, 3))
    for chain in FINGERS.values():
        # arbitrary but distinct direction per finger, so fingers don't overlap
        direction = np.array([chain[-1] % 3 - 1, 1.0, 0.0])
        direction = direction / np.linalg.norm(direction)
        for i, idx in enumerate(chain):
            coords[idx] = coords[0] + direction * (i * 0.05)
    return coords


def make_curled_hand() -> np.ndarray:
    """Every finger bent ~90 degrees at each joint after the first."""
    coords = np.zeros((21, 3))
    for chain in FINGERS.values():
        pos = coords[0].copy()
        directions = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([-1.0, 0.0, 0.0])]
        for i, idx in enumerate(chain[1:]):
            d = directions[i % len(directions)]
            pos = pos + d * 0.04
            coords[idx] = pos
    return coords


def test_landmarks_to_array_accepts_plain_list():
    coords = [(float(i), float(i), 0.0) for i in range(21)]
    arr = landmarks_to_array(coords)
    assert arr.shape == (21, 3)


def test_landmarks_to_array_rejects_wrong_length():
    with pytest.raises(ValueError):
        landmarks_to_array([(0.0, 0.0, 0.0)] * 5)


def test_feature_vector_has_expected_length():
    coords = make_straight_hand()
    feats = extract_features(coords)
    assert feats.shape == (len(FEATURE_NAMES),)
    assert len(FEATURE_NAMES) == 20  # 5 fingers * 3 angles + 5 tip distances


def test_open_hand_angles_are_near_straight():
    coords = make_straight_hand()
    feats = extract_features(coords)
    angle_values = feats[: 15]  # first 15 features are joint angles
    assert np.all(angle_values > 150), f"expected near-straight angles, got {angle_values}"


def test_curled_hand_angles_are_smaller_than_open_hand():
    open_feats = extract_features(make_straight_hand())
    curled_feats = extract_features(make_curled_hand())
    assert curled_feats[:15].mean() < open_feats[:15].mean()


def test_features_are_translation_invariant():
    coords = make_straight_hand()
    shifted = coords + np.array([1.0, -2.0, 0.5])
    assert np.allclose(extract_features(coords), extract_features(shifted), atol=1e-8)


def test_features_are_scale_invariant():
    coords = make_straight_hand()
    scaled = coords * 3.0
    # angles are scale-invariant; normalized tip distances are scale-invariant too
    assert np.allclose(extract_features(coords), extract_features(scaled), atol=1e-6)


def test_degenerate_hand_does_not_crash():
    """All landmarks at the exact same point (e.g. a bad detection) shouldn't raise."""
    coords = np.zeros((21, 3))
    feats = extract_features(coords)
    assert np.all(np.isfinite(feats))
