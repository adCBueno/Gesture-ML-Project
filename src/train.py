"""
Train a hand-gesture classifier from labeled landmark data.

This mirrors a normal supervised-learning workflow, just applied to your own
data instead of a course dataset:
    1. load raw landmarks -> engineer features (features.py)
    2. split into train/test, GROUPED BY SESSION (not by row) to avoid leakage
    3. try a couple of models with cross-validation, not a single fixed split
    4. evaluate the winner on the held-out test set (classification report +
       confusion matrix) -- accuracy alone hides class-specific problems
    5. persist the trained pipeline (scaler + model) for reuse

Usage:
    python src/train.py                          # uses data/raw/sample_landmarks.csv
    python src/train.py --data data/raw/landmarks.csv --out models/gesture_classifier.joblib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURE_NAMES, extract_features  # noqa: E402

LANDMARK_COLS = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]


def load_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    missing = set(LANDMARK_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing landmark columns: {sorted(missing)[:5]}...")

    raw = df[LANDMARK_COLS].to_numpy().reshape(len(df), 21, 3)
    X = np.array([extract_features(sample) for sample in raw])
    y = df["label"].to_numpy()
    groups = df["session_id"].to_numpy()
    return X, y, groups


def stratified_group_split(
    y: np.ndarray, groups: np.ndarray, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Group-aware AND class-aware split: whole sessions go entirely to train or
    test (no leakage), while keeping every class represented in both splits.

    Plain GroupShuffleSplit picks sessions at random without looking at their
    labels -- with only a handful of sessions per class, that can easily
    leave a class with zero test sessions (you'll see exactly this if you
    swap it back in and re-run a few times). We split sessions class-by-class
    instead.
    """
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for label in sorted(set(y)):
        label_groups = np.array(sorted(set(groups[y == label])))
        rng.shuffle(label_groups)
        n_test = max(1, round(len(label_groups) * test_size)) if len(label_groups) > 1 else 0
        test_groups = set(label_groups[:n_test])
        for idx in np.where(y == label)[0]:
            (test_idx if groups[idx] in test_groups else train_idx).append(idx)
    return np.array(train_idx), np.array(test_idx)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data", default=str(root / "data" / "raw" / "sample_landmarks.csv"))
    parser.add_argument("--out", default=str(root / "models" / "gesture_classifier.joblib"))
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_path = Path(args.data)
    print(f"Loading {data_path} ...")
    X, y, groups = load_dataset(data_path)
    print(f"{len(X)} samples, {len(FEATURE_NAMES)} features, classes: {sorted(set(y))}")
    n_groups = len(set(groups))
    print(f"{n_groups} recording sessions")

    # Group-aware AND class-aware split: whole sessions go to either train or
    # test (never both), and every class stays represented in each split.
    train_idx, test_idx = stratified_group_split(y, groups, args.test_size, args.seed)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]

    candidates = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]),
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),  # not required for trees, kept for a uniform pipeline
            ("clf", RandomForestClassifier(n_estimators=200, random_state=args.seed)),
        ]),
    }

    n_splits = max(2, min(5, len(set(groups_train))))
    cv = GroupKFold(n_splits=n_splits)

    print(f"\nCross-validation ({n_splits}-fold, grouped by session):")
    best_name, best_score, best_pipeline = None, -np.inf, None
    for name, pipeline in candidates.items():
        scores = cross_val_score(pipeline, X_train, y_train, groups=groups_train, cv=cv)
        print(f"  {name:20s} accuracy = {scores.mean():.3f} (+/- {scores.std():.3f})")
        if scores.mean() > best_score:
            best_name, best_score, best_pipeline = name, scores.mean(), pipeline

    print(f"\nBest model by CV: {best_name} ({best_score:.3f}) -- refitting on full train set")
    best_pipeline.fit(X_train, y_train)

    y_pred = best_pipeline.predict(X_test)
    print("\nHeld-out test set report:")
    print(classification_report(y_test, y_pred))
    labels = sorted(set(y))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("Confusion matrix (rows=true, cols=predicted):")
    print("labels:", labels)
    print(cm)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": best_pipeline, "feature_names": FEATURE_NAMES, "classes": labels, "model_name": best_name},
        out_path,
    )
    print(f"\nSaved trained pipeline to {out_path}")


if __name__ == "__main__":
    main()
