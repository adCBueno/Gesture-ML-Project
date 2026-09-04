"""
Export the trained scikit-learn pipeline to a Core ML (.mlmodel) file so it
can run on-device in an iOS app -- no server round-trip, works offline.

IMPORTANT dependency note (a real deployment gotcha, not a mistake in your
setup): coremltools' scikit-learn converter only supports scikit-learn
<= 1.5.1. If you train with a newer scikit-learn, this script will refuse to
run. This is normal in ML deployment -- the environment that trains a model
and the environment that converts/serves it often need different pinned
dependency versions. The clean way to handle it is two virtual environments:

    python -m venv .venv-train        # whatever scikit-learn you like
    source .venv-train/bin/activate
    pip install -r requirements.txt
    python src/train.py

    python -m venv .venv-export
    source .venv-export/bin/activate
    pip install "scikit-learn==1.5.1" coremltools joblib
    python src/export_coreml.py

Usage:
    python src/export_coreml.py
    python src/export_coreml.py --model models/gesture_classifier.joblib --out models/GestureClassifier.mlmodel
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
from sklearn import __version__ as sklearn_version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--model", default=str(root / "models" / "gesture_classifier.joblib"))
    parser.add_argument("--out", default=str(root / "models" / "GestureClassifier.mlmodel"))
    args = parser.parse_args()

    try:
        import coremltools as ct
    except ImportError as e:
        raise SystemExit("Missing dependency. Run: pip install coremltools") from e

    if not hasattr(ct.converters, "sklearn") or not hasattr(ct.converters.sklearn, "convert"):
        raise SystemExit(
            "coremltools disabled its scikit-learn converter for this environment "
            f"(you have scikit-learn {sklearn_version}). It requires scikit-learn <= 1.5.1. "
            "Use a separate venv for this export step -- see the docstring at the top of "
            "this file."
        )

    bundle = joblib.load(args.model)
    pipeline = bundle["pipeline"]
    feature_names = bundle["feature_names"]

    print(f"Converting {args.model} (trained with '{bundle['model_name']}') to Core ML ...")
    coreml_model = ct.converters.sklearn.convert(
        pipeline,
        input_features=feature_names,
        output_feature_names=("gesture", "gesture_probabilities"),
    )

    coreml_model.author = "Adri"
    coreml_model.short_description = (
        "Hand-gesture classifier trained on MediaPipe hand-landmark features "
        "(joint angles + normalized fingertip distances)."
    )
    for name in feature_names:
        try:
            coreml_model.input_description[name] = "See src/features.py for how this is computed."
        except AttributeError:
            pass  # feature naming can vary slightly by coremltools version; not critical

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    coreml_model.save(str(out_path))
    print(f"Saved {out_path}")
    print(
        "\nDrag this .mlmodel into your Xcode project. In Swift, feed it the same "
        f"{len(feature_names)} features (see src/features.py for how to compute them "
        "from MediaPipe/Vision hand-pose landmarks -- the ordering must match exactly)."
    )


if __name__ == "__main__":
    main()
