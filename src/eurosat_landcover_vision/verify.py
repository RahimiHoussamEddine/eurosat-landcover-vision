"""Check committed benchmark artifacts and run a CPU inference smoke check."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image

from eurosat_landcover_vision.dataset import CLASS_NAMES, EXPECTED_IMAGE_COUNT
from eurosat_landcover_vision.inference import load_classifier, predict_image

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts" / "v1"
REPORT = ROOT / "reports" / "RESULTS.md"
FIGURES = ROOT / "reports" / "figures"


def verify() -> dict[str, object]:
    required = [
        ARTIFACTS / "analysis_summary.json",
        ARTIFACTS / "data_audit.json",
        ARTIFACTS / "split_manifest.json",
        ARTIFACTS / "landcover_cnn_state.pt",
        ARTIFACTS / "training_metadata.json",
        ARTIFACTS / "model_comparison.csv",
        ARTIFACTS / "per_class_metrics.csv",
        ARTIFACTS / "confusion_matrix.csv",
        ARTIFACTS / "robustness.csv",
        ARTIFACTS / "training_history.csv",
        REPORT,
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required benchmark outputs are missing: {missing}")

    summary = json.loads((ARTIFACTS / "analysis_summary.json").read_text())
    audit = summary["data_audit"]
    split = summary["split"]
    if audit["image_count"] != EXPECTED_IMAGE_COUNT:
        raise ValueError("The artifact manifest does not record all 27,000 source images")
    if len(summary["class_names"]) != len(CLASS_NAMES):
        raise ValueError("Expected exactly ten classes")
    if (
        sum(split[name] for name in ("train_rows", "validation_rows", "test_rows"))
        != EXPECTED_IMAGE_COUNT
    ):
        raise ValueError("Train, validation, and test row counts do not sum to the source size")
    if summary["test_score_shape"] != [split["test_rows"], len(CLASS_NAMES)]:
        raise ValueError("Test score dimensions do not match the split manifest")

    comparison = pd.read_csv(ARTIFACTS / "model_comparison.csv")
    if comparison["model"].tolist() != ["Residual CNN", "HOG + linear SVM"]:
        raise ValueError("Expected residual CNN and HOG + linear SVM comparison rows")
    per_class = pd.read_csv(ARTIFACTS / "per_class_metrics.csv")
    if per_class["class_name"].tolist() != CLASS_NAMES:
        raise ValueError("Per-class metrics do not match the ten expected classes")
    matrix = pd.read_csv(ARTIFACTS / "confusion_matrix.csv", index_col=0)
    if matrix.shape != (len(CLASS_NAMES), len(CLASS_NAMES)):
        raise ValueError("Confusion matrix has the wrong shape")
    if int(matrix.to_numpy().sum()) != split["test_rows"]:
        raise ValueError("Confusion matrix total differs from the held-out test size")
    if not all(
        (FIGURES / name).is_file()
        for name in ("model_comparison.png", "confusion_matrix.png", "robustness.png")
    ):
        raise FileNotFoundError("One or more report figures are missing")

    model = load_classifier(ARTIFACTS / "landcover_cnn_state.pt")
    predictions = predict_image(
        model,
        Image.new("RGB", (64, 64), color=(96, 128, 160)),
        summary["normalization_mean"],
        summary["normalization_std"],
        top_k=3,
    )
    if len(predictions) != 3 or any(row["class_name"] not in CLASS_NAMES for row in predictions):
        raise ValueError("Model CPU inference did not return three valid class predictions")
    if any(not 0.0 <= row["confidence"] <= 1.0 for row in predictions):
        raise ValueError("Inference scores fall outside [0, 1]")

    return {
        "status": "ok",
        "source_images": audit["image_count"],
        "classes": len(CLASS_NAMES),
        "test_rows": split["test_rows"],
        "model_comparison_rows": len(comparison),
        "inference_top_k": len(predictions),
    }


def main() -> None:
    result = verify()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
