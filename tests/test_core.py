from __future__ import annotations

import numpy as np
from PIL import Image

from eurosat_landcover_vision.dataset import CLASS_NAMES
from eurosat_landcover_vision.inference import predict_image
from eurosat_landcover_vision.model import (
    LandCoverCNN,
    class_stratified_bootstrap,
    metrics_from_scores,
)
from eurosat_landcover_vision.pipeline import _split_data


def test_duplicate_groups_do_not_cross_splits_and_split_is_reproducible() -> None:
    labels = np.repeat(np.arange(len(CLASS_NAMES)), 50)
    groups = np.asarray([f"sample-{index}" for index in range(len(labels))])
    groups[1] = groups[0]
    groups[201] = groups[200]

    first = _split_data(labels, groups, seed=42)
    second = _split_data(labels, groups, seed=42)

    assert all(np.array_equal(left, right) for left, right in zip(first, second, strict=True))
    split_groups = [set(groups[indices]) for indices in first]
    assert all(
        split_groups[left].isdisjoint(split_groups[right])
        for left in range(3)
        for right in range(left + 1, 3)
    )
    assert sum(map(len, first)) == len(labels)


def test_metrics_and_class_stratified_bootstrap_are_finite_and_seeded() -> None:
    truth = np.repeat(np.arange(len(CLASS_NAMES)), 2)
    scores = np.zeros((len(truth), len(CLASS_NAMES)), dtype=np.float64)
    scores[np.arange(len(truth)), truth] = 1.0
    scores[0] = 0.0
    scores[0, 1] = 1.0

    metrics, report, confusion = metrics_from_scores(truth, scores, CLASS_NAMES)
    first = class_stratified_bootstrap(truth, scores, seed=7, replicates=50)
    second = class_stratified_bootstrap(truth, scores, seed=7, replicates=50)

    assert metrics["accuracy"] == 0.95
    assert len(report) == len(CLASS_NAMES)
    assert confusion.sum() == len(truth)
    assert first == second
    assert 0.0 <= first["accuracy"]["ci_low"] <= first["accuracy"]["ci_high"] <= 1.0


def test_model_image_inference_returns_ranked_class_scores() -> None:
    model = LandCoverCNN()
    model.eval()
    mean = [0.5, 0.5, 0.5]
    std = [0.25, 0.25, 0.25]

    result = predict_image(model, Image.new("RGB", (80, 40), "green"), mean, std, top_k=3)

    assert len(result) == 3
    assert all(row["class_name"] in CLASS_NAMES for row in result)
    assert all(0.0 <= row["confidence"] <= 1.0 for row in result)
    assert [row["confidence"] for row in result] == sorted(
        (row["confidence"] for row in result), reverse=True
    )
