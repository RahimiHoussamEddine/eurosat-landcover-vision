"""Compact residual CNN and CPU training/evaluation utilities."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as functional
from skimage.feature import hog
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Dataset

LOGGER = logging.getLogger(__name__)
MODEL_CONFIG_ID = "landcover-cnn-rgb-16-32-64-one-block-v1"


class ResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.norm1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(
            output_channels, output_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm2 = nn.BatchNorm2d(output_channels)
        self.activation = nn.GELU()
        self.skip = (
            nn.Identity()
            if input_channels == output_channels and stride == 1
            else nn.Sequential(
                nn.Conv2d(
                    input_channels, output_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(output_channels),
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.skip(inputs)
        values = self.activation(self.norm1(self.conv1(inputs)))
        values = self.norm2(self.conv2(values))
        return self.activation(values + residual)


class LandCoverCNN(nn.Module):
    """A small residual image classifier trained from scratch on 64x64 patches."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
        )
        self.stage1 = ResidualBlock(16, 16)
        self.stage2 = ResidualBlock(16, 32, stride=2)
        self.stage3 = ResidualBlock(32, 64, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.25)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = self.stem(inputs)
        values = self.stage1(values)
        values = self.stage2(values)
        values = self.stage3(values)
        values = self.pool(values).flatten(1)
        return self.classifier(self.dropout(values))


class TensorImageDataset(Dataset):
    def __init__(
        self,
        images: torch.Tensor,
        labels: np.ndarray,
        indices: np.ndarray,
        mean: torch.Tensor,
        std: torch.Tensor,
        *,
        augment: bool,
    ) -> None:
        self.images = images
        self.labels = labels
        self.indices = indices
        self.mean = mean.reshape(3, 1, 1)
        self.std = std.reshape(3, 1, 1)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = int(self.indices[position])
        image = self.images[index].to(dtype=torch.float32).div_(255.0)
        if self.augment:
            if torch.rand(()) < 0.5:
                image = image.flip(-1)
            if torch.rand(()) < 0.5:
                image = image.flip(-2)
            image = torch.rot90(image, int(torch.randint(0, 4, (1,)).item()), dims=(-2, -1))
        image = (image - self.mean) / self.std
        return image, torch.tensor(int(self.labels[index]), dtype=torch.long)


def _predict_tensor(
    model: LandCoverCNN,
    images: torch.Tensor,
    labels: np.ndarray,
    indices: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    batch_size: int,
    corruption: str = "clean",
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits = []
    loader = DataLoader(
        TensorImageDataset(images, labels, indices, mean, std, augment=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    with torch.inference_mode():
        for batch, _ in loader:
            raw = (batch * std.reshape(1, 3, 1, 1) + mean.reshape(1, 3, 1, 1)).clamp(0, 1)
            if corruption == "brightness_plus_20_percent":
                raw = (raw * 1.2).clamp(0, 1)
            elif corruption == "mean_blur_3x3":
                kernel = (
                    torch.tensor(
                        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
                        dtype=raw.dtype,
                    )
                    / 16.0
                )
                weight = kernel.reshape(1, 1, 3, 3).repeat(3, 1, 1, 1)
                raw = functional.conv2d(
                    functional.pad(raw, (1, 1, 1, 1), mode="replicate"),
                    weight,
                    groups=3,
                )
            elif corruption != "clean":
                raise ValueError(f"Unknown corruption: {corruption}")
            normalized = (raw - mean.reshape(1, 3, 1, 1)) / std.reshape(1, 3, 1, 1)
            logits.append(model(normalized).cpu())
    predictions = torch.cat(logits).numpy()
    truth = labels[indices]
    return truth, predictions


def train_cnn(
    images: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    output_path: Path,
    *,
    seed: int = 20261102,
    epochs: int = 5,
    batch_size: int = 256,
    patience: int = 2,
) -> tuple[LandCoverCNN, dict[str, Any], pd.DataFrame]:
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    tensor_images = torch.from_numpy(images).permute(0, 3, 1, 2)
    train_float = tensor_images[train_indices].to(dtype=torch.float32) / 255.0
    mean = train_float.mean(dim=(0, 2, 3))
    std = train_float.std(dim=(0, 2, 3), unbiased=False).clamp_min(1e-6)
    del train_float
    train_dataset = TensorImageDataset(
        tensor_images, labels, train_indices, mean, std, augment=True
    )
    validation_dataset = TensorImageDataset(
        tensor_images, labels, validation_indices, mean, std, augment=False
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = LandCoverCNN(num_classes=10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1, min_lr=0.00001
    )
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_score = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch), target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

        model.eval()
        validation_truth = []
        validation_predictions = []
        with torch.inference_mode():
            for batch, target in validation_loader:
                scores = model(batch).argmax(dim=1)
                validation_truth.extend(target.tolist())
                validation_predictions.extend(scores.tolist())
        val_macro_f1 = float(f1_score(validation_truth, validation_predictions, average="macro"))
        val_accuracy = float(accuracy_score(validation_truth, validation_predictions))
        scheduler.step(val_macro_f1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_macro_f1": val_macro_f1,
                "validation_accuracy": val_accuracy,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        LOGGER.info(
            "epoch %d/%d train_loss=%.4f validation_macro_f1=%.4f validation_accuracy=%.4f",
            epoch,
            epochs,
            history[-1]["train_loss"],
            val_macro_f1,
            val_accuracy,
        )
        if val_macro_f1 > best_score + 0.0001:
            best_score = val_macro_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    metadata = {
        "model_config_id": MODEL_CONFIG_ID,
        "architecture": "LandCoverCNN small residual CNN trained from scratch",
        "seed": seed,
        "requested_epochs": epochs,
        "completed_epochs": len(history),
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_score,
        "batch_size": batch_size,
        "patience": patience,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "label_smoothing": 0.05,
        "augmentation": ["horizontal flip", "vertical flip", "90-degree rotations"],
        "normalization_mean": [float(value) for value in mean],
        "normalization_std": [float(value) for value in std],
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    (output_path.parent / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return model, metadata, pd.DataFrame(history)


def hog_features(images: np.ndarray) -> np.ndarray:
    features = [
        hog(
            image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            channel_axis=-1,
            feature_vector=True,
        )
        for image in images
    ]
    return np.asarray(features, dtype=np.float32)


def fit_hog_baseline(
    images: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    *,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    train_features = hog_features(images[train_indices])
    evaluation_features = hog_features(images[evaluation_indices])
    classifier = LinearSVC(C=1.0, dual="auto", max_iter=8_000, random_state=seed)
    classifier.fit(train_features, labels[train_indices])
    scores = classifier.decision_function(evaluation_features)
    return (
        {
            "estimator": "LinearSVC",
            "C": 1.0,
            "max_iter": 8_000,
            "random_state": seed,
            "feature_extractor": ("HOG, 9 orientations, 8x8 pixels per cell, 2x2 cells per block"),
            "train_features_shape": list(train_features.shape),
            "evaluation_features_shape": list(evaluation_features.shape),
        },
        scores,
        labels[evaluation_indices],
    )


def metrics_from_scores(
    truth: np.ndarray,
    scores: np.ndarray,
    class_names: list[str],
    *,
    probabilities_usable: bool = True,
) -> tuple[dict[str, float | None], pd.DataFrame, np.ndarray]:
    shifted = scores - scores.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    one_hot = np.eye(len(class_names), dtype=np.float32)[truth]
    confidences = probabilities.max(axis=1)
    correct = predictions == truth
    calibration_error = 0.0
    for lower in np.linspace(0, 1, 11)[:-1]:
        upper = lower + 0.1
        mask = (confidences >= lower) & (confidences < upper if upper < 1 else confidences <= upper)
        if mask.any():
            calibration_error += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidences[mask].mean())
            )
    report = []
    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for true, predicted in zip(truth, predictions, strict=True):
        confusion[int(true), int(predicted)] += 1
    for class_id, class_name in enumerate(class_names):
        true_positive = int(confusion[class_id, class_id])
        false_positive = int(confusion[:, class_id].sum() - true_positive)
        false_negative = int(confusion[class_id, :].sum() - true_positive)
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report.append(
            {
                "class_name": class_name,
                "support": int(confusion[class_id, :].sum()),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    metric: dict[str, float | None] = {
        "accuracy": float(correct.mean()),
        "balanced_accuracy": float(np.mean(np.diag(confusion) / confusion.sum(axis=1))),
        "macro_f1": float(np.mean([row["f1"] for row in report])),
        "top3_accuracy": float(
            np.mean(np.any(np.argsort(-scores, axis=1)[:, :3] == truth[:, None], axis=1))
        ),
        "multiclass_brier": (
            float(np.mean(np.square(probabilities - one_hot).sum(axis=1)))
            if probabilities_usable
            else None
        ),
        "expected_calibration_error_10_bin": calibration_error if probabilities_usable else None,
    }
    return metric, pd.DataFrame(report), confusion


def class_stratified_bootstrap(
    truth: np.ndarray,
    scores: np.ndarray,
    *,
    seed: int,
    replicates: int = 500,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    predicted = scores.argmax(axis=1)
    by_class = [np.flatnonzero(truth == class_id) for class_id in range(int(truth.max()) + 1)]
    accuracy = np.empty(replicates, dtype=np.float64)
    macro_f1 = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sample = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in by_class]
        )
        accuracy[replicate] = np.mean(predicted[sample] == truth[sample])
        macro_f1[replicate] = f1_score(
            truth[sample], predicted[sample], average="macro", zero_division=0
        )
    return {
        "accuracy": {
            "ci_low": float(np.quantile(accuracy, 0.025)),
            "ci_high": float(np.quantile(accuracy, 0.975)),
        },
        "macro_f1": {
            "ci_low": float(np.quantile(macro_f1, 0.025)),
            "ci_high": float(np.quantile(macro_f1, 0.975)),
        },
        "bootstrap_replicates": replicates,
        "seed": seed,
    }


def evaluate_cnn(
    model: LandCoverCNN,
    images: np.ndarray,
    labels: np.ndarray,
    evaluation_indices: np.ndarray,
    class_names: list[str],
    normalization_mean: list[float],
    normalization_std: list[float],
    *,
    batch_size: int = 128,
    seed: int = 20261104,
    bootstrap_replicates: int = 500,
    artifacts_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, np.ndarray]:
    tensor_images = torch.from_numpy(images).permute(0, 3, 1, 2)
    mean = torch.tensor(normalization_mean, dtype=torch.float32)
    std = torch.tensor(normalization_std, dtype=torch.float32)
    truth, clean_scores = _predict_tensor(
        model,
        tensor_images,
        labels,
        evaluation_indices,
        mean,
        std,
        batch_size=batch_size,
    )
    metrics, per_class, confusion = metrics_from_scores(truth, clean_scores, class_names)
    confidence = class_stratified_bootstrap(
        truth, clean_scores, seed=seed, replicates=bootstrap_replicates
    )
    robustness_rows = []
    for corruption in ("clean", "mean_blur_3x3", "brightness_plus_20_percent"):
        corrupted_truth, scores = _predict_tensor(
            model,
            tensor_images,
            labels,
            evaluation_indices,
            mean,
            std,
            batch_size=batch_size,
            corruption=corruption,
        )
        predicted = scores.argmax(axis=1)
        robustness_rows.append(
            {
                "condition": corruption,
                "accuracy": float(accuracy_score(corrupted_truth, predicted)),
                "macro_f1": float(f1_score(corrupted_truth, predicted, average="macro")),
                "rows": len(corrupted_truth),
            }
        )
    metrics["accuracy_95_percent_ci_low"] = confidence["accuracy"]["ci_low"]
    metrics["accuracy_95_percent_ci_high"] = confidence["accuracy"]["ci_high"]
    metrics["macro_f1_95_percent_ci_low"] = confidence["macro_f1"]["ci_low"]
    metrics["macro_f1_95_percent_ci_high"] = confidence["macro_f1"]["ci_high"]
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    per_class.to_csv(artifacts_dir / "per_class_metrics.csv", index=False, float_format="%.8f")
    pd.DataFrame(confusion, index=class_names, columns=class_names).to_csv(
        artifacts_dir / "confusion_matrix.csv"
    )
    robustness = pd.DataFrame(robustness_rows)
    robustness.to_csv(artifacts_dir / "robustness.csv", index=False, float_format="%.8f")
    return metrics, per_class, robustness, clean_scores
