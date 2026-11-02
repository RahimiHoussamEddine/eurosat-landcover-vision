"""Reproducible EuroSAT split, model comparison, robustness, and reporting."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

from eurosat_landcover_vision.dataset import CLASS_NAMES, load_image_folder, sha256_file
from eurosat_landcover_vision.inference import load_classifier
from eurosat_landcover_vision.model import (
    MODEL_CONFIG_ID,
    class_stratified_bootstrap,
    evaluate_cnn,
    fit_hog_baseline,
    metrics_from_scores,
    train_cnn,
)

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261102
BOOTSTRAP_SEED = 20261103
BOOTSTRAP_REPLICATES = 500
DATA_ROOT = ROOT / "data" / "processed" / "EuroSAT_RGB"
ARCHIVE_PATH = ROOT / "data" / "raw" / "EuroSAT_RGB.zip"
ARTIFACTS = ROOT / "artifacts" / "v1"
REPORTS = ROOT / "reports"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _split_data(
    labels: np.ndarray, groups: np.ndarray, *, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def partition(
        partition_labels: np.ndarray,
        partition_groups: np.ndarray,
        *,
        folds: int,
        heldout_folds: int,
        fold_seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        splitter = StratifiedGroupKFold(
            n_splits=folds,
            shuffle=True,
            random_state=fold_seed,
        )
        heldout_parts = []
        training_parts = []
        for fold, (_training, heldout) in enumerate(
            splitter.split(np.zeros(len(partition_labels)), partition_labels, partition_groups)
        ):
            (heldout_parts if fold < heldout_folds else training_parts).append(heldout)
        return np.sort(np.concatenate(training_parts)), np.sort(np.concatenate(heldout_parts))

    development_indices, test_indices = partition(
        labels, groups, folds=20, heldout_folds=3, fold_seed=seed
    )
    train_relative, validation_relative = partition(
        labels[development_indices],
        groups[development_indices],
        folds=17,
        heldout_folds=3,
        fold_seed=seed + 1,
    )
    train_indices = development_indices[train_relative]
    validation_indices = development_indices[validation_relative]
    groups_by_split = [
        set(groups[split]) for split in (train_indices, validation_indices, test_indices)
    ]
    if any(groups_by_split[i] & groups_by_split[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Exact duplicate images crossed a data split")
    for indices in (train_indices, validation_indices, test_indices):
        if len(np.unique(labels[indices])) != len(CLASS_NAMES):
            raise ValueError("A split is missing at least one class")
    return train_indices, validation_indices, test_indices


def _class_counts(labels: np.ndarray, indices: np.ndarray) -> dict[str, int]:
    counts = np.bincount(labels[indices], minlength=len(CLASS_NAMES))
    return {name: int(counts[index]) for index, name in enumerate(CLASS_NAMES)}


def _recover_training_metadata(
    model: torch.nn.Module,
    history: pd.DataFrame,
    images: np.ndarray,
    train_indices: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    """Rebuild metadata when a prior run saved weights before its manifest."""
    tensor_images = torch.from_numpy(images).permute(0, 3, 1, 2)
    train_float = tensor_images[train_indices].to(dtype=torch.float32).div_(255.0)
    mean = train_float.mean(dim=(0, 2, 3))
    std = train_float.std(dim=(0, 2, 3), unbiased=False).clamp_min(1e-6)
    best = history.loc[history["validation_macro_f1"].idxmax()]
    return {
        "model_config_id": MODEL_CONFIG_ID,
        "architecture": "LandCoverCNN small residual CNN trained from scratch",
        "seed": seed,
        "requested_epochs": 5,
        "completed_epochs": len(history),
        "best_epoch": int(best["epoch"]),
        "best_validation_macro_f1": float(best["validation_macro_f1"]),
        "batch_size": 256,
        "patience": 2,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "label_smoothing": 0.05,
        "augmentation": ["horizontal flip", "vertical flip", "90-degree rotations"],
        "normalization_mean": [float(value) for value in mean],
        "normalization_std": [float(value) for value in std],
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def _create_figures() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    import matplotlib.pyplot as plt

    comparison = pd.read_csv(ARTIFACTS / "model_comparison.csv")
    matrix = pd.read_csv(ARTIFACTS / "confusion_matrix.csv", index_col=0)
    robustness = pd.read_csv(ARTIFACTS / "robustness.csv")
    figure_dir = REPORTS / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.4), layout="constrained")
    metric_columns = ["accuracy", "macro_f1", "top3_accuracy"]
    x = np.arange(len(metric_columns))
    width = 0.34
    colors = {"Residual CNN": "#007c91", "HOG + linear SVM": "#e07a28"}
    for offset, model_name in enumerate(comparison["model"]):
        row = comparison[comparison["model"] == model_name].iloc[0]
        values = [row[column] for column in metric_columns]
        bars = ax.bar(
            x + (offset - 0.5) * width,
            values,
            width,
            label=model_name,
            color=colors[model_name],
            alpha=0.92,
        )
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.012,
                f"{value:.3f}",
                ha="center",
                fontsize=8,
            )
    ax.set_xticks(x, ["Accuracy", "Macro F1", "Top-3 accuracy"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Test-set score")
    ax.set_title("EuroSAT RGB model comparison on the locked test split")
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(figure_dir / "model_comparison.png", dpi=170, facecolor="white")
    plt.close(fig)

    normalized = matrix.div(matrix.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    display = ax.imshow(normalized.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Residual CNN row-normalized confusion matrix")
    for row in range(len(CLASS_NAMES)):
        for column in range(len(CLASS_NAMES)):
            value = normalized.iloc[row, column]
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value > 0.55 else "#1f2937",
            )
    fig.colorbar(display, ax=ax, label="Share of true class")
    fig.savefig(figure_dir / "confusion_matrix.png", dpi=170, facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.6), layout="constrained")
    x = np.arange(len(robustness))
    ax.bar(x, robustness["accuracy"], color=["#007c91", "#e07a28", "#7b61a8"])
    ax.set_xticks(x, ["Clean", "3×3 blur", "+20% brightness"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Controlled image-corruption sensitivity")
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    for index, row in robustness.iterrows():
        ax.text(index, row["accuracy"] + 0.015, f"{row['accuracy']:.3f}", ha="center")
    fig.savefig(figure_dir / "robustness.png", dpi=170, facecolor="white")
    plt.close(fig)


def _write_report(summary: dict[str, Any]) -> None:
    comparison = pd.DataFrame(summary["model_comparison"])
    robustness = pd.DataFrame(summary["robustness"])
    class_metrics = pd.read_csv(ARTIFACTS / "per_class_metrics.csv")
    confusion = pd.read_csv(ARTIFACTS / "confusion_matrix.csv", index_col=0).to_numpy()
    confusion_pairs = []
    for true_id, true_name in enumerate(CLASS_NAMES):
        for predicted_id, predicted_name in enumerate(CLASS_NAMES):
            if true_id != predicted_id and confusion[true_id, predicted_id] > 0:
                confusion_pairs.append(
                    (int(confusion[true_id, predicted_id]), true_name, predicted_name)
                )
    confusion_pairs.sort(reverse=True)
    cnn = summary["cnn_test_metrics"]
    lines = [
        "# EuroSAT land-cover classification results",
        "",
        "## Study question",
        "",
        "Can a compact image model classify EuroSAT RGB land-cover patches, where does it fail, and how sensitive is it to mild image corruption?",
        "",
        "## Data and split",
        "",
        f"- Official EuroSAT RGB v2 archive: {summary['data_archive']['image_count']:,} valid 64×64 RGB images across {len(CLASS_NAMES)} classes.",
        f"- Archive SHA-256: `{summary['data_archive']['sha256']}`; exact duplicate image copies: {summary['data_audit']['duplicate_image_copies']:,}.",
        f"- Duplicate-group-aware split: {summary['split']['train_rows']:,} train, {summary['split']['validation_rows']:,} validation, {summary['split']['test_rows']:,} locked test images.",
        "",
        "## Locked test performance",
        "",
        "| Model | Accuracy (95% CI) | Macro F1 (95% CI) | Balanced accuracy | Top-3 accuracy | Brier score |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        accuracy_ci = (
            f"{row.accuracy:.3f} "
            f"({row.accuracy_95_percent_ci_low:.3f}, {row.accuracy_95_percent_ci_high:.3f})"
        )
        f1_ci = (
            f"{row.macro_f1:.3f} "
            f"({row.macro_f1_95_percent_ci_low:.3f}, {row.macro_f1_95_percent_ci_high:.3f})"
        )
        brier = f"{row.multiclass_brier:.3f}" if pd.notna(row.multiclass_brier) else "not estimated"
        lines.append(
            f"| {row.model} | {accuracy_ci} | {f1_ci} | {row.balanced_accuracy:.3f} | {row.top3_accuracy:.3f} | {brier} |"
        )
    lines.extend(
        [
            "",
            "![Model comparison](figures/model_comparison.png)",
            "",
            f"The residual CNN selected epoch {summary['cnn_training']['best_epoch']} by validation macro F1. Its test accuracy is {cnn['accuracy']:.3%} (95% class-stratified bootstrap CI {cnn['accuracy_95_percent_ci_low']:.3%} to {cnn['accuracy_95_percent_ci_high']:.3%}); macro F1 is {cnn['macro_f1']:.3%}.",
            "",
            "## Error analysis",
            "",
            "The lowest-F1 classes are:",
            "",
            "| Class | Precision | Recall | F1 | Support |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in class_metrics.nsmallest(5, "f1").itertuples(index=False):
        lines.append(
            f"| {row.class_name} | {row.precision:.3f} | {row.recall:.3f} | {row.f1:.3f} | {row.support:,} |"
        )
    lines.extend(["", "Largest off-diagonal confusions:", ""])
    for count, true_name, predicted_name in confusion_pairs[:5]:
        lines.append(f"- {true_name} predicted as {predicted_name}: {count} images.")
    lines.extend(
        [
            "",
            "![Row-normalized CNN confusion matrix](figures/confusion_matrix.png)",
            "",
            "## Controlled robustness checks",
            "",
            "| Test condition | Accuracy | Macro F1 |",
            "|---|---:|---:|",
        ]
    )
    condition_labels = {
        "clean": "Clean RGB",
        "mean_blur_3x3": "3×3 mild blur",
        "brightness_plus_20_percent": "+20% brightness",
    }
    for row in robustness.itertuples(index=False):
        lines.append(
            f"| {condition_labels[row.condition]} | {row.accuracy:.3f} | {row.macro_f1:.3f} |"
        )
    lines.extend(
        [
            "",
            "![Controlled corruption sensitivity](figures/robustness.png)",
            "",
            "These controlled corruptions are stress checks, not a domain-generalization evaluation. Confidence intervals are class-stratified bootstrap intervals on the fixed test split; they do not include split or training-seed uncertainty.",
            "",
            "## Interpretation limits",
            "",
            "EuroSAT RGB contains anonymized, geographically sampled image patches. The random stratified split does not establish performance on a new region, time, sensor, or spatially separated map tile. Some classes are visually similar and the study does not weight class mistakes by operational cost. The app and API are benchmark demos, not land-cover mapping products. Dataset attribution and terms are recorded in [NOTICE.md](../NOTICE.md).",
            "",
        ]
    )
    (REPORTS / "RESULTS.md").write_text("\n".join(lines))


def build() -> dict[str, Any]:
    if not ARCHIVE_PATH.is_file():
        raise FileNotFoundError("Run `make fetch` before `make build`.")
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError("Extracted EuroSAT image folder is missing. Run `make fetch`.")
    if (
        sha256_file(ARCHIVE_PATH)
        != "b4f5b234ecb7d7ff9c6cddb046543b4717c53fd6e9815be6c0e80cc614f51b90"
    ):
        raise ValueError("EuroSAT archive SHA-256 mismatch")
    images, labels, _paths, groups, data_audit = load_image_folder(DATA_ROOT)
    groups_array = np.asarray(groups)
    train_indices, validation_indices, test_indices = _split_data(labels, groups_array, seed=SEED)
    split = {
        "seed": SEED,
        "train_rows": len(train_indices),
        "validation_rows": len(validation_indices),
        "test_rows": len(test_indices),
        "train_class_counts": _class_counts(labels, train_indices),
        "validation_class_counts": _class_counts(labels, validation_indices),
        "test_class_counts": _class_counts(labels, test_indices),
        "split_method": (
            "stratified group k-fold with 3/20 outer folds for test and "
            "3/17 development folds for validation; exact duplicate hashes stay together"
        ),
    }
    model_path = ARTIFACTS / "landcover_cnn_state.pt"
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    metadata_path = ARTIFACTS / "training_metadata.json"
    history_path = ARTIFACTS / "training_history.csv"
    model = None
    training = None
    history = None
    if model_path.is_file() and history_path.is_file():
        candidate_history = pd.read_csv(history_path)
        candidate_model = load_classifier(model_path)
        if metadata_path.is_file():
            candidate = json.loads(metadata_path.read_text())
        else:
            candidate = _recover_training_metadata(
                candidate_model,
                candidate_history,
                images,
                train_indices,
                seed=SEED,
            )
            _write_json(metadata_path, candidate)
        if (
            candidate.get("model_config_id") == MODEL_CONFIG_ID
            and candidate.get("seed") == SEED
            and candidate.get("requested_epochs") == 5
            and candidate.get("batch_size") == 256
            and candidate.get("patience") == 2
        ):
            model = candidate_model
            training = candidate
            history = candidate_history
    if model is None:
        model, training, history = train_cnn(
            images,
            labels,
            train_indices,
            validation_indices,
            model_path,
            seed=SEED,
        )
        history.to_csv(history_path, index=False, float_format="%.8f")
    cnn_metrics, _, robustness, cnn_scores = evaluate_cnn(
        model,
        images,
        labels,
        test_indices,
        CLASS_NAMES,
        training["normalization_mean"],
        training["normalization_std"],
        seed=BOOTSTRAP_SEED,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        artifacts_dir=ARTIFACTS,
    )
    hog_model_info, hog_scores, hog_truth = fit_hog_baseline(
        images,
        labels,
        train_indices,
        test_indices,
        seed=SEED,
    )
    hog_metrics, _, _ = metrics_from_scores(
        hog_truth,
        hog_scores,
        CLASS_NAMES,
        probabilities_usable=False,
    )
    hog_confidence = class_stratified_bootstrap(
        hog_truth,
        hog_scores,
        seed=BOOTSTRAP_SEED + 1,
        replicates=BOOTSTRAP_REPLICATES,
    )
    hog_metrics["accuracy_95_percent_ci_low"] = hog_confidence["accuracy"]["ci_low"]
    hog_metrics["accuracy_95_percent_ci_high"] = hog_confidence["accuracy"]["ci_high"]
    hog_metrics["macro_f1_95_percent_ci_low"] = hog_confidence["macro_f1"]["ci_low"]
    hog_metrics["macro_f1_95_percent_ci_high"] = hog_confidence["macro_f1"]["ci_high"]
    comparison = pd.DataFrame(
        [
            {"model": "Residual CNN", **cnn_metrics},
            {"model": "HOG + linear SVM", **hog_metrics},
        ]
    )
    comparison.to_csv(ARTIFACTS / "model_comparison.csv", index=False, float_format="%.8f")
    comparison_records = comparison.to_dict(orient="records")
    for record in comparison_records:
        for key, value in record.items():
            if pd.isna(value):
                record[key] = None
    data_archive = {
        "name": "EuroSAT RGB v2",
        "url": "https://zenodo.org/records/7711810",
        "image_count": data_audit["image_count"],
        "md5": "f46e308c4d50d4bf32fedad2d3d62f3b",
        "sha256": sha256_file(ARCHIVE_PATH),
        "source_rows": len(labels),
    }
    summary = {
        "project": "EuroSAT Land-Cover Vision",
        "class_names": CLASS_NAMES,
        "data_archive": data_archive,
        "data_audit": data_audit,
        "split": split,
        "cnn_training": training,
        "cnn_test_metrics": cnn_metrics,
        "hog_test_metrics": hog_metrics,
        "hog_baseline": hog_model_info,
        "model_comparison": comparison_records,
        "robustness": robustness.to_dict(orient="records"),
        "test_rows": len(test_indices),
        "test_score_shape": list(cnn_scores.shape),
        "normalization_mean": training["normalization_mean"],
        "normalization_std": training["normalization_std"],
        "probability_note": "CNN softmax scores are uncalibrated confidence scores; calibration metrics are descriptive.",
        "geographic_generalization": "not tested; split is stratified and exact-duplicate-group-aware, not spatial",
    }
    _write_json(ARTIFACTS / "data_audit.json", data_audit)
    _write_json(ARTIFACTS / "split_manifest.json", split)
    _write_json(ARTIFACTS / "analysis_summary.json", summary)
    _create_figures()
    _write_report(summary)
    return summary
