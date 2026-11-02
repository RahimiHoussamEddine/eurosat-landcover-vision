"""Streamlit benchmark explorer and EuroSAT sample-image classifier."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from eurosat_landcover_vision.inference import load_classifier, predict_image

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "v1"


@st.cache_data
def load_tables() -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = json.loads((ARTIFACTS / "analysis_summary.json").read_text())
    comparison = pd.read_csv(ARTIFACTS / "model_comparison.csv")
    per_class = pd.read_csv(ARTIFACTS / "per_class_metrics.csv")
    confusion = pd.read_csv(ARTIFACTS / "confusion_matrix.csv", index_col=0)
    robustness = pd.read_csv(ARTIFACTS / "robustness.csv")
    return summary, comparison, per_class, confusion, robustness


@st.cache_resource
def load_model_and_normalization() -> tuple:
    summary = json.loads((ARTIFACTS / "analysis_summary.json").read_text())
    model = load_classifier(ARTIFACTS / "landcover_cnn_state.pt")
    return model, summary["normalization_mean"], summary["normalization_std"]


st.set_page_config(page_title="EuroSAT Land-Cover Vision", page_icon="🛰️", layout="wide")
st.title("EuroSAT Land-Cover Vision")
st.caption(
    "A compact residual CNN and HOG baseline for Sentinel-2 RGB benchmark patches, "
    "with class-level error and corruption analysis."
)

required = [
    ARTIFACTS / "analysis_summary.json",
    ARTIFACTS / "model_comparison.csv",
    ARTIFACTS / "landcover_cnn_state.pt",
    ARTIFACTS / "confusion_matrix.csv",
    ARTIFACTS / "per_class_metrics.csv",
    ARTIFACTS / "robustness.csv",
]
if not all(path.is_file() for path in required):
    st.error("Model artifacts are missing. Run `make fetch` and `make build` first.")
    st.stop()

summary, comparison, per_class, confusion, robustness = load_tables()
cnn = summary["cnn_test_metrics"]
metrics = st.columns(4)
metrics[0].metric(
    "Test accuracy",
    f"{cnn['accuracy']:.1%}",
    f"95% CI {cnn['accuracy_95_percent_ci_low']:.1%}–{cnn['accuracy_95_percent_ci_high']:.1%}",
)
metrics[1].metric("Macro F1", f"{cnn['macro_f1']:.3f}")
metrics[2].metric("Top-3 accuracy", f"{cnn['top3_accuracy']:.1%}")
metrics[3].metric("Test images", f"{summary['test_rows']:,}")

left, right = st.columns([1.1, 0.9])
with left:
    st.subheader("CNN vs HOG + linear SVM")
    comparison_view = comparison[
        ["model", "accuracy", "macro_f1", "balanced_accuracy", "top3_accuracy"]
    ].rename(
        columns={
            "model": "Model",
            "accuracy": "Accuracy",
            "macro_f1": "Macro F1",
            "balanced_accuracy": "Balanced accuracy",
            "top3_accuracy": "Top-3 accuracy",
        }
    )
    st.dataframe(comparison_view, hide_index=True, use_container_width=True)
    fig = px.bar(
        comparison,
        x="model",
        y=["accuracy", "macro_f1", "top3_accuracy"],
        barmode="group",
        labels={"model": "Model", "value": "Test score", "variable": "Metric"},
        title="Locked test split",
        template="plotly_white",
    )
    fig.update_layout(yaxis_range=[0, 1], legend_title_text="Metric")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Class-level recall and F1")
    class_fig = px.bar(
        per_class.sort_values("f1"),
        x="f1",
        y="class_name",
        orientation="h",
        color="recall",
        range_x=[0, 1],
        labels={"f1": "F1", "class_name": "Class", "recall": "Recall"},
        template="plotly_white",
    )
    class_fig.update_layout(yaxis_title="", legend_title_text="Recall")
    st.plotly_chart(class_fig, use_container_width=True)

st.subheader("CNN confusion matrix")
normalized = confusion.div(confusion.sum(axis=1), axis=0)
confusion_fig = px.imshow(
    normalized,
    text_auto=".2f",
    aspect="auto",
    color_continuous_scale="Blues",
    labels={"x": "Predicted class", "y": "True class", "color": "Recall share"},
    template="plotly_white",
)
confusion_fig.update_layout(height=620)
st.plotly_chart(confusion_fig, use_container_width=True)

st.subheader("Sensitivity to mild image corruption")
robustness_view = robustness.rename(
    columns={"condition": "Condition", "accuracy": "Accuracy", "macro_f1": "Macro F1"}
)
st.dataframe(robustness_view, hide_index=True, use_container_width=True)

st.subheader("Try an image")
uploaded = st.file_uploader("Upload a JPEG or PNG satellite patch", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    if uploaded.size > 5 * 1024 * 1024:
        st.error("Image uploads are limited to 5 MiB.")
        st.stop()
    image = Image.open(uploaded)
    if image.width > 4096 or image.height > 4096 or image.width * image.height > 16_777_216:
        st.error("Image dimensions exceed the 4096-pixel limit.")
        st.stop()
    image = image.convert("RGB")
    model, mean, std = load_model_and_normalization()
    predictions = predict_image(model, image, mean, std, top_k=3)
    preview, results = st.columns([1, 1])
    with preview:
        st.image(image, caption="Input resized to 64×64 for scoring", width=260)
    with results:
        st.write(f"**Top class:** {predictions[0]['class_name']}")
        st.write(f"Uncalibrated softmax confidence: {predictions[0]['confidence']:.1%}")
        st.dataframe(pd.DataFrame(predictions), hide_index=True, use_container_width=True)

with st.expander("Data, split, and model limits"):
    st.write(
        f"The complete {summary['data_archive']['image_count']:,}-image archive was audited. "
        f"The split used {summary['split']['train_rows']:,} train, "
        f"{summary['split']['validation_rows']:,} validation, and "
        f"{summary['split']['test_rows']:,} test images; exact duplicate hashes were kept in one split."
    )
    st.write(
        "No ImageNet pretrained weights were used. The CNN was trained from scratch with "
        "horizontal/vertical flips and 90-degree rotations. Confidence intervals use a "
        "class-stratified bootstrap on the fixed test set."
    )
    st.write(
        "This random split does not test transfer to new geographic regions, dates, sensors, "
        "or spatially separated map tiles. Uploaded-image scores are benchmark-model outputs, "
        "not operational land-cover labels."
    )
    st.write("See `docs/EXPERIMENT_PROTOCOL.md` and `reports/RESULTS.md` in the repository.")

st.caption(
    "EuroSAT RGB is attributed to its authors and Copernicus Sentinel-2; see NOTICE.md. "
    "The displayed softmax confidence is uncalibrated."
)
