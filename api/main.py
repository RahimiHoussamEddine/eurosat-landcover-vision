"""FastAPI image inference endpoint for the compact EuroSAT classifier."""

from __future__ import annotations

import io
import json
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from eurosat_landcover_vision.inference import load_classifier, predict_image

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "v1"
MODEL_PATH = ARTIFACTS / "landcover_cnn_state.pt"
SUMMARY_PATH = ARTIFACTS / "analysis_summary.json"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}

app = FastAPI(
    title="EuroSAT Land-Cover Classifier",
    version="1.0.0",
    description="CPU inference for 64x64 Sentinel-2 RGB land-cover benchmark patches.",
)


@lru_cache(maxsize=1)
def _assets() -> tuple:
    summary = json.loads(SUMMARY_PATH.read_text())
    model = load_classifier(MODEL_PATH)
    return (
        model,
        summary["normalization_mean"],
        summary["normalization_std"],
        summary["data_archive"]["sha256"],
    )


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {"status": "ok", "model_available": MODEL_PATH.is_file() and SUMMARY_PATH.is_file()}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:  # noqa: B008
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPEG or PNG image.")
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image uploads are limited to 5 MiB.")
    try:
        with Image.open(io.BytesIO(contents)) as image:
            if image.width > 4096 or image.height > 4096 or image.width * image.height > 16_777_216:
                raise HTTPException(
                    status_code=413, detail="Image dimensions exceed the 4096-pixel limit."
                )
            image.load()
            rgb = image.convert("RGB")
    except HTTPException:
        raise
    except Image.DecompressionBombError as error:
        raise HTTPException(
            status_code=413, detail="Image dimensions exceed the supported limit."
        ) from error
    except (OSError, UnidentifiedImageError) as error:
        raise HTTPException(
            status_code=400, detail="The uploaded file is not a valid image."
        ) from error
    if not MODEL_PATH.is_file() or not SUMMARY_PATH.is_file():
        raise HTTPException(status_code=503, detail="Model artifacts are not installed.")
    model, mean, std, source_sha256 = _assets()
    predictions = predict_image(model, rgb, mean, std, top_k=3)
    return {
        "predictions": predictions,
        "score_type": "uncalibrated softmax confidence",
        "model": "LandCoverCNN v1",
        "source_dataset_sha256": source_sha256,
        "warning": "Benchmark classifier only; not validated for geographic mapping or other image sources.",
    }
