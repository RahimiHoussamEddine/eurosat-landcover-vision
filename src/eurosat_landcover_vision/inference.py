"""Image preprocessing and CPU inference shared by the app and API."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from eurosat_landcover_vision.dataset import CLASS_NAMES
from eurosat_landcover_vision.model import LandCoverCNN


def load_classifier(checkpoint_path: Path) -> LandCoverCNN:
    model = LandCoverCNN(num_classes=len(CLASS_NAMES))
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def preprocess_image(image: Image.Image, mean: list[float], std: list[float]) -> torch.Tensor:
    resized = image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.uint8).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).to(dtype=torch.float32).div_(255.0)
    mean_tensor = torch.tensor(mean, dtype=torch.float32).reshape(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=torch.float32).reshape(3, 1, 1)
    return ((tensor - mean_tensor) / std_tensor).unsqueeze(0)


def predict_image(
    model: LandCoverCNN,
    image: Image.Image,
    mean: list[float],
    std: list[float],
    *,
    top_k: int = 3,
) -> list[dict[str, float | str]]:
    if torch.get_num_threads() > 4:
        torch.set_num_threads(4)
    inputs = preprocess_image(image, mean, std)
    with torch.inference_mode():
        probabilities = torch.softmax(model(inputs), dim=1)[0]
    values, indices = torch.topk(probabilities, k=min(top_k, len(CLASS_NAMES)))
    return [
        {"class_name": CLASS_NAMES[int(index)], "confidence": float(value)}
        for value, index in zip(values, indices, strict=True)
    ]
