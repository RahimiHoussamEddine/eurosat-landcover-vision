"""EuroSAT archive integrity checks and deterministic image loading."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]
EXPECTED_IMAGE_COUNT = 27_000
EXPECTED_ARCHIVE_MD5 = "f46e308c4d50d4bf32fedad2d3d62f3b"
EXPECTED_ARCHIVE_SHA256 = "b4f5b234ecb7d7ff9c6cddb046543b4717c53fd6e9815be6c0e80cc614f51b90"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_image_folder(
    root: Path,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], dict[str, Any]]:
    """Load and audit the official class-folder archive into NHWC uint8 memory."""
    image_paths = []
    labels = []
    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Expected EuroSAT class directory: {class_dir}")
        for path in sorted(class_dir.glob("*.jpg")):
            image_paths.append(path)
            labels.append(label)
    if len(image_paths) != EXPECTED_IMAGE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_IMAGE_COUNT:,} JPEG images; found {len(image_paths):,}"
        )

    arrays = []
    hashes = []
    dimensions = Counter()
    for path in image_paths:
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        hashes.append(digest)
        try:
            with Image.open(path) as image:
                image.load()
                dimensions[tuple(image.size)] += 1
                if image.size != (64, 64):
                    raise ValueError(f"Unexpected image size {image.size} in {path.name}")
                arrays.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
        except (OSError, UnidentifiedImageError) as error:
            raise ValueError(f"Could not decode image {path}") from error

    label_by_hash: dict[str, int] = {}
    for digest, label in zip(hashes, labels, strict=True):
        previous = label_by_hash.setdefault(digest, label)
        if previous != label:
            raise ValueError("The same image bytes appear with conflicting class labels")
    duplicate_excess = len(hashes) - len(set(hashes))
    counts = Counter(CLASS_NAMES[label] for label in labels)
    images = np.stack(arrays, axis=0)
    label_array = np.asarray(labels, dtype=np.int64)
    groups = hashes
    audit = {
        "image_count": len(image_paths),
        "class_names": CLASS_NAMES,
        "class_counts": dict(sorted(counts.items())),
        "unique_image_hashes": len(set(hashes)),
        "duplicate_image_copies": duplicate_excess,
        "image_dimensions": {f"{w}x{h}": n for (w, h), n in sorted(dimensions.items())},
        "corrupt_images": 0,
        "image_dtype": str(images.dtype),
    }
    return images, label_array, [str(path) for path in image_paths], groups, audit
