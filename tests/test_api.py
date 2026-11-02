from __future__ import annotations

import asyncio
from io import BytesIO

import pytest
from api.main import app, health, predict
from fastapi import HTTPException
from PIL import Image


class MemoryUpload:
    def __init__(self, content_type: str, payload: bytes) -> None:
        self.content_type = content_type
        self.payload = payload

    async def read(self, size: int) -> bytes:
        return self.payload[:size]


def _image_payload() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (64, 64), (64, 128, 192)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_health_reports_generated_model_artifacts() -> None:
    assert health() == {"status": "ok", "model_available": True}
    assert any(getattr(route, "path", None) == "/predict" for route in app.routes)


def test_predict_rejects_unsupported_content_type() -> None:
    with pytest.raises(HTTPException) as error:
        asyncio.run(predict(MemoryUpload("text/plain", b"text")))

    assert error.value.status_code == 415


def test_predict_returns_top_three_for_small_rgb_image() -> None:
    body = asyncio.run(predict(MemoryUpload("image/jpeg", _image_payload())))

    assert len(body["predictions"]) == 3
    assert body["score_type"] == "uncalibrated softmax confidence"
    assert all(0.0 <= row["confidence"] <= 1.0 for row in body["predictions"])
