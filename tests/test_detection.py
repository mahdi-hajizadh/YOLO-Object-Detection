"""Tests for image decoding and detection summaries."""

from io import BytesIO

import pytest
from PIL import Image

from detection import InvalidImageError, load_image, summarize_detections


def make_image(fmt="PNG", size=(10, 10)):
    buffer = BytesIO()
    Image.new("RGB", size, "red").save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
def test_load_image(fmt):
    image = load_image(make_image(fmt))
    assert image.mode == "RGB"
    assert image.size == (10, 10)


@pytest.mark.parametrize("data", [b"", b"invalid", b"x" * (15 * 1024 * 1024 + 1)])
def test_reject_bad_image(data):
    with pytest.raises(InvalidImageError):
        load_image(data)


def test_reject_gif():
    with pytest.raises(InvalidImageError):
        load_image(make_image("GIF"))


def test_summarize():
    entries, counts = summarize_detections([
        {"name": "car", "confidence": 0.9},
        {"name": "car", "confidence": 0.7},
        {"name": "person", "confidence": 0.5},
    ])
    assert len(entries) == 3
    assert counts == {"car": 2, "person": 1}


def test_empty_summary():
    assert summarize_detections([]) == ([], {})


def test_bad_confidence():
    with pytest.raises(ValueError):
        summarize_detections([{"name": "car", "confidence": 1.1}])
