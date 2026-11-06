"""Image validation and YOLO result reporting helpers."""

from collections import Counter
from io import BytesIO
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000


class InvalidImageError(ValueError):
    """Raised when an uploaded image cannot be safely processed."""


def load_image(data: bytes) -> Image.Image:
    """Decode a bounded image upload, normalize orientation and return RGB pixels."""
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise InvalidImageError("Image must be nonempty and at most 15 MB.")
    try:
        with Image.open(BytesIO(data)) as source:
            if source.format not in {"JPEG", "PNG"}:
                raise InvalidImageError("Only JPEG and PNG images are supported.")
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise InvalidImageError("Image dimensions exceed the 25 MP limit.")
            source.load()
            return ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        if isinstance(exc, InvalidImageError):
            raise
        raise InvalidImageError("Unable to read a valid JPEG or PNG image.") from exc


def summarize_detections(items: list[dict[str, Any]]) -> tuple[list[tuple[str, float]], dict[str, int]]:
    """Validate and format YOLO summary entries for display."""
    detections = []
    for item in items:
        name = str(item["name"])
        confidence = float(item["confidence"])
        if not 0 <= confidence <= 1:
            raise ValueError("Detection confidence must be between 0 and 1.")
        detections.append((name, confidence))
    return detections, dict(Counter(name for name, _ in detections))
