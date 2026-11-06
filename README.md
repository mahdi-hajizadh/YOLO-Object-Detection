# YOLO Photo Inspector

A small Streamlit application for object detection and per-class counting in uploaded JPEG and PNG images using Ultralytics YOLO.

## Features

- JPEG/PNG upload with 15 MB and 25 megapixel validation limits
- EXIF orientation correction and RGB normalization
- Adjustable detection confidence (5–95%)
- Annotated image, individual detection confidence and per-class counts
- Cached model initialization and user-friendly error reporting

## Requirements

Python 3.10+ is recommended. Install dependencies in a virtual environment:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

```bash
streamlit run detect_app.py
```

Open the local URL printed by Streamlit, upload a photo, and adjust the confidence slider. The application does not run inference until a photo is uploaded.

The model is configured as `yolo26n.pt`. Ultralytics may download the weights on first run; network access or a locally available compatible weight file is required. Model weights are not bundled in this repository. Model accuracy and detected classes depend on the selected weights.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m compileall -q detect_app.py detection.py tests
```

The unit tests cover image decoding, input rejection, and count aggregation. They do not test actual YOLO inference or the Streamlit UI.

## Repository layout

- `detect_app.py` — Streamlit UI and YOLO inference
- `detection.py` — image validation and result aggregation
- `tests/` — dependency-light unit tests
- `.streamlit/config.toml` — upload limit
- `detect_app_original.py` — original implementation retained for reference
- `TEST_REPORT.md` — verification details
- `GITHUB_PUBLICATION.txt` — publication commands

## Security and operational notes

Only JPEG and PNG are accepted; uploads are bounded in bytes and decoded pixel count. Files are processed in memory. Avoid uploading confidential photos to untrusted deployments. Model weights are excluded from version control. Inference can be CPU-intensive; GPU acceleration depends on your PyTorch installation.

## License

No license was supplied with the original source. This repository is **not licensed for redistribution** until its copyright holder selects and adds a license. Third-party packages and model weights have independent licensing terms; check Ultralytics licensing requirements before public or commercial distribution.
