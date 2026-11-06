"""Streamlit front end for image object detection with YOLO."""

import logging

import streamlit as st
from ultralytics import YOLO

from detection import InvalidImageError, load_image, summarize_detections

MODEL_PATH = "yolo26x.pt"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner="Loading YOLO model...")
def load_model() -> YOLO:
    """Load and reuse the configured YOLO model across Streamlit reruns."""
    return YOLO(MODEL_PATH)


def main() -> None:
    """Render the uploader and run inference only for a supplied image."""
    st.set_page_config(page_title="YOLO Photo Inspector", page_icon="🔎")
    st.title("What is in my photo?")
    st.write("Upload a JPEG or PNG to detect and count objects.")
    uploaded = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"])
    limit = st.slider("Minimum confidence (%)", min_value=5, max_value=95, value=25)

    if uploaded is None:
        st.info("Upload a photo to begin object detection.")
        return

    try:
        photo = load_image(uploaded.getvalue())
    except InvalidImageError as exc:
        st.error(str(exc))
        return

    try:
        model = load_model()
        with st.spinner("Detecting objects..."):
            result = model(photo, conf=limit / 100, verbose=False)[0]
            detections, counts = summarize_detections(result.summary())
            annotated = result.plot(pil=True)
    except (OSError, RuntimeError, ValueError, KeyError, IndexError, TypeError) as exc:
        logger.exception("Object detection failed")
        st.error("Detection failed. Check model availability and server logs.")
        return

    st.image(annotated, caption="What YOLO found", use_container_width=True)
    st.subheader("Detection Results")
    st.write(f"The model found {len(detections)} things.")
    for name, confidence in detections:
        st.write(f"{name}: {confidence * 100:.0f}%")

    st.subheader("Detection Report")
    if counts:
        for name, count in counts.items():
            st.write(f"{name}: {count}")
    else:
        st.info("No objects detected.")


if __name__ == "__main__":
    main()
