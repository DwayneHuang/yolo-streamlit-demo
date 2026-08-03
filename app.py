import gc
import os
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError
import streamlit as st
import torch
from ultralytics import YOLO


# ============================================================
# Streamlit Cloud / CPU configuration
# ============================================================

# Streamlit Community Cloud is CPU-only for this app.
# Limit PyTorch threads so the app does not consume excessive CPU.
CPU_THREADS = max(1, min(2, os.cpu_count() or 1))
torch.set_num_threads(CPU_THREADS)

# Resolve paths relative to this file, not the current working directory.
# This is more reliable on Streamlit Community Cloud.
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
SAMPLE_IMAGE_PATH = ASSETS_DIR / "sample.jpg"

MAX_FILE_SIZE_MB = 5
MAX_IMAGE_DIM = 640
INFERENCE_IMAGE_SIZE = 640

TASK_MODELS = {
    "Object Detection": {
        "filename": "yolov8n.pt",
        "desc": "Detects bounding boxes and categories for 80 COCO classes.",
    },
    "Image Classification": {
        "filename": "yolov8n-cls.pt",
        "desc": "Classifies images using ImageNet classes.",
    },
}


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="YOLOv8 Interactive Web Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
    }

    .sub-header {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }

    .stButton > button {
        background-color: #2563EB;
        color: white;
        font-weight: 600;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
    }

    .stButton > button:hover {
        background-color: #1D4ED8;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Model loading
# ============================================================

@st.cache_resource(show_spinner=False)
def load_yolo_model(model_filename: str):
    """
    Load and cache a YOLO model.

    If the .pt file is not already present, Ultralytics downloads
    the official pretrained weight automatically and caches it.
    """
    return YOLO(model_filename)


# ============================================================
# Image utilities
# ============================================================

def resize_image_if_needed(
    image: Image.Image,
    max_dim: int = MAX_IMAGE_DIM,
) -> Image.Image:
    """Resize an image while preserving its aspect ratio."""
    width, height = image.size

    if width <= max_dim and height <= max_dim:
        return image

    ratio = min(max_dim / width, max_dim / height)
    new_size = (
        max(1, int(width * ratio)),
        max(1, int(height * ratio)),
    )

    return image.resize(new_size, Image.Resampling.LANCZOS)


def load_uploaded_image(uploaded_file) -> Image.Image | None:
    """Safely decode an uploaded image."""
    try:
        image = Image.open(uploaded_file)
        image.load()
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("无法读取图片。请上传有效的 JPG、PNG 或 WEBP 文件。")
        return None


def load_sample_image() -> Image.Image | None:
    """Load the optional local sample image."""
    if not SAMPLE_IMAGE_PATH.exists():
        st.warning(
            "未找到 `assets/sample.jpg`。"
            "当前部署仍然可以正常使用，请选择「Upload Image」上传图片。"
        )
        return None

    try:
        return Image.open(SAMPLE_IMAGE_PATH).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        st.warning("`assets/sample.jpg` 无法读取，请改用上传图片。")
        return None


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.title("⚙️ YOLO Configuration")
    st.divider()

    task = st.selectbox(
        "🎯 Select Vision Task",
        list(TASK_MODELS.keys()),
    )
    model_info = TASK_MODELS[task]

    st.caption(f"Model: `{model_info['filename']}`")
    st.info(model_info["desc"])
    st.divider()

    conf_threshold = st.slider(
        "🎚️ Confidence Threshold",
        min_value=0.05,
        max_value=1.0,
        value=0.25,
        step=0.05,
    )

    st.divider()

    input_source = st.radio(
        "📥 Image Input Source",
        ["Upload Image", "Use Sample Image"],
    )

    st.divider()
    st.caption(f"CPU threads: {CPU_THREADS}")
    st.caption("Inference device: CPU")


# ============================================================
# Main header
# ============================================================

st.markdown(
    "<div class='main-header'>⚡ Lightweight YOLOv8 Demo</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='sub-header'>"
    "Powered by Ultralytics YOLOv8n | "
    "Optimized for CPU Streamlit Community Cloud"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# Input image
# ============================================================

input_image = None

if input_source == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload JPG / PNG / WEBP (Max 5MB)",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / (1024 * 1024)

        if file_size_mb > MAX_FILE_SIZE_MB:
            st.error(
                f"File size {file_size_mb:.2f} MB exceeds "
                f"the {MAX_FILE_SIZE_MB} MB limit."
            )
        else:
            input_image = load_uploaded_image(uploaded_file)

else:
    input_image = load_sample_image()


# ============================================================
# Inference
# ============================================================

if input_image is not None:
    input_image = resize_image_if_needed(input_image)

    width, height = input_image.size

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ Input Image")
        st.image(
            input_image,
            use_container_width=True,
            caption=f"{width} × {height}px",
        )

        run_inference = st.button(
            "🚀 Run Inference",
            use_container_width=True,
            type="primary",
        )

    with col2:
        st.subheader("🎯 Prediction")

    if run_inference:
        try:
            with st.spinner(
                f"Loading {model_info['filename']} and running inference..."
            ):
                model = load_yolo_model(model_info["filename"])

                # CPU-only inference for Streamlit Community Cloud.
                with torch.inference_mode():
                    results = model.predict(
                        source=input_image,
                        conf=conf_threshold,
                        imgsz=INFERENCE_IMAGE_SIZE,
                        device="cpu",
                        verbose=False,
                    )

                result = results[0]

            # ------------------------------------------------
            # Object Detection
            # ------------------------------------------------
            if task == "Object Detection":
                plotted = result.plot()

                with col2:
                    st.image(
                        plotted[:, :, ::-1],
                        use_container_width=True,
                        caption="Detection Result",
                    )

                st.divider()
                st.subheader("📊 Analysis Results")

                boxes = result.boxes

                if boxes is not None and len(boxes) > 0:
                    detections = []

                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        confidence = float(box.conf[0].item())

                        detections.append(
                            {
                                "Class": model.names[cls_id],
                                "Confidence": f"{confidence:.2%}",
                                "Bounding Box": [
                                    round(float(x), 1)
                                    for x in box.xyxy[0].tolist()
                                ],
                            }
                        )

                    df = pd.DataFrame(detections)

                    metric_col, table_col = st.columns([1, 3])

                    with metric_col:
                        st.metric(
                            "Total Objects",
                            len(df),
                        )

                    with table_col:
                        st.dataframe(
                            df,
                            use_container_width=True,
                            hide_index=True,
                        )

                else:
                    st.info("No objects detected.")

            # ------------------------------------------------
            # Image Classification
            # ------------------------------------------------
            elif task == "Image Classification":
                probs = result.probs

                # Keep the original UI behavior: show the plotted
                # classification result in the Prediction column.
                plotted = result.plot()

                with col2:
                    st.image(
                        plotted[:, :, ::-1],
                        use_container_width=True,
                        caption="Classification Result",
                    )

                st.divider()
                st.subheader("📊 Top 5 Predictions")

                if probs is not None:
                    top_indices = probs.top5
                    top_confidences = probs.top5conf

                    predictions = []

                    for idx, confidence in zip(
                        top_indices,
                        top_confidences,
                    ):
                        class_id = int(idx)
                        confidence_value = float(confidence)

                        predictions.append(
                            {
                                "Class": model.names[class_id],
                                "Confidence": f"{confidence_value:.2%}",
                                "Probability": confidence_value,
                            }
                        )

                    df = pd.DataFrame(predictions)

                    if not df.empty:
                        result_col, chart_col = st.columns([1, 2])

                        with result_col:
                            st.metric(
                                "Top Prediction",
                                df.iloc[0]["Class"],
                                df.iloc[0]["Confidence"],
                            )

                            st.dataframe(
                                df[["Class", "Confidence"]],
                                use_container_width=True,
                                hide_index=True,
                            )

                        with chart_col:
                            chart_df = df[
                                ["Class", "Probability"]
                            ].set_index("Class")

                            st.bar_chart(chart_df)

                    else:
                        st.info("No classification result returned.")

        except Exception as exc:
            st.error("Inference failed.")
            st.exception(exc)

        finally:
            gc.collect()

    else:
        with col2:
            st.info("Click 'Run Inference' to generate predictions.")

else:
    st.info("👈 Upload an image or select a valid sample image.")


# ============================================================
# Footer
# ============================================================

st.divider()

st.markdown(
    "<div style='text-align:center;color:#94A3B8;font-size:0.85rem'>"
    "YOLOv8 Streamlit Demo | Ultralytics + PyTorch + Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
