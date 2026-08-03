import gc
import os
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch
from ultralytics import YOLO

# CPU optimization
torch.set_num_threads(2)

# Page config
st.set_page_config(
    page_title="YOLOv8 Interactive Web Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS Styling
st.markdown(
    """
<style>
.main-header { font-size: 2.2rem; font-weight: 700; color: #1E293B; }
.sub-header { font-size: 1rem; color: #64748B; margin-bottom: 1.5rem; }
/* 让按钮在容器中完全填充，保持响应式布局 */
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

# Configuration Constants
TASK_MODELS = {
    "Object Detection": {
        "filename": "yolov8n.pt",
        "desc": "Detects bounding boxes and categories for 80 COCO classes.",
    },
    "Image Classification": {
        "filename": "yolov8n-cls.pt",
        "desc": "Classifies images using ImageNet 1000 classes.",
    },
}
MAX_FILE_SIZE_MB = 5
MAX_IMAGE_DIM = 640


@st.cache_resource(show_spinner=False)
def load_yolo_model(model_filename: str):
    return YOLO(model_filename)


def resize_image_if_needed(image: Image.Image, max_dim: int = 640) -> Image.Image:
    width, height = image.size
    if width <= max_dim and height <= max_dim:
        return image
    ratio = min(max_dim / width, max_dim / height)
    return image.resize(
        (int(width * ratio), int(height * ratio)), Image.Resampling.LANCZOS
    )


# Sidebar Configuration
with st.sidebar:
    st.title("⚙️ YOLO Configuration")
    st.divider()

    task = st.selectbox("🎯 Select Vision Task", list(TASK_MODELS.keys()))
    model_info = TASK_MODELS[task]

    st.caption(f"Model: `{model_info['filename']}`")
    st.info(model_info["desc"])
    st.divider()

    conf_threshold = st.slider("🎚️ Confidence Threshold", 0.05, 1.0, 0.25, 0.05)
    st.divider()

    input_source = st.radio(
        "📥 Image Input Source", ["Upload Image", "Use Sample Image"]
    )

# Main Page Header
st.markdown(
    "<div class='main-header'>⚡ Lightweight YOLOv8 Demo</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='sub-header'>Powered by Ultralytics YOLOv8n | Optimized for CPU Streamlit Deployment</div>",
    unsafe_allow_html=True,
)

# Input Image Handling
input_image = None
if input_source == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload JPG / PNG / WEBP (Max 5MB)", type=["jpg", "jpeg", "png", "webp"]
    )
    if uploaded_file:
        file_size = uploaded_file.size / (1024 * 1024)
        if file_size > MAX_FILE_SIZE_MB:
            st.error(f"File size {file_size:.2f}MB exceeds 5MB limit.")
        else:
            try:
                input_image = Image.open(uploaded_file).convert("RGB")
            except Exception:
                st.error("Unable to read image file.")
else:
    sample_path = "assets/sample.jpg"
    if os.path.exists(sample_path):
        input_image = Image.open(sample_path).convert("RGB")
    else:
        st.warning("Sample image not found.")

# Inference Logic
if input_image:
    input_image = resize_image_if_needed(input_image, MAX_IMAGE_DIM)
    width, height = input_image.size

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ Input Image")
        st.image(
            input_image, use_container_width=True, caption=f"{width}x{height}px"
        )
        # 添加 use_container_width=True 使按钮与 col1 容器完全同宽且响应式缩放
        run = st.button("🚀 Run Inference", use_container_width=True)

    with col2:
        st.subheader("🎯 Prediction")

    if run:
        try:
            with st.spinner("Running YOLO inference..."):
                model = load_yolo_model(model_info["filename"])
                with torch.no_grad():
                    results = model.predict(
                        source=input_image,
                        conf=conf_threshold,
                        imgsz=640,
                        device="cpu",
                        verbose=False,
                    )
                result = results[0]

            # 1. 目标检测任务
            if task == "Object Detection":
                plotted = result.plot()
                with col2:
                    st.image(
                        plotted[:, :, ::-1],
                        use_container_width=True,
                        caption="Detection Result",
                    )

                # 将分析与表格数据显示在下方
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
                                    round(x, 1) for x in box.xyxy[0].tolist()
                                ],
                            }
                        )

                    df = pd.DataFrame(detections)
                    m_col, d_col = st.columns([1, 3])
                    with m_col:
                        st.metric("Total Objects", len(df))
                    with d_col:
                        st.dataframe(df, use_container_width=True)
                else:
                    st.info("No objects detected.")

            # 2. 图像分类任务
            elif task == "Image Classification":
                probs = result.probs

                # 分类任务同样在 Prediction 栏渲染绘制后的类别结果图
                plotted = result.plot()
                with col2:
                    st.image(
                        plotted[:, :, ::-1],
                        use_container_width=True,
                        caption="Classification Result",
                    )

                # 将分析与图表数据统一显示在下方
                st.divider()
                st.subheader("📊 Top 5 Predictions")

                if probs is not None:
                    predictions = []
                    chart_data = []

                    for idx, confidence in zip(
                        probs.top5, probs.top5conf.tolist()
                    ):
                        conf_val = float(confidence)
                        cls_name = model.names[idx]
                        predictions.append(
                            {
                                "Class": cls_name,
                                "Confidence": f"{conf_val:.2%}",
                            }
                        )
                        chart_data.append(
                            {"Class": cls_name, "Probability": conf_val}
                        )

                    df = pd.DataFrame(predictions)
                    chart_df = pd.DataFrame(chart_data)

                    res_col1, res_col2 = st.columns([1, 2])
                    with res_col1:
                        st.metric(
                            "Top Prediction",
                            df.iloc[0]["Class"],
                            df.iloc[0]["Confidence"],
                        )
                        st.dataframe(df, use_container_width=True)
                    with res_col2:
                        st.bar_chart(chart_df.set_index("Class")["Probability"])

        except Exception as e:
            st.exception(e)
        finally:
            gc.collect()
    else:
        with col2:
            st.info("Click 'Run Inference' to generate predictions.")
else:
    st.info("👈 Upload image or select sample image.")

# Footer
st.divider()
st.markdown(
    "<div style='text-align:center;color:#94A3B8;font-size:0.85rem'>"
    "YOLOv8 Streamlit Demo | Ultralytics + PyTorch + Streamlit"
    "</div>",
    unsafe_allow_html=True,
)