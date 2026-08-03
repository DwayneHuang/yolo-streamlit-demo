import streamlit as st
import os
import gc
import torch
import numpy as np
from PIL import Image
import pandas as pd
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="YOLOv8 Interactive Web Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom CSS for Modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .stButton>button {
        width: 100%;
        background-color: #2563EB;
        color: white;
        font-weight: 600;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Constants & Model Configuration
# ---------------------------------------------------------
HF_REPO_ID = "ultralytics/yolov8"

TASK_MODELS = {
    "Object Detection": {
        "filename": "yolov8n.pt",
        "desc": "Detects bounding boxes & categories for 80 COCO object classes."
    },
    "Image Classification": {
        "filename": "yolov8n-cls.pt",
        "desc": "Classifies the dominant object/scene across 1000 ImageNet classes."
    },
    "Segmentation": {
        "filename": "yolov8n-seg.pt",
        "desc": "Generates precise pixel-level masks and bounding boxes."
    }
}

MAX_FILE_SIZE_MB = 5
MAX_IMAGE_DIM = 1280

# ---------------------------------------------------------
# Model Loader with Cache & Memory Management
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_yolo_model(model_filename: str):
    """
    Downloads model weights automatically from Hugging Face Hub if unavailable locally,
    caches the initialized YOLO instance using @st.cache_resource.
    """
    try:
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=model_filename)
    except Exception as e:
        # Fallback to default Ultralytics weight resolution if Hugging Face hub call raises error
        model_path = model_filename

    model = YOLO(model_path)
    return model


def handle_task_switch(selected_task: str):
    """
    Enforces memory optimization: when switching tasks, unload the previous model
    from memory by clearing st.cache_resource and running garbage collection.
    """
    if "active_task" not in st.session_state:
        st.session_state.active_task = selected_task
    elif st.session_state.active_task != selected_task:
        st.cache_resource.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        st.session_state.active_task = selected_task


def resize_image_if_needed(image: Image.Image, max_dim: int = MAX_IMAGE_DIM) -> Image.Image:
    """
    Resizes large images so maximum dimension does not exceed max_dim (1280px).
    Protects memory and speeds up inference on Streamlit Cloud CPU.
    """
    width, height = image.size
    if width > max_dim or height > max_dim:
        if width > height:
            new_width = max_dim
            new_height = int(height * (max_dim / width))
        else:
            new_height = max_dim
            new_width = int(width * (max_dim / height))
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚙️ YOLO Configuration")
    st.markdown("---")

    # 1. Task Selection
    task = st.selectbox(
        "🎯 Select Vision Task",
        options=list(TASK_MODELS.keys()),
        index=0,
        help="Select between Object Detection, Image Classification, or Instance Segmentation."
    )
    
    # Enforce task switch memory cleanup
    handle_task_switch(task)

    st.caption(f"**Model**: `{TASK_MODELS[task]['filename']}` (YOLOv8 Nano)")
    st.info(TASK_MODELS[task]['desc'])

    st.markdown("---")

    # 2. Confidence Threshold Slider
    conf_threshold = st.slider(
        "🎚️ Confidence Threshold",
        min_value=0.05,
        max_value=1.00,
        value=0.25,
        step=0.05,
        help="Minimum confidence score required for prediction display."
    )

    # 3. Input Mode Selector
    st.markdown("---")
    input_source = st.radio(
        "📥 Image Input Source",
        options=["Upload Image", "Use Sample Image"],
        index=0
    )

# ---------------------------------------------------------
# Main Interface
# ---------------------------------------------------------
st.markdown("<div class='main-header'>⚡ Lightweight YOLOv8 Interactive Demo</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Powered by Ultralytics YOLOv8n & Hugging Face Hub | Optimized for Streamlit Community Cloud</div>", unsafe_allow_html=True)

# Image Acquisition
input_image = None
image_file_name = ""

if input_source == "Upload Image":
    uploaded_file = st.file_uploader(
        "Choose an image (JPG, JPEG, PNG, WEBP — Max 5MB)",
        type=["jpg", "jpeg", "png", "webp"]
    )
    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            st.error(f"❌ File size ({file_size_mb:.2f} MB) exceeds the maximum allowed limit of {MAX_FILE_SIZE_MB} MB. Please upload a smaller image.")
        else:
            try:
                input_image = Image.open(uploaded_file).convert("RGB")
                image_file_name = uploaded_file.name
            except Exception as e:
                st.error(f"❌ Error loading image file: {e}")
else:
    sample_path = "assets/sample.jpg"
    if os.path.exists(sample_path):
        try:
            input_image = Image.open(sample_path).convert("RGB")
            image_file_name = "sample.jpg"
        except Exception as e:
            st.error(f"❌ Could not load sample image: {e}")
    else:
        st.warning("⚠️ Sample image file not found.")

# Display & Inference Workflow
if input_image is not None:
    # Resize image if dimensions exceed limit (memory protection)
    input_image = resize_image_if_needed(input_image)
    orig_w, orig_h = input_image.size

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ Input Image")
        st.image(input_image, use_column_width=True, caption=f"Resolution: {orig_w}x{orig_h}px")

    with col2:
        st.subheader("🎯 Prediction Visualizer")
        run_inference = st.button("🚀 Run Inference")

    if run_inference:
        with st.spinner(f"Loading {TASK_MODELS[task]['filename']} from Hugging Face & running inference..."):
            try:
                model = load_yolo_model(TASK_MODELS[task]['filename'])
                results = model.predict(source=input_image, conf=conf_threshold, verbose=False)
                res = results[0]

                # Render visualized image
                res_plotted = res.plot()  # BGR numpy array
                res_rgb = res_plotted[:, :, ::-1]  # Convert BGR to RGB
                
                with col2:
                    st.image(res_rgb, use_column_width=True, caption="Inference Output")

                st.markdown("---")
                st.subheader("📊 Detailed Prediction Results")

                if task in ["Object Detection", "Segmentation"]:
                    boxes = res.boxes
                    if boxes is not None and len(boxes) > 0:
                        detections = []
                        for box in boxes:
                            cls_id = int(box.cls[0].item())
                            cls_name = model.names[cls_id]
                            conf = float(box.conf[0].item())
                            xyxy = [round(x, 1) for x in box.xyxy[0].tolist()]

                            detections.append({
                                "Class": cls_name,
                                "Confidence": f"{conf:.2%}",
                                "Confidence Score": round(conf, 4),
                                "Bounding Box (xmin, ymin, xmax, ymax)": str(xyxy)
                            })
                        
                        df_dets = pd.DataFrame(detections)
                        st.dataframe(
                            df_dets[["Class", "Confidence", "Bounding Box (xmin, ymin, xmax, ymax)"]],
                            use_container_width=True
                        )

                        # Summary Metrics
                        st.write(f"**Total Detections Found:** `{len(detections)}`")
                    else:
                        st.info("ℹ️ No objects detected above the selected confidence threshold.")

                elif task == "Image Classification":
                    probs = res.probs
                    if probs is not None:
                        top5_indices = probs.top5
                        top5_conf = probs.top5conf.tolist()
                        
                        cls_results = []
                        for idx, conf in zip(top5_indices, top5_conf):
                            cls_name = model.names[idx]
                            cls_results.append({
                                "Class Name": cls_name,
                                "Confidence": round(float(conf), 4),
                                "Confidence (%)": f"{float(conf):.2%}"
                            })

                        df_cls = pd.DataFrame(cls_results)

                        # Display top prediction metric
                        top_pred = cls_results[0]
                        st.metric(
                            label="Top Classification Result",
                            value=top_pred["Class Name"],
                            delta=top_pred["Confidence (%)"]
                        )

                        st.subheader("Top 5 Predictions")
                        st.dataframe(df_cls[["Class Name", "Confidence (%)"]], use_container_width=True)
                        st.bar_chart(df_cls.set_index("Class Name")["Confidence"])

            except Exception as err:
                st.error(f"❌ Inference failed: {err}")
            finally:
                # Memory protection: release temporary arrays and garbage collect
                gc.collect()

else:
    st.info("👈 Please upload an image or select the sample image from the sidebar to begin.")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94A3B8; font-size: 0.85rem;'>"
    "YOLOv8 Streamlit Demo • Built with Ultralytics, PyTorch, & Streamlit Community Cloud"
    "</div>",
    unsafe_allow_html=True
)
