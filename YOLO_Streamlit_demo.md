Build a lightweight interactive YOLO demo using Streamlit.

Requirements:

1. Framework:
- Use Streamlit only.
- The app must be suitable and deployable on Streamlit Community Cloud.

2. YOLO model:
- Use YOLOv8n (nano) from Ultralytics.
- Download model automatically from Hugging Face when the Streamlit server starts.
- Use @st.cache_resource to cache the loaded model and prevent reloading on every Streamlit rerun.
- Memory optimization is critical:
  - Do NOT load or cache multiple YOLO models simultaneously.
  - Never keep detection, segmentation, and classification models in memory at the same time.
  - When users switch tasks, unload the previous model before loading the new one.
  - Cache only the currently selected task model.

3. UI:
Create a simple interactive interface with a beautiful and clean Streamlit design style and responsive layout:
- Sidebar:
  - Task selector:
    - Object Detection
    - Image Classification
    - Segmentation (only if memory allows)
  - Confidence threshold slider.
- Main area:
  - Image uploader.
  - Display uploaded image.
  - Run inference button.
  - Display prediction results:
    - Bounding boxes for detection.
    - Class labels and confidence scores.
    - Masks for segmentation if supported.

4. Input and memory protection:
- Never reload YOLO model on every interaction.
- Limit uploaded image size to 5MB.
- Reject oversized files with a clear error message.
- Resize large images before inference (maximum dimension: 1280px).
- Avoid storing uploaded images and inference results permanently.
- Release temporary variables after inference.
- Optimize for Streamlit Cloud free CPU/RAM environment.

5. Model loading:
- Implement a clean model loading function:
  - Use @st.cache_resource.
  - Download weights automatically from Hugging Face if unavailable locally.
  - Prevent duplicate model initialization.

6. Deployment:
Include:
- requirements.txt
- README.md with Streamlit Cloud deployment instructions.

7. Code quality:
- Keep project structure simple.
- Add error handling for invalid images/files.
- Write clean production-style code suitable for a public AI demo.