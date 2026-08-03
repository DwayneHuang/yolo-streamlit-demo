# ⚡ Interactive YOLOv8 Streamlit Demo

A lightweight, memory-optimized, interactive Computer Vision demo powered by **Ultralytics YOLOv8** and **Streamlit**. Designed specifically to run seamlessly within **Streamlit Community Cloud**'s free tier resources.

---

## 🌟 Key Features

- **Multi-Task Support**:
  - 🎯 **Object Detection** (`yolov8n.pt`): Detects bounding boxes and labels across 80 COCO classes.
  - 🏷️ **Image Classification** (`yolov8n-cls.pt`): Predicts top classes across 1000 ImageNet categories with probability distribution.
  - ✂️ **Instance Segmentation** (`yolov8n-seg.pt`): Generates pixel-accurate mask overlays.
- **Automatic Hugging Face Integration**: Model weights are downloaded automatically from Hugging Face Hub (`ultralytics/yolov8`) upon server start and cached resourcefully.
- **Strict Memory Protection for Streamlit Cloud**:
  - Single-model caching via `@st.cache_resource`. Switching tasks immediately unloads the previous model and invokes garbage collection (`gc.collect()`).
  - 5MB file upload limit to prevent memory spikes.
  - Automatic image downscaling (maximum 1280px dimension) before running inference.
- **Interactive Controls**: Real-time confidence threshold slider, sample image preview, and detailed prediction result tables.

---

## 🚀 Quick Start (Local Setup)

### Prerequisites

- Python 3.9 - 3.11
- Git

### 1. Clone the repository
```bash
git clone https://github.com/your-username/yolo-streamlit-demo.git
cd yolo-streamlit-demo
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

The web application will open automatically in your browser at `http://localhost:8501`.

---

## ☁️ Deploying to Streamlit Community Cloud

Deploying this app to Streamlit Community Cloud is free and takes less than 2 minutes:

1. **Push your code to GitHub**:
   Ensure `app.py`, `requirements.txt`, `README.md`, and `assets/` are pushed to a public or private GitHub repository.

2. **Sign in to Streamlit Community Cloud**:
   Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with your GitHub account.

3. **Deploy App**:
   - Click **New App**.
   - Select your Repository, Branch (`main` or `master`), and set **Main file path** to `app.py`.
   - Click **Deploy!**

Streamlit Cloud will automatically install dependencies from `requirements.txt` and launch the app.

---

## 🛠️ Project Structure

```
.
├── app.py               # Main Streamlit application script
├── requirements.txt     # Python dependencies for deployment
├── README.md            # App documentation & deployment guide
├── YOLO_Streamlit_demo.md # Original specification prompt
└── assets/
    └── sample.jpg       # Pre-loaded sample image for instant testing
```

---

## 📄 License

MIT License. Free for personal, academic, and commercial use.
