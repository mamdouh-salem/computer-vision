<<<<<<< HEAD
---
title: Real-Time Age Detection
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: age_deployment/app.py
app_port: 5000
pinned: false
---


=======
>>>>>>> 99a898649caaa7f730acb54c626ddc97715a483e
# 👁️ Computer Vision Portfolio

Welcome to my professional Computer Vision architecture and deep learning portfolio. This repository highlights production-ready applications, advanced transformer models, multi-framework engines, and containerized deployment pipelines.

## 📁 Portfolio Projects

### 1. 🧬 Breast Ultrasound Tissue Classification & Semantic Segmentation
* **Location:** `/ultrasound_analysis`
* **Technologies:** Python, Keras 3 (Mapped to PyTorch Backend Engine), Hugging Face Transformers (`ViT`, `SegFormer`), ResNet-50, OpenCV, NumPy, WSL2 (Ubuntu)
* **Core Functionality:** Processes multi-class medical ultrasound scan streams (`benign`, `malignant`, `normal`) and performs high-fidelity pixel-by-pixel tumor contour tracing.
* **Architecture Highlight:** Orchestrates a cross-framework pipeline utilizing Keras 3 abstractions running natively on a stable PyTorch GPU graph. Integrates global Self-Attention transformers via Hugging Face Hub, deploying dual-axis bilinear up-sampling sequences to prevent micro-gradient trace collapse on consumer hardware (NVIDIA RTX 3050).

### 2. 🚀 Production Real-Time Age Detection & Deployment
* **Location:** `/age_deployment`
* **Technologies:** Python, TensorFlow (Keras 3), OpenCV, Flask, Gunicorn, Docker
* **Core Functionality:** Processes live webcam video frames via a base64 streaming data pipeline, runs real-time calibration tuning, applies a rolling moving average filter for prediction stability, and classifies ages deterministically.
* **Architecture Highlight:** Containerized with Docker and served via Gunicorn for high-throughput, multi-threaded request handling.

### 3. 🧪 Age Model Training & Core Development
* **Location:** `/age_detection`
* **Technologies:** TensorFlow, Deep CNNs, Data Augmentation
* **Core Functionality:** Explores underlying dataset pre-processing, convolutional layers hyperparameter tuning, model training scripts, and regression evaluation matrices.

---
*Maintained by Mamdouh Salem. Built using VS Code and WSL2 (Ubuntu).*

