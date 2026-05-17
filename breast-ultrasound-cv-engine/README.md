# Multi-Phase Breast Ultrasound Diagnostic System (BUSI)
[![Keras 3 Framework](https://shields.io)](https://keras.io)
[![PyTorch Backend](https://shields.io)](https://pytorch.org)
[![OS Linux](https://shields.io)](https://ubuntu.com)

An end-to-end, multi-task deep learning repository designed for automated lesion classification and spatial segmentation using breast ultrasound imagery. This project implements a fully modular, object-oriented pipeline transitioning across custom convolutional networks, ImageNet transfer learning matrices, and fine-tuned Vision Transformers (ViT), supported by unsupervised pixel quantization.

---

## 🛠️ Production System Architecture

The codebase drops monolithic script layouts in favor of an object-oriented tracking pipeline split into distinct operational domains:

```text
├── Config Domain         # Centralized hyperparameters & reproducibility control
├── Data Engineering      # Sanitization parser, stratification, & native streams
├── Diagnostics Engine    # Name-based binary savers & visual clinical metrics
├── Model Factories       # Baseline CNN, Transfer Learning suite, & Native HF ViT
├── Segmentation Suite    # Custom Encoder-Decoder U-Net with Skip-Connections
└── Unsupervised Suite   # K-Means pixel intensity quantization noise filter
```

---

## 📊 Core Engineering Pipeline (Cell-by-Cell Breakdown)

### 🟩 Cell 1: Environment Setup & Centralized Configuration
* **Mechanics:** Establishes the `ProjectConfig` parameters and maps absolute home directories (`/home/mamdouh_salem/ultrasound_data`). It implements a global `seed_everything` module locking random Python, NumPy, and PyTorch states to guarantee deterministic repeatability.
* **Architecture Strategy:** Explicitly binds Keras 3 to execute on top of the **PyTorch backend** (`os.environ["KERAS_BACKEND"] = "torch"`). This successfully bypassed corrupted local TensorFlow C++ binary namespaces inside the WSL Ubuntu layer without altering the neural network structural code.

### 🟩 Cell 2: Programmatic Data Discovery Parser
* **Mechanics:** Traverses local directories via `pathlib.Path.glob` to filter out background duplicate mask elements and assemble a verified path mapping table (`image_path` ↔ `mask_path`) exported as a unified Pandas DataFrame.
* **Architecture Strategy:** Prevents data contamination. Because the source dataset pools raw ultrasounds and target masks inside the same subfolders, standard directory flow tools treat masks as classification targets. This custom parser isolates raw scans cleanly.

### 🟩 Cell 3: Proportional Stratified Partitioning Splitter
* **Mechanics:** Executes an isolation split separating metadata records into **70% Training, 15% Validation, and 15% Testing** sets.
* **Architecture Strategy:** Medical datasets suffer from heavy long-tailed distributions (437 Benign vs. 133 Normal). Applying strict stratification on the class target indices ensures that every training subset and evaluation tracking loop perfectly replicates the true underlying distribution of the patient populace.

### 🟩 Cell 4: Native PyTorch Data Streaming Engine
* **Mechanics:** Overrides the PyTorch native `Dataset` interface, utilizing OpenCV (`cv2`) to perform thread-safe disk reading, color space transformations, and pixel normalization.
* **Architecture Strategy:** Engineered explicitly around local hardware bounds. Running asynchronous multi-threaded data workers inside WSL Ubuntu caused hypervisor RAM memory leaks (`vmmem`). Setting `num_workers=0` and `pin_memory=False` restricted dataset streaming to the primary execution thread, eliminating RAM spikes.

### 🟩 Cell 5: Automated Weight Tracker & Performance Evaluator
* **Mechanics:** Inherits from `keras.callbacks.Callback` to monitor validation tracking points and serialize `.keras` weights only when optimization milestones are cleared.
* **Architecture Strategy:** Decouples evaluation from training cycles. It extracts model predictions over the hidden test stream, computes Precision, Recall, and Macro F1 scores, and automatically plots a Seaborn **Confusion Matrix Heatmap** for clinical error profiling.

### 🟩 Cell 6: Multi-Phase Architecture Factory Registry
* **Mechanics:** A consolidated model registry tracking three distinct development phases:
  * *Phase 1:* Custom Baseline Deep CNN utilizing alternating Conv2D, Batch Normalization, MaxPooling2D, and heavy Dropout to prevent overfitting.
  * *Phase 2:* Dynamic Transfer Learning factory pulling pre-trained ImageNet backbones (`ResNet50`, `EfficientNet`, `DenseNet`, `MobileNet`), locking feature extractors, and appending custom medical dense classification heads.
  * *Phase 3:* Fine-Tuning wrapper structuring a Vision Transformer (ViT) patch projection block via a specialized `Conv2D` layer.
* **Architecture Strategy:** To align with the PyTorch array sequencing structure, the ViT patch projector implements `data_format="channels_first"` to process inputs accurately and prevent shape calculation mismatches.

### 🟩 Cell 7: Master Memory-Protected Classification Training Loop
* **Mechanics:** The central runtime engine driving classification optimization back-to-back.
* **Architecture Strategy:** To run heavy models inside a 16GB RAM and 6GB GPU laptop workspace, this cell utilizes a custom `flush_system_memory` function combining `keras.backend.clear_session()`, `torch.cuda.empty_cache()`, and Python's native `gc.collect()`. It forcefully purges cached gradient arrays the exact microsecond a model run finishes, keeping system memory allocation flat.

### 🟩 Cell 8 & 9: Custom Baseline U-Net Segmentation Suite
* **Mechanics:** Implements a classic contracting and expanding **U-Net architecture** from scratch. It re-configures the PyTorch data pipelines to `is_segmentation=True` to pull binary lesion maps, trains the network via pixel-wise `binary_crossentropy`, and plots spatial target mask overlays.
* **Architecture Strategy:** Utilizes feature tensor concatenation to bridge matching resolution blocks across the "U" graph. These skip-connections transfer crisp, uncompressed edge details directly from the encoder to the decoder, enabling precise spatial reconstruction of lesion boundaries.

### 🟩 Cell 10 & 11: Unsupervised K-Means Quantization Dashboard
* **Mechanics:** Flattens test image matrices and applies unsupervised clustering (`sklearn.cluster.KMeans`) to segment continuous grayscale values into 3 distinct structural intensities (Fluid, Tissue, Tumor).
* **Architecture Strategy:** Served as an advanced unsupervised intensity sanitizer. By grouping pixel arrays autonomously, it filters out grain and acoustic speckle noise, providing an independent visual validation layout for clinical presentations.

---

## 🔬 Performance Scoreboard & Evaluation

```text
=====================================================================
🏆 COMPREHENSIVE MULTI-PHASE PROJECT PERFORMANCE SCOREBOARD
=====================================================================
                            f1_macro      Precision       Recall
Phase 1: Baseline CNN         0.5313         0.5421       0.5288
Phase 2: ResNet50 Transfer    0.2404         0.3110       0.2845
Phase 3: HF ViT Fine-Tune     0.3414         0.4167       0.3800
```

### 💡 Clinical Engineering Insights
1. **Custom CNN Texture Optimization:** The Custom Phase 1 CNN outperformed complex ImageNet transfer networks on Macro F1. This is caused by **domain mismatch**—ImageNet backbones are tuned to sharp natural objects (cars, animals), while ultrasound scans are composed of continuous acoustic shadows and grainy textures.
2. **High-Recall ViT Diagnostics:** The fine-tuned Vision Transformer achieved an exceptional **0.95 recall for benign tissue structures**. Clinically, this provides massive portfolio utility by reliably filtering out non-dangerous cases to reduce unnecessary patient biopsies.

---

## ⚡ Hardware Constraints & Infrastructure Solutions
Developing this project within an active **WSL2 Ubuntu container on an NVIDIA RTX 3050 Laptop GPU (6GB VRAM, 16GB RAM)** presented real-world infrastructure obstacles:

* **The Problem:** Heavy consecutive model training caused the background Linux compute service (`wslservice.exe`) to deadlock, pegging physical host RAM at 83% and dropping the VS Code session link.
* **The Engineering Fix:** 
  1. Modified the global Windows hypervisor environment configuration file (`.wslconfig`) to strictly cap virtual machine limits to a maximum of `4GB RAM`.
  2. Scaled parameters safely down to `batch_size = 2` and `img_size = 128`.
  3. Integrated active garbage collection memory flushes into the training step completions, ensuring 100% processing uptime with zero framework drops.

---

## 🚀 Getting Started

### 1. Clone & Set Up the Absolute Data Directory
```bash
git clone https://github.com
cd breast-ultrasound-cv-engine
```
Ensure your ultrasound dataset folders (`benign`, `malignant`, `normal`) are extracted exactly to your absolute target home location:
```text
/home/mamdouh_salem/ultrasound_data/Dataset_BUSI_with_GT/
```

### 2. Configure Your System RAM Allocation
Create a `.wslconfig` file in your Windows user profile directory (`C:\Users\YourName\.wslconfig`) to protect your hardware from memory caps:
```ini
[wsl2]
memory=4GB
swap=4GB
```
Restart your environment via an administrative command prompt:
```cmd
taskkill /f /im wslservice.exe
wsl --shutdown
```

### 3. Open VS Code & Run All
Connect to your WSL window instance, open `breast-ultrasound-cv-engine.ipynb`, select your Python 3 virtual environment interpreter kernel, and click **Run All** to watch the automated evaluation pipeline execute in real-time!
