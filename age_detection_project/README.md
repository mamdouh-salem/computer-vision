# Real-Time Age Estimation: Adaptive MobileNetV2 System

A professional Computer Vision suite featuring custom training pipelines for dual-head age estimation. This project implements a fine-tuned MobileNetV2 architecture with custom high-level Python classes for data orchestration and model lifecycle management.

## 🧠 Engineering Highlights

### **Custom Architecture & Fine-Tuning**
We utilized a MobileNetV2 backbone, stripping the top layers to implement task-specific heads:
* **Regression Head:** Final layer uses a single neuron with a **Linear activation** to predict continuous age values.
* **Classification Head:** Final layer uses a **Softmax activation** (Multiple neurons) to categorize ages into discrete bins.

### **Training Dynamics & Optimization**
* **Early Convergence (Regression):** The regressor reached its optimum state in just **8 epochs**, far ahead of the scheduled 15, demonstrating highly efficient weight initialization.
* **Categorical Stability (Classification):** Required **13 epochs** to stabilize, achieving robust accuracy across diverse age groups.

## 🛠️ Modular OOP Architecture
The project is built on custom Python classes to ensure scalability and clean code:

* **`DataAugmentor`**: Handles real-time geometric transformations and brightness adjustments to simulate various camera lighting conditions.
* **`DataLoader`**: A high-performance generator optimized to handle the **66,000+ image** dataset without memory overflow.
* **`ModelTrainer`**: A unified class for regression and classification that manages training loops, callbacks, and history logging.
* **`ModelEvaluator`**: Automates the generation of Loss/Accuracy curves and Weight Distribution plots (Mean, Std Dev) for the first convolutional layers.

## 📂 Project Structure
* `AGE_PROJECT_FINAL.ipynb`: Full implementation of the OOP pipeline.
* `/saved_models`: 
    * `age_regression_final.keras`: (Converged @ 8 Epochs)
    * `age_classifier_final.keras`: (Converged @ 13 Epochs)

## 📊 Dataset Specifications
* **Volume:** ~66,000 face images.
* **Scope:** Diverse range of ages, genders, and ethnicities.
* **Handling:** All data is processed via our custom `DataLoader` to manage the massive file count efficiently.

---
**Author:** Mamdouh Salem  
**Focus:** Computer Vision | Deep Learning Architecture | Pythonic ML Systems