# Real-Time Age Detection Deployment Pipeline

A production-grade, containerized machine learning microservice that serves an active deep regression model predicting ages from live camera streams.

## 🏗️ Architectural Overview
* **Concurrency Engine:** Integrated **Gunicorn** running thread pools to securely handle concurrent request pipelines without blocking.
* **Data Processing Layer:** Custom optimized base64 data URI parsing routine engineered with padding calculations compliant with strict data requirements.
* **Stability Filtering:** Implemented a rolling moving average lookback window buffer (`MAX_BUFFER = 15`) to smooth prediction values across sequential frames.
* **Deterministic Class Mapping:** Features calibration scaling coefficients mapping raw outputs into accurate demographic groups (`Child`, `Young Adult`, `Adult`, `Senior`).

## 🛠️ Production Tech Stack
* **Core framework:** Flask (v3.1.3)
* **Inference Engine:** TensorFlow CPU (v2.18.0) & Keras (v3.14.0)
* **Image Processing:** OpenCV Headless (v4.10.0.84)
* **Process Manager:** Gunicorn (v23.0.0)
* **Container Tool:** Docker (Debian/Python-Slim base)

## 🐳 Local Deployment & Verification

### Prerequisites
* Docker Engine installed and running
* Local model weight files loaded inside `saved_models/age_regression_final.keras`

### 1. Build the Production Container
```bash
docker build -t age-detection-app .
```

### 2. Launch the Application Service
```bash
docker run -d -p 5000:5000 --name age-detection-container age-detection-app
```

### 3. Review Live Server Telemetry
```bash
docker logs -f age-detection-container
```
The console will display professional, thread-safe timestamp logs confirming successful initialization loops:
```text
2026-05-14 04:23:17 [INFO] Starting gunicorn 23.0.0
2026-05-14 04:23:19 [INFO] Attempting to load model from: /app/saved_models/age_regression_final.keras
2026-05-14 04:23:20 [INFO] --- SUCCESS: MODEL LOADED ---
```
