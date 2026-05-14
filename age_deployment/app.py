from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import base64
from tensorflow.keras.layers import BatchNormalization
import os
import logging

# --- LOGGING CONFIGURATION ---
# This sets up clean timestamps and tags for your terminal outputs
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- CONFIGURATION & STABILITY BUFFERS ---
age_history = []
MAX_BUFFER = 15  # Number of frames used to calculate the rolling average

# --- COMPATIBILITY WRAPPER ---
class CompatibleBatchNormalization(BatchNormalization):
    def __init__(self, **kwargs):
        forbidden_keys = ['renorm', 'renorm_clipping', 'renorm_momentum', 'synchronized']
        for key in forbidden_keys:
            kwargs.pop(key, None)
        super().__init__(**kwargs)

# --- MODEL INITIALIZATION ---
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'saved_models', 'age_regression_final.keras')

logger.info(f"Attempting to load model from: {model_path}")

reg_model = tf.keras.models.load_model(
    model_path,
    custom_objects={'BatchNormalization': CompatibleBatchNormalization},
    compile=False
)
logger.info("--- SUCCESS: MODEL LOADED ---")

def preprocess(uri):
    try:
        if not isinstance(uri, str):
            logger.warning("Preprocessing Error: URI is not a string")
            return None
            
        # 1. Clean the base64 string properly
        if ',' in uri:
            encoded_data = uri.split(',', 1)[1].strip()
        else:
            encoded_data = uri.strip()
        
        # 2. Add padding if necessary (Numpy 2.0 is strict about this)
        missing_padding = len(encoded_data) % 4
        if missing_padding:
            encoded_data += '=' * (4 - missing_padding)
        
        # 3. Decode to bytes
        image_bytes = base64.b64decode(encoded_data)
        
        # 4. Convert to OpenCV image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.warning("Preprocessing Error: OpenCV could not decode the image")
            return None
        
        # 5. Resize and Color Space conversion
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 6. Normalize to [-1, 1] range
        img_tensor = (img.astype('float32') / 127.5) - 1.0
        
        # Return with batch dimension
        return np.expand_dims(img_tensor, axis=0)
        
    except Exception as e:
        logger.error(f"Preprocessing Exception: {e}")
        return None
    
@app.route('/')
def index():
    """Renders the core UI frontend interface."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles real-time webcam frames, scales metrics, and maps age classes."""
    global age_history
    if reg_model is None: 
        logger.error("Predict Route Error: Regression inference model is offline")
        return jsonify({'error': 'Regression inference model is offline'}), 500

    try:
        content = request.get_json(silent=True)
        if not content or 'image' not in content: 
            return jsonify({'error': 'Missing valid image payload'}), 400
            
        frame = preprocess(content.get('image', ''))
        if frame is None: 
            return jsonify({'error': 'Invalid image data'}), 400

        # 1. Base Model Inference
        raw_prediction = reg_model.predict(frame, verbose=0)
        raw_age = float(raw_prediction.item())
        
        # 2. Calibration Tuning
        calibrated_age = (raw_age * 0.87) + 1.0 
        
        # 3. Rolling Moving Average Filter
        age_history.append(calibrated_age)
        if len(age_history) > MAX_BUFFER: 
            age_history.pop(0)
        final_age = sum(age_history) / len(age_history)

        # 4. Strict Deterministic Mapping Thresholds
        if final_age < 18:
            group_label = "Child"
        elif final_age < 32:
            group_label = "Young Adult"
        elif final_age < 50:
            group_label = "Adult"
        else:
            group_label = "Senior"

        # Production thread-safe logging output
        logger.info(f"Metrics -> Raw: {round(raw_age, 1)} | Calibrated: {round(final_age, 1)} | Class: {group_label}")
        
        return jsonify({
            'age': round(final_age, 1), 
            'group': group_label
        })
        
    except Exception as e:
        logger.error(f"Runtime Route Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Fallback development environment target
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
