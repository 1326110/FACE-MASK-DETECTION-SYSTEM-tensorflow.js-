import os
import tensorflowjs as tfjs
from tensorflow.keras.models import load_model

model_path = "mask_detector.h5"
output_dir = "web_model"

if not os.path.exists(model_path):
    print(f"[ERROR] Could not find {model_path}. Run train.py first")
    exit(1)

print("[INFO] Loading trained Keras model...")
model = load_model(model_path, compile=False)

print(f"[INFO] Converting model to TensorFlow.js format in '{output_dir}'...")

tfjs.converters.save_keras_model(model, output_dir)

print("[SUCCESS] Conversion complete! 🚀")
