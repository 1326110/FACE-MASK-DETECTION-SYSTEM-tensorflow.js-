import os
import tensorflowjs as tfjs
from tensorflow.keras.models import load_model

model_path = "mask_detector.h5"
output_dir = "web_model"

if not os.path.exists(model_path):
    print(f"[ERROR] Could not find {model_path}. Make sure to train the model first by running train.py")
    exit(1)

print("[INFO] Loading trained Keras model...")
model = load_model(model_path)

print(f"[INFO] Converting model to TensorFlow.js format (Float16 Quantized) in '{output_dir}'...")
# Float16 Quantization shrinks the model by 50% providing massive loading speed improvements
# and better performance over WebGL networks.
tfjs.converters.save_keras_model(model, output_dir, quantization_dtype_np="float16")
print("[SUCCESS] Conversion complete! Model size has been highly optimized.")
