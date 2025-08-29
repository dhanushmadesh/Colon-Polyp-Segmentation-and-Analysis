import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# === CONFIG ===
MODEL_PATH = "model/colon_classifier.h5"
IMAGE_PATH = r"C:\Users\dhanu\Desktop\MAJOR PROJECT\Polyp-Segmentation\2.jpeg"  # <--- Use raw string path
IMAGE_SIZE = (224, 224)

# === Load Model ===
model = load_model(MODEL_PATH)

# === Load & Preprocess Image ===
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"❌ Image not found at: {IMAGE_PATH}")

img = load_img(IMAGE_PATH, target_size=IMAGE_SIZE)
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# === Predict ===
prediction = model.predict(img_array)[0][0]
label = "Malignant" if prediction > 0.5 else "Benign"

print(f"\n🧠 Prediction: {label} (Confidence: {prediction:.2f})")
