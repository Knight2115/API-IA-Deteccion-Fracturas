import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import os
import sys

# Asegurar que fracture_detector esté en el path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fracture_detector.model import EdgeExtractor  # necesario para cargar el modelo

# Cargar modelo
MODEL_PATH = "models/best_model_v3.keras"
model = load_model(MODEL_PATH, custom_objects={"EdgeExtractor": EdgeExtractor})

def predict_image(image_path):
    """Realiza la inferencia en una imagen JPG dada"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("No se pudo leer la imagen")

    img = cv2.resize(img, (224, 224))
    img_rgb = img.astype("float32") / 255.0

    # Generar bordes
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges = cv2.resize(edges, (224, 224))
    edges = edges.astype("float32") / 255.0
    edges = np.stack([edges] * 3, axis=-1)

    # Concatenar RGB + edges
    input_img = np.concatenate([img_rgb, edges], axis=-1)  # (224, 224, 6)
    input_img = np.expand_dims(input_img, axis=0)  # (1, 224, 224, 6)

    pred = model.predict(input_img)[0][0]
    label = "Fractura" if pred >= 0.5 else "No Fractura"
    confidence = float(pred) if pred >= 0.5 else 1 - float(pred)

    return {
        "Etiqueta": label,
        "Probabilidad": round(confidence, 4),
        "Nivel de Confianza": (
            "Alta" if confidence > 0.8 or confidence < 0.2 else "Media"
        )
    }
