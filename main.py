# main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from inference import predict_image
import shutil
import os
import uuid

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith((".jpg", ".jpeg", ".png")):
        return JSONResponse(status_code=400, content={"detail": "Formato de imagen no soportado"})

    # Guardar temporalmente la imagen
    filename = f"{uuid.uuid4().hex}.jpg"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = predict_image(file_path)
        os.remove(file_path)  # Eliminar la imagen temporal después de predecir
        return result
    except Exception as e:
        os.remove(file_path)
        return JSONResponse(status_code=500, content={"detail": f"Error en la inferencia: {str(e)}"})
