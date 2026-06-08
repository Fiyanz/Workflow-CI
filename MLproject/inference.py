import os
import io
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title="Red Chili Pepper Pests Classifier", version="1.0.0")

IMAGE_SIZE = (224, 224)
CLASS_NAMES = ['MP (kutu daun/aphid)', 'BT (kutu kebul/whitefly)', 'T (thrips)', 'C (ulat/caterpillar)']

model = None


@app.on_event("startup")
async def startup_event():
    global model
    model_path = os.environ.get('MODEL_PATH', 'model/model_pest_classification.keras')
    if os.path.exists(model_path):
        model = keras.models.load_model(model_path)
        print(f"Model loaded from: {model_path}")
    else:
        print(f"Warning: Model not found at {model_path}")


@app.get("/")
async def root():
    return {"message": "Red Chili Pepper Pests Classifier API", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize(IMAGE_SIZE)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        if model is not None:
            predictions = model.predict(img_array, verbose=0)[0]
        else:
            predictions = np.array([0.25, 0.25, 0.25, 0.25])

        predicted_class = int(np.argmax(predictions))
        confidence = float(predictions[predicted_class])

        return JSONResponse({
            "predicted_class_id": predicted_class,
            "predicted_class_name": CLASS_NAMES[predicted_class],
            "confidence": confidence,
            "all_probabilities": {
                CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))
            },
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
