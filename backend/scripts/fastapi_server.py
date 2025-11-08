from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers
from tensorflow import keras

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rebuild your EXACT architecture (without BatchNorm for compatibility)
try:
    # Load the base MobileNetV2
    base_model = MobileNetV2(
        input_shape=(128, 128, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    # Rebuild your model structure (simplified - no BatchNorm)
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Load ONLY the weights from your trained model
    model.load_weights('../models/malaria_mobilenetv2_model.keras')
    
    print("✓ Malaria model loaded successfully!")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None

def preprocess_image(image):
    """Preprocess image exactly as in training"""
    image = image.resize((128, 128))
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image, dtype=np.float32)
    img_array = img_array / 255.0  # Rescale like your training
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.get("/")
def home():
    return {
        "message": "Malaria Prediction API",
        "model_loaded": model is not None
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        processed = preprocess_image(image)
        
        # Predict
        prediction = model.predict(processed, verbose=0)
        probability = float(prediction[0][0])
        
        # Your class logic from training code
        # class_mode='binary' in your code means:
        # 0 = Parasitized, 1 = Uninfected
        if probability >= 0.5:
            predicted_class = "Uninfected"
            confidence = probability * 100
            risk = "Low"
        else:
            predicted_class = "Parasitized"
            confidence = (1 - probability) * 100
            risk = "High" if confidence > 80 else "Moderate"
        
        return {
            "prediction": predicted_class,
            "confidence": round(confidence, 2),
            "risk_level": risk,
            "probabilities": {
                "Parasitized": round((1 - probability) * 100, 2),
                "Uninfected": round(probability * 100, 2)
            }
        }
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
