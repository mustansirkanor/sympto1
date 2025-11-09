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

# Rebuild model architecture
try:
    print("Loading model...")
    
    # Load base model WITHOUT imagenet weights
    base_model = MobileNetV2(
        input_shape=(128, 128, 3),
        include_top=False,
        weights=None  # Don't use ImageNet weights
    )
    base_model.trainable = False
    
    # Build model matching your training architecture
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Build model with dummy input to initialize weights
    model.build((None, 128, 128, 3))
    
    # Load weights
    model.load_weights('../models/malaria_mobilenetv2_model.keras')
    
    # Test with random data
    test_input = np.random.rand(1, 128, 128, 3).astype(np.float32)
    test_output = model.predict(test_input, verbose=0)
    print(f"✓ Model loaded successfully")
    print(f"✓ Test prediction: {test_output[0][0]:.4f}")
    
except Exception as e:
    print(f"✗ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    model = None

def preprocess_image(image):
    """Preprocess image exactly as in training"""
    # Resize to 128x128
    image = image.resize((128, 128))
    
    # Convert to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array
    img_array = np.array(image, dtype=np.float32)
    
    # Debug
    print(f"  Image shape: {img_array.shape}")
    print(f"  Value range before scaling: {img_array.min():.1f} to {img_array.max():.1f}")
    
    # Rescale to 0-1 (matching training)
    img_array = img_array / 255.0
    
    print(f"  Value range after scaling: {img_array.min():.4f} to {img_array.max():.4f}")
    
    # Add batch dimension
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
        print(f"\n{'='*60}")
        print(f"Processing: {file.filename}")
        
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Preprocess
        processed = preprocess_image(image)
        
        # Predict
        prediction = model.predict(processed, verbose=0)
        raw_probability = float(prediction[0][0])
        
        print(f"  Raw model output: {raw_probability:.6f}")
        
        # Determine class based on your training logic
        if raw_probability >= 0.5:
            predicted_class = "Uninfected"
            confidence = raw_probability * 100
            risk = "Low"
        else:
            predicted_class = "Parasitized"
            confidence = (1 - raw_probability) * 100
            risk = "High" if confidence > 80 else "Moderate"
        
        print(f"  Prediction: {predicted_class} ({confidence:.2f}%)")
        print(f"{'='*60}\n")
        
        return {
            "prediction": predicted_class,
            "confidence": round(confidence, 2),
            "risk_level": risk,
            "probabilities": {
                "Parasitized": round((1 - raw_probability) * 100, 2),
                "Uninfected": round(raw_probability * 100, 2)
            }
        }
        
    except Exception as e:
        print(f"✗ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
