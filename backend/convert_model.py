import tensorflow as tf
from tensorflow import keras
import os

# Try different loading methods
print("Attempting to load and convert model...")

try:
    # Method 1: Load with custom objects
    print("\nMethod 1: Loading with compile=False...")
    model = keras.models.load_model('backend\models\malaria_mobilenetv2_model.h5', compile=False)
    
    # Save in new format
    print("Saving in SavedModel format...")
    model.save('../models/malaria_model_saved', save_format='tf')
    print("✓ Model converted successfully to: models/malaria_model_saved/")
    
except Exception as e:
    print(f"✗ Method 1 failed: {e}")
    
    try:
        # Method 2: Build model manually and load weights
        print("\nMethod 2: Loading weights only...")
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
        from tensorflow.keras.models import Sequential
        
        # Recreate MobileNetV2 architecture
        base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights=None)
        
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(1, activation='sigmoid')
        ])
        
        # Load weights
        model.load_weights('backend\models\malaria_mobilenetv2_model.h5')
        
        # Save in new format
        model.save('../models/malaria_model_saved', save_format='tf')
        print("✓ Model weights loaded and saved to: models/malaria_model_saved/")
        
    except Exception as e2:
        print(f"✗ Method 2 failed: {e2}")
        print("\nPlease provide your model training code so we can rebuild it correctly.")
