import tensorflow as tf
import numpy as np
import cv2
import time
# from sklearn.svm import SVC  <-- Sid will use this later
# import joblib <-- Sid will use this to load his trained SVM

# --- SID: YOUR INSTRUCTIONS ---
# 1. Put your trained EfficientNet and DenseNet models in backend/models/
# 2. We are NOT using them to predict. We are using them to extract features.
# 3. Write the SVM logic to classify the concatenated vector.

class FeatureFusionEngine:
    def __init__(self):
        print("⚙️ Initializing Neural Fusion Engine...")
        
        # Load Sid's models (Tell him to update these paths)
        try:
            self.effnet = tf.keras.models.load_model("../models/clean_effnet.h5", compile=False)
            self.densenet = tf.keras.models.load_model("../models/clean_densenet.h5", compile=False)
            # self.svm_classifier = joblib.load("../models/svm_fusion_weights.pkl") # Sid uncomments this later
            print("✅ Models loaded successfully.")
        except Exception as e:
            print(f"⚠️ Warning: Models not found. Using dummy mode for testing. ({e})")
            self.effnet = None
            self.densenet = None

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """Decodes the image and applies your Otsu Brain Crop."""
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # --- SID: Put Abel's Otsu Thresholding crop code here ---
        # For now, we just resize to 224x224
        img_resized = cv2.resize(img, (224, 224))
        
        return img_resized

    def extract_and_fuse(self, image_bytes: bytes) -> dict:
        """
        This is the ONLY function main.py will call.
        It must return the exact dictionary format below.
        """
        start_time = time.time()
        
        img = self.preprocess_image(image_bytes)
        img_batch = np.expand_dims(img, axis=0)

        # If models aren't loaded yet, return dummy data so the React app doesn't crash
        if self.effnet is None:
            time.sleep(1) # Fake processing time
            return {
                "diagnosis": "Glioma (Dummy)",
                "confidence": 99.9,
                "inference_time_ms": round((time.time() - start_time) * 1000, 2)
            }

        # --- SID: FEATURE EXTRACTION LOGIC ---
        # 1. Pass image through EfficientNet, get the raw 1D vector (e.g., 1280 numbers)
        # eff_features = self.effnet_feature_extractor.predict(img_batch)
        
        # 2. Pass image through DenseNet, get the raw 1D vector (e.g., 1024 numbers)
        # dense_features = self.densenet_feature_extractor.predict(img_batch)
        
        # 3. Concatenate: [1280] + [1024] = [2304] length vector
        # fused_vector = np.concatenate([eff_features, dense_features], axis=1)
        
        # 4. Pass fused vector to SVM to get the final diagnosis
        # final_prediction = self.svm_classifier.predict(fused_vector)
        # confidence = self.svm_classifier.predict_proba(fused_vector).max() * 100
        
        # NOTE: Sid, you need to write the actual extraction code above. 
        # For now, returning dummy logic so Adi's frontend keeps working.
        
        end_time = time.time()
        
        return {
            "diagnosis": "Meningioma", # Replace with final_prediction
            "confidence": 85.5,        # Replace with confidence
            "inference_time_ms": round((end_time - start_time) * 1000, 2)
        }

# Instantiate the engine so it loads the models into RAM exactly once at startup
fusion_engine = FeatureFusionEngine()