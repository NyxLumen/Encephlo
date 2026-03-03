import os
import numpy as np
import tensorflow as tf
import torch
import cv2
import joblib
from tensorflow.keras.models import load_model, Model
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
# We force the CPU for inference to avoid memory conflicts if you don't have a massive GPU
# (Or set to 'cuda' if you want to risk it)
DEVICE = torch.device("cpu") 

class FusionEngine:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        print(f"🔥 FUSION ENGINE: Initializing from {models_dir}...")
        
        # 1. Load TensorFlow Models (DenseNet + EfficientNet)
        # (Assuming you have these saved as .h5 or .keras from a previous step)
        # If you don't have them yet, we will create placeholders or load the real ones.
        self.densenet = self._load_tf_model("densenet_headless.h5")
        self.efficientnet = self._load_tf_model("efficientnet_headless.h5")
        
        # 2. Load PyTorch ViT (The one currently training on Colab!)
        print("   - Loading Vision Transformer (PyTorch)...")
        self.vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit_model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k", 
            num_labels=4
        )
        
        # Load the weights you are about to download from Colab
        vit_path = os.path.join(models_dir, "vit_feature_extractor.pt")
        if os.path.exists(vit_path):
            # We load the weights into the internal 'vit' component
            self.vit_model.vit.load_state_dict(torch.load(vit_path, map_location=DEVICE))
            self.vit_model.to(DEVICE)
            self.vit_model.eval()
            print("     ✅ ViT Loaded Successfully.")
        else:
            print(f"     ⚠️ WARNING: {vit_path} not found. Using untuned ViT (Bad for accuracy!).")

        # 3. Load the SVM (The Judge)
        svm_path = os.path.join(models_dir, "svm_fusion_weights.pkl")
        if os.path.exists(svm_path):
            self.svm = joblib.load(svm_path)
            print("     ✅ SVM Fusion Layer Loaded.")
        else:
            self.svm = None
            print("     ⚠️ WARNING: SVM weights not found.")

    def _load_tf_model(self, filename):
        path = os.path.join(self.models_dir, filename)
        if os.path.exists(path):
            print(f"   - Loading TensorFlow Model: {filename}...")
            return load_model(path)
        else:
            print(f"   ⚠️ Model {filename} missing.")
            return None

    def preprocess_tf(self, image_path):
        """Standard preprocessing for CNNs"""
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (224, 224))
        img = img / 255.0  # Normalize 0-1
        img = tf.expand_dims(img, axis=0) # Batch dim
        return img

    def preprocess_torch(self, image_path):
        """Preprocessing for ViT"""
        image = Image.open(image_path).convert("RGB")
        inputs = self.vit_processor(images=image, return_tensors="pt")
        return inputs['pixel_values'].to(DEVICE)

    def generate_heatmap(self, image_path):
        """
        Generates the 3D Texture Heatmap using ScoreCAM on the DenseNet model.
        (Using TF because it's better for visual maps than ViT).
        """
        # Placeholder for the ScoreCAM logic we'll add next
        # For now, returns a dummy color map if real logic fails
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        heatmap = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        return heatmap

    def predict(self, image_path):
        print(f"🧠 Processing: {image_path}")
        
        # 1. Extract Features
        # TF Features
        tf_input = self.preprocess_tf(image_path)
        feat_dense = self.densenet.predict(tf_input, verbose=0) if self.densenet else np.zeros((1, 1024))
        feat_eff = self.efficientnet.predict(tf_input, verbose=0) if self.efficientnet else np.zeros((1, 1280))
        
        # PyTorch Features
        pt_input = self.preprocess_torch(image_path)
        with torch.no_grad():
            vit_out = self.vit_model.vit(pt_input).pooler_output
            feat_vit = vit_out.cpu().numpy() # Convert tensor to numpy

        # 2. Fuse (Concatenate)
        # Shape: [1, 1024 + 1280 + 768] = [1, 3072]
        fusion_vector = np.concatenate([feat_dense, feat_eff, feat_vit], axis=1)
        
        # 3. Final Verdict (SVM)
        if self.svm:
            prediction_idx = self.svm.predict(fusion_vector)[0]
            confidence = np.max(self.svm.predict_proba(fusion_vector))
        else:
            prediction_idx = 0
            confidence = 0.0

        classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        result = classes[prediction_idx]
        
        print(f"🎯 Diagnosis: {result} ({confidence*100:.2f}%)")
        return result, confidence