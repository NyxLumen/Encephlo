import os
import numpy as np
import tensorflow as tf
import torch
import joblib
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import cv2

# Force CPU inference for stability on your laptop
DEVICE = torch.device("cpu") 

class FusionEngine:
    def __init__(self, models_dir="models"):
        # Make the path absolute so it never gets lost again
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(base_dir, models_dir)
        print(f"🔥 FUSION ENGINE: Initializing from {self.models_dir}...")
        
        # 1. Load TensorFlow Model (DenseNet)
        dense_path = os.path.join(self.models_dir, "densenet_headless.keras")
        if os.path.exists(dense_path):
            tf_model = tf.keras.models.load_model(dense_path, compile=False)
            # DYNAMIC DECAPITATION: Strip the final layer to get the 1024 vector
            self.densenet = tf.keras.Model(inputs=tf_model.input, outputs=tf_model.layers[-2].output)
            print("   ✅ DenseNet (Decapitated) Loaded.")
        else:
            print(f"   ⚠️ ERROR: {dense_path} missing.")
            self.densenet = None

        # 2. Load PyTorch ViT
        self.vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=4)
        
        vit_path = os.path.join(self.models_dir, "vit_feature_extractor.pt")
        if os.path.exists(vit_path):
            self.vit_model.vit.load_state_dict(torch.load(vit_path, map_location=DEVICE))
            self.vit_model.to(DEVICE)
            self.vit_model.eval()
            print("   ✅ ViT Loaded Successfully.")
        else:
            print(f"   ⚠️ WARNING: {vit_path} not found.")

        # 3. Load SVM
        svm_path = os.path.join(self.models_dir, "svm_fusion_weights.pkl")
        if os.path.exists(svm_path):
            self.svm = joblib.load(svm_path)
            print("   ✅ SVM Fusion Layer Loaded.")
        else:
            self.svm = None
            print("   ⚠️ WARNING: SVM weights not found.")

    def preprocess_tf(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (224, 224))
        img = img / 255.0 
        img = tf.expand_dims(img, axis=0) 
        return img

    def preprocess_torch(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.vit_processor(images=image, return_tensors="pt")
        return inputs['pixel_values'].to(DEVICE)

    def predict(self, image_path):
        print(f"🧠 Processing: {image_path}")
        
        # 1. Extract TF Features (1024 dims)
        if self.densenet:
            tf_input = self.preprocess_tf(image_path)
            feat_dense = self.densenet.predict(tf_input, verbose=0)
        else:
            feat_dense = np.zeros((1, 1024))
        
        # 2. Extract PyTorch Features (768 dims)
        pt_input = self.preprocess_torch(image_path)
        with torch.no_grad():
            outputs = self.vit_model.vit(pt_input)
            # Grab the [CLS] token just like we did in SVM training
            feat_vit = outputs.last_hidden_state[:, 0, :].numpy()

        # 3. Fuse to 1792 dims
        fusion_vector = np.concatenate([feat_dense, feat_vit], axis=1)
        
        # 4. Final Verdict (SVM)
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