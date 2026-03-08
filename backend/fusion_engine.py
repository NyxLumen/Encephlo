import os
import numpy as np
import tensorflow as tf
import torch
import joblib
import cv2
import base64
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image

DEVICE = torch.device("cpu") 

class FusionEngine:
    def __init__(self, models_dir="models"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(base_dir, models_dir)
        print(f"🔥 FUSION ENGINE: Initializing from {self.models_dir}...")
        
        # 1. Load TensorFlow Model (DenseNet)
        dense_path = os.path.join(self.models_dir, "densenet_headless.keras")
        if os.path.exists(dense_path):
            tf_model = tf.keras.models.load_model(dense_path, compile=False)
            
            # Decapitated 1D Extractor (For SVM)
            self.densenet = tf.keras.Model(inputs=tf_model.input, outputs=tf_model.layers[-2].output)
            
            # SPATIAL EXTRACTOR (For Heatmap)
            # Find the last convolutional layer (4D output: batch, height, width, channels)
            last_conv_layer = None
            for layer in reversed(tf_model.layers):
                if len(layer.output_shape) == 4:
                    last_conv_layer = layer
                    break
            
            self.cam_model = tf.keras.Model(inputs=tf_model.input, outputs=last_conv_layer.output)
            print("   ✅ DenseNet & CAM Extractor Loaded.")
        else:
            print(f"   ⚠️ ERROR: {dense_path} missing.")
            self.densenet = None
            self.cam_model = None

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

    def generate_heatmap(self, image_path):
        if not self.cam_model:
            return None
            
        # Extract spatial features (usually 7x7x1024 or similar)
        img_array = self.preprocess_tf(image_path)
        conv_outputs = self.cam_model.predict(img_array, verbose=0)[0]
        
        # Average the channels to find where the CNN is looking
        heatmap = np.mean(conv_outputs, axis=-1)
        
        # Normalize between 0 and 1
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
            
        # Colorize and overlay
        original_img = cv2.imread(image_path)
        original_img = cv2.resize(original_img, (224, 224))
        
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        
        # COLORMAP_INFERNO gives that high-tech medical look (black/purple/orange/yellow)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_INFERNO)
        
        # Blend original scan with the heatmap (60% MRI, 40% Heatmap)
        superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)
        
        # Encode to Base64
        _, buffer = cv2.imencode('.jpg', superimposed_img)
        b64_string = base64.b64encode(buffer).decode('utf-8')
        
        return f"data:image/jpeg;base64,{b64_string}"

    def predict(self, image_path):
        print(f"🧠 Processing: {image_path}")
        
        # 1. Feature Extraction
        tf_input = self.preprocess_tf(image_path)
        feat_dense = self.densenet.predict(tf_input, verbose=0) if self.densenet else np.zeros((1, 1024))
        
        pt_input = self.preprocess_torch(image_path)
        with torch.no_grad():
            outputs = self.vit_model.vit(pt_input)
            feat_vit = outputs.last_hidden_state[:, 0, :].numpy()

        fusion_vector = np.concatenate([feat_dense, feat_vit], axis=1)
        
        # 2. Classification
        if self.svm:
            prediction_idx = self.svm.predict(fusion_vector)[0]
            confidence = np.max(self.svm.predict_proba(fusion_vector))
        else:
            prediction_idx, confidence = 0, 0.0

        classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        result = classes[prediction_idx]
        
        # 3. Generate Visuals
        b64_heatmap = self.generate_heatmap(image_path)
        
        print(f"🎯 Diagnosis: {result} ({confidence*100:.2f}%)")
        # Now returning 3 items: diagnosis, confidence, and the encoded image
        return result, confidence, b64_heatmap