import os
import numpy as np
import tensorflow as tf
import torch
import joblib
import cv2
import base64
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image

# Force CPU inference for stability
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
            # Decapitated 1D Extractor for SVM ONLY. We no longer use this for heatmaps.
            self.densenet = tf.keras.Model(inputs=tf_model.input, outputs=tf_model.layers[-2].output)
            print("   ✅ DenseNet Feature Extractor Loaded.")
        else:
            print(f"   ⚠️ ERROR: {dense_path} missing.")
            self.densenet = None

        # 2. Load PyTorch ViT
        self.vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        # Force the config to ALWAYS track attention layers
        self.vit_model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k", 
            num_labels=4,
            output_attentions=True 
        )
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

    def generate_vit_attention(self, image_path, pt_input):
        # 1. Forward pass requesting attention weights
        with torch.no_grad():
            # Explicitly declare pixel_values
            outputs = self.vit_model.vit(pixel_values=pt_input, output_attentions=True)
            
        # Safety catch just in case
        if not outputs.attentions:
            print("⚠️ WARNING: Transformer failed to output attention weights.")
            return None
            
        # 2. Grab the attention weights from the very last Transformer block
        attention = outputs.attentions[-1] 
        
        # 3. Average the attention across all Transformer heads
        attention_heads = attention.mean(dim=1) 
        
        # 4. The [CLS] token is index 0. We map how much attention it paid to the 196 image patches
        cls_attention = attention_heads[0, 0, 1:] 
        
        # 5. Reshape the 196 patches back into a 14x14 spatial grid
        grid_size = int(np.sqrt(cls_attention.shape[0]))
        attention_map = cls_attention.reshape(grid_size, grid_size).numpy()
        
        # 6. Normalize the map
        attention_map = np.maximum(attention_map, 0)
        if np.max(attention_map) != 0:
            attention_map /= np.max(attention_map)
            
        # 7. Resize, colorize, and superimpose
        original_img = cv2.imread(image_path)
        original_img = cv2.resize(original_img, (224, 224))
        
        # Scale 14x14 grid up to 224x224 smoothly
        heatmap_resized = cv2.resize(attention_map, (224, 224), interpolation=cv2.INTER_CUBIC)
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        
        # COLORMAP_JET gives the classic deep blue to bright red thermal look
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(original_img, 0.5, heatmap_color, 0.5, 0)
        
        # 8. Encode to Base64
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
        
        # 3. Generate Visuals (ViT Attention Rollout)
        b64_heatmap = self.generate_vit_attention(image_path, pt_input)
        
        print(f"🎯 Diagnosis: {result} ({confidence*100:.2f}%)")
        return result, confidence, b64_heatmap