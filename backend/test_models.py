import os
import numpy as np

# Suppress annoying TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import torch
import tensorflow as tf
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
# Put a random MRI scan in the backend folder and rename it to test_image.jpg
TEST_IMAGE = "test_image.jpg" 

VIT_PATH = "models/vit_feature_extractor.pt"
# Change this to whatever Sid named his file
DENSENET_PATH = "models/densenet121.h5" 

print("🚀 INITIATING LOCAL MODEL TEST...\n")

# ─────────────────────────────────────────────
# 1. TEST PYTORCH (Adi's ViT)
# ─────────────────────────────────────────────
def test_vit():
    print("🧠 --- TESTING VISION TRANSFORMER (PyTorch) ---")
    if not os.path.exists(VIT_PATH):
        print(f"❌ Error: Could not find {VIT_PATH}")
        return None

    try:
        # Load the base architecture
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=4)
        
        # Inject your 99.53% weights
        model.vit.load_state_dict(torch.load(VIT_PATH, map_location="cpu"))
        model.eval()
        print("✅ ViT Weights Loaded Successfully.")

        # Process Image
        image = Image.open(TEST_IMAGE).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        # Run Inference
        with torch.no_grad():
            features = model.vit(**inputs).pooler_output
            
        print(f"✅ ViT Inference Complete! Extracted Vector Shape: {features.shape}")
        return features.numpy()

    except Exception as e:
        print(f"❌ ViT Error: {e}")
        return None

# ─────────────────────────────────────────────
# 2. TEST TENSORFLOW (Sid's DenseNet)
# ─────────────────────────────────────────────
def test_tf():
    print("\n🔬 --- TESTING DENSENET (TensorFlow) ---")
    if not os.path.exists(DENSENET_PATH):
        print(f"⚠️ Warning: Could not find {DENSENET_PATH}. Tell Sid to hurry up.")
        return None

    try:
        # Load Model
        tf_model = tf.keras.models.load_model(DENSENET_PATH, compile=False)
        print("✅ DenseNet Loaded Successfully.")

        # Process Image (Standard TF CNN Preprocessing)
        img = tf.io.read_file(TEST_IMAGE)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (224, 224))
        img = img / 255.0
        img = tf.expand_dims(img, axis=0)

        # Run Inference
        features = tf_model.predict(img, verbose=0)
        print(f"✅ DenseNet Inference Complete! Extracted Vector Shape: {features.shape}")
        return features

    except Exception as e:
        print(f"❌ DenseNet Error: {e}")
        return None

# ─────────────────────────────────────────────
# EXECUTE
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.exists(TEST_IMAGE):
        print(f"🛑 STOP: You need to put an MRI image named '{TEST_IMAGE}' in this folder first!")
    else:
        vit_feat = test_vit()
        tf_feat = test_tf()

        if vit_feat is not None and tf_feat is not None:
            print("\n🧬 --- TESTING FUSION CAPABILITY ---")
            # This is exactly what the SVM will see
            fusion_vector = np.concatenate([tf_feat, vit_feat], axis=1)
            print(f"✅ Fusion Successful! Final Vector Shape: {fusion_vector.shape}")
            print("\n🎉 ALL SYSTEMS GO. Ready to train the SVM!")