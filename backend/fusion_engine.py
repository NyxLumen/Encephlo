import os
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
import base64
import pickle
from PIL import Image

# Force CPU inference for stability on the web server
DEVICE = torch.device("cpu")

class FusionEngine:
    def __init__(self, models_dir="models"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(base_dir, models_dir)
        print(f"🔥 NEURAL ENGINE: Booting 3072-D PyTorch Stack from {self.models_dir}...")
        
        # 1. IEEE Clinical Image Transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(), # We handle resizing in OpenCV now
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 2. Load & Decapitate DenseNet121 (1024-D)
        try:
            self.densenet = models.densenet121(weights=None)
            self.densenet.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(1024, 4))
            self.densenet.load_state_dict(torch.load(os.path.join(self.models_dir, "densenet121_best_weights.pth"), map_location=DEVICE))
            self.densenet.classifier = nn.Identity()
            self.densenet = self.densenet.to(DEVICE).eval()
            print("   ✅ DenseNet121 Loaded & Decapitated.")
        except Exception as e:
            print(f"   ⚠️ ERROR Loading DenseNet: {e}")

        # 3. Load & Decapitate EfficientNet-B0 (1280-D)
        try:
            self.effnet = models.efficientnet_b0(weights=None)
            self.effnet.classifier[1] = nn.Linear(1280, 4)
            self.effnet.load_state_dict(torch.load(os.path.join(self.models_dir, "efficientnet_b0_best_weights.pth"), map_location=DEVICE))
            self.effnet.classifier = nn.Identity()
            self.effnet = self.effnet.to(DEVICE).eval()
            print("   ✅ EfficientNet-B0 Loaded & Decapitated.")
        except Exception as e:
            print(f"   ⚠️ ERROR Loading EfficientNet: {e}")

        # 4. Load & Decapitate ViT-B/16 (768-D)
        try:
            self.vit = models.vit_b_16(weights=None)
            self.vit.heads.head = nn.Linear(768, 4)
            self.vit.load_state_dict(torch.load(os.path.join(self.models_dir, "vit_b_16_best_weights.pth"), map_location=DEVICE))
            self.vit.heads.head = nn.Identity()
            self.vit = self.vit.to(DEVICE).eval()
            print("   ✅ ViT-B/16 Loaded & Decapitated.")
        except Exception as e:
            print(f"   ⚠️ ERROR Loading ViT: {e}")

        # 5. Load Master SVM Fusion Judge
        try:
            svm_path = os.path.join(self.models_dir, "master_svm_model.pkl")
            with open(svm_path, 'rb') as f:
                self.svm = pickle.load(f)
            print("   ✅ 3072-D Master SVM Fusion Layer Loaded.")
        except Exception as e:
            print(f"   ⚠️ ERROR Loading SVM: {e}")
            self.svm = None

    def apply_clinical_preprocessing(self, image_path):
        """
        The OpenCV Sanitization pipeline. 
        Strips the skull, crops the void, and applies CLAHE contrast enhancement.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"OpenCV could not read the image at {image_path}")
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Extreme Contour Auto-Cropping (Skull/Void Stripping)
        _, thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            cropped_gray = gray[y:y+h, x:x+w]
        else:
            cropped_gray = gray

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_gray = clahe.apply(cropped_gray)
        
        final_img = cv2.cvtColor(clahe_gray, cv2.COLOR_GRAY2BGR)

        # Standardize to 224x224 for the Neural Networks
        resized_img = cv2.resize(final_img, (224, 224), interpolation=cv2.INTER_CUBIC)
        
        # We also return the cleaned OpenCV image array for the Heatmap generator later
        return resized_img

    def preprocess_tensor(self, cv2_img):
        """Converts the cleaned OpenCV image array to a PyTorch tensor."""
        # Convert BGR (OpenCV) to RGB (PyTorch/PIL standard)
        rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        
        tensor = self.transform(pil_img).unsqueeze(0)
        return tensor.to(DEVICE)

    def generate_spatial_heatmap(self, cleaned_cv2_img, tensor_input):
        """Generates a thermal heatmap using EfficientNet's final spatial activations."""
        with torch.no_grad():
            features = self.effnet.features(tensor_input) # Shape: [1, 1280, 7, 7]
            heatmap = torch.mean(features, dim=1).squeeze().numpy()
            heatmap = np.maximum(heatmap, 0)
            if np.max(heatmap) != 0:
                heatmap /= np.max(heatmap)

        # Superimpose onto the CLEANED image, not the raw uploaded image
        heatmap_resized = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_CUBIC)
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(cleaned_cv2_img, 0.6, heatmap_color, 0.4, 0)
        
        _, buffer = cv2.imencode('.jpg', superimposed_img)
        b64_string = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{b64_string}"

    def predict(self, image_path):
        print(f"\n🧠 Processing MRI Scan: {image_path}")
        
        # 1. Clinical Sanitization (OpenCV)
        cleaned_cv2_img = self.apply_clinical_preprocessing(image_path)
        
        # 2. Tensor Conversion
        tensor = self.preprocess_tensor(cleaned_cv2_img)
        
        # 3. The 3072-D Feature Extraction (The Forge)
        with torch.no_grad():
            f_dense = self.densenet(tensor).numpy() # 1024-D
            f_eff = self.effnet(tensor).numpy()     # 1280-D
            f_vit = self.vit(tensor).numpy()        # 768-D
            
        # 4. Horizontal Fusion
        fusion_vector = np.concatenate((f_dense, f_eff, f_vit), axis=1) # 3072-D Vector
        
        # 5. Master SVM Classification
        if self.svm:
            prediction_idx = self.svm.predict(fusion_vector)[0]
            confidence = np.max(self.svm.predict_proba(fusion_vector))
        else:
            raise Exception("SVM Model not loaded. Cannot predict.")

        classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        diagnosis = classes[prediction_idx]
        
        # 6. Generate Heatmap Evidence (Mapped against the cleaned image)
        b64_heatmap = self.generate_spatial_heatmap(cleaned_cv2_img, tensor)
        
        print(f"🎯 Diagnosis Locked: {diagnosis} ({confidence*100:.2f}%)")
        return diagnosis, confidence, b64_heatmap