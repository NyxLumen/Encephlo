import cv2
import os
import numpy as np
from glob import glob
from pathlib import Path

def process_mri(image_path, output_path):
    # 1. Load Image
    img = cv2.imread(image_path)
    if img is None:
        return False
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Extreme Contour Auto-Cropping (Skull/Void Stripping)
    # Threshold the image to separate brain tissue from the black void
    _, thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)
    
    # Find all contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Grab the largest contour (assumed to be the brain)
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        
        # Crop the image tightly around the brain
        cropped_img = img[y:y+h, x:x+w]
        cropped_gray = gray[y:y+h, x:x+w]
    else:
        # Fallback if contour fails
        cropped_img = img
        cropped_gray = gray

    # 3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # Enhances the soft tissue contrast without spiking the noise
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_gray = clahe.apply(cropped_gray)
    
    # Merge the CLAHE enhanced grayscale back into a 3-channel image (required for ViT/CNNs)
    final_img = cv2.cvtColor(clahe_gray, cv2.COLOR_GRAY2BGR)

    # 4. Standardize to 224x224
    resized_img = cv2.resize(final_img, (224, 224), interpolation=cv2.INTER_CUBIC)

    # 5. Save the cleaned tensor
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, resized_img)
    return True

def batch_process_dataset(input_dir, output_dir):
    print(f"🔥 INITIALIZING OPENCV SANITIZATION PIPELINE...")
    
    # Grab all jpgs/pngs in the raw dataset
    image_paths = glob(os.path.join(input_dir, '**', '*.*'), recursive=True)
    image_paths = [p for p in image_paths if p.endswith(('.jpg', '.jpeg', '.png'))]
    
    total = len(image_paths)
    success_count = 0
    
    for i, path in enumerate(image_paths):
        # Mirror the folder structure in the new cleaned directory
        relative_path = os.path.relpath(path, input_dir)
        out_path = os.path.join(output_dir, relative_path)
        
        if process_mri(path, out_path):
            success_count += 1
            
        if i % 100 == 0:
            print(f"   Processed: {i}/{total}")

    print(f"✅ SANITIZATION COMPLETE. Cleaned {success_count}/{total} scans.")

if __name__ == "__main__":
    # CHANGE THESE TO YOUR ACTUAL FOLDER PATHS
    RAW_DATASET_DIR = "data/raw" 
    CLEAN_DATASET_DIR = "data/cleaned"
    
    batch_process_dataset(RAW_DATASET_DIR, CLEAN_DATASET_DIR)