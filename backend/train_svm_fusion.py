import os
import numpy as np
import tensorflow as tf
from sklearn.svm import SVC
import joblib
import cv2
from tqdm import tqdm

# Configuration
if os.path.exists("/content"):
    # Google Colab path
    BASE_DIR = "/content"
    MODELS_DIR = "/content/models"
else:
    # Local Windows path
    BASE_DIR = r"c:\Users\Siddharth Gupta\Desktop\main_encephlo\Encephlo"
    MODELS_DIR = os.path.join(BASE_DIR, "backend", "models")

TRAIN_DIR = os.path.join(BASE_DIR, "MRI images", "Training")

EFFNET_PATH = os.path.join(MODELS_DIR, "efficientnetb0.keras")
DENSENET_PATH = os.path.join(MODELS_DIR, "densenet121.keras")
VIT_PATH = os.path.join(MODELS_DIR, "vit_feature_extractor")  # SavedModel from train_vit.py

OUTPUT_SVM_PATH = os.path.join(MODELS_DIR, "svm_fusion_weights.pkl")

print("Loading Feature Extractors...")

# 1. Load EfficientNet and chop head (look for GlobalAveragePooling2D)
full_effnet = tf.keras.models.load_model(EFFNET_PATH, compile=False)
eff_pool_layer_name = None
for layer in full_effnet.layers:
    if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
        eff_pool_layer_name = layer.name
        break
effnet_extractor = tf.keras.Model(inputs=full_effnet.input, outputs=full_effnet.get_layer(eff_pool_layer_name).output)
print(f"EfficientNet Extractor Output Shape: {effnet_extractor.output_shape}") # Expected (None, 1280)

# 2. Load DenseNet and chop head
full_densenet = tf.keras.models.load_model(DENSENET_PATH, compile=False)
dense_pool_layer_name = None
for layer in full_densenet.layers:
    if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
        dense_pool_layer_name = layer.name
        break
densenet_extractor = tf.keras.Model(inputs=full_densenet.input, outputs=full_densenet.get_layer(dense_pool_layer_name).output)
print(f"DenseNet Extractor Output Shape: {densenet_extractor.output_shape}") # Expected (None, 1024)

# 3. Load ViT Extractor (768-D)
try:
    vit_extractor = tf.keras.models.load_model(VIT_PATH, compile=False)
    print("Loaded ViT Extractor.")
except:
    print("Warning: ViT Extractor not found. Please run train_vit.py first. Proceeding with dummy ViT for skeleton test.")
    class DummyViT:
        def predict(self, x, verbose=0):
            return np.zeros((x.shape[0], 768))
    vit_extractor = DummyViT()

def preprocess_for_cnn(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    # EfficientNet internally preprocesses, but let's pass a standard [0,255] range as the model expects
    img_expanded = np.expand_dims(img, axis=0)
    return img_expanded

def preprocess_for_densenet(img_path):
    # DenseNet usually expects normalization / preprocessed inputs based on Imagenet
    # For now, we will pass the same as EfficientNet, but using tf.keras.applications.densenet.preprocess_input
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = tf.keras.applications.densenet.preprocess_input(img.astype(np.float32))
    img_expanded = np.expand_dims(img, axis=0)
    return img_expanded

def preprocess_for_vit(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    # Move channels to front: (3, 224, 224) if HF model expects it 
    # OR leave as (224, 224, 3) if Keras HF wrapper expects NHWC. Usually Keras Wrapper expects NHWC.
    # Let's assume NHWC since we use Keras TFViTModel
    img = np.transpose(img, (2, 0, 1)) # standard for pure HF Models
    img_expanded = np.expand_dims(img, axis=0)
    return img_expanded

# Data Collection
X_train = []
Y_train = []

classes = sorted(os.listdir(TRAIN_DIR))
class_idx = {cls: idx for idx, cls in enumerate(classes)}

print("\nExtracting Features from Training Set...")
# To test script instantly, we will only take 10 images per class
for cls in classes:
    cls_dir = os.path.join(TRAIN_DIR, cls)
    if not os.path.isdir(cls_dir): continue
    
    images = os.listdir(cls_dir)
    print(f"Processing class: {cls}")
    
    # Take a sample of images for real training
    for img_name in tqdm(images): 
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')): continue
        
        img_path = os.path.join(cls_dir, img_name)
        
        try:
            # 1. EfficientNet
            eff_input = preprocess_for_cnn(img_path)
            eff_feat = effnet_extractor.predict(eff_input, verbose=0)[0]  # (1280,)
            
            # 2. DenseNet
            dense_input = preprocess_for_densenet(img_path)
            dense_feat = densenet_extractor.predict(dense_input, verbose=0)[0] # (1024,)
            
            # 3. ViT
            vit_input = preprocess_for_vit(img_path)
            vit_feat = vit_extractor.predict(vit_input, verbose=0)[0] # (768,)
            if len(vit_feat.shape) > 1: # if outputting batch, seq, hidden
                vit_feat = vit_feat[0] # take cls token if needed
                
            # Fusion
            fused_vector = np.concatenate([eff_feat, dense_feat, vit_feat]) # 1280 + 1024 + 768 = 3072
            
            X_train.append(fused_vector)
            Y_train.append(class_idx[cls])
        except Exception as e:
            # print(f"Error processing {img_name}: {e}")
            pass

X_train = np.array(X_train)
Y_train = np.array(Y_train)

print(f"\nTraining SVM with Feature Vectors of shape {X_train.shape}...")
svm = SVC(probability=True, kernel='linear', C=1.0)
svm.fit(X_train, Y_train)

train_acc = svm.score(X_train, Y_train)
print(f"SVM Training Accuracy: {train_acc * 100:.2f}%")

joblib.dump(svm, OUTPUT_SVM_PATH)
print(f"Saved Fusion Model Weights to {OUTPUT_SVM_PATH}")
