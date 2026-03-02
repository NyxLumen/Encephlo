import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
from transformers import TFViTForImageClassification, ViTImageProcessor

# ─────────────────────────────────────────────────────────────────────────────
# Set paths
# ─────────────────────────────────────────────────────────────────────────────
# Encephlo directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# MRI images directory is under densenet/
DATA_DIR = os.path.join(BASE_DIR, '..', 'MRI images')
TRAIN_DIR = os.path.join(DATA_DIR, 'Training')
TEST_DIR  = os.path.join(DATA_DIR, 'Testing')

MODELS_DIR     = os.path.join(BASE_DIR, 'models')
DENSENET_PATH  = os.path.join(MODELS_DIR, 'densenet121.keras')
VIT_DIR        = os.path.join(MODELS_DIR, 'vit_finetuned')
SVM_PATH       = os.path.join(MODELS_DIR, 'svm_fusion_weights.pkl')

BATCH_SIZE = 16
IMG_SIZE = (224, 224)
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
NUM_CLASSES = len(CLASSES)

print("=" * 60)
print("  ENCEPHLO — Fusion Training Pipeline ")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Prepare Datasets (tf.data.Dataset) - Efficient for extracting and fine-tuning
# ─────────────────────────────────────────────────────────────────────────────
print("\n[v] Loading datasets...")

train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR, labels='inferred', label_mode='categorical',
    class_names=CLASSES, color_mode='rgb',
    batch_size=BATCH_SIZE, image_size=IMG_SIZE, shuffle=True
)

test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR, labels='inferred', label_mode='categorical',
    class_names=CLASSES, color_mode='rgb',
    batch_size=BATCH_SIZE, image_size=IMG_SIZE, shuffle=False
)

# Unshuffled train dataset for feature extraction
train_ds_unshuffled = keras.utils.image_dataset_from_directory(
    TRAIN_DIR, labels='inferred', label_mode='categorical',
    class_names=CLASSES, color_mode='rgb',
    batch_size=BATCH_SIZE, image_size=IMG_SIZE, shuffle=False
)

def extract_labels(ds):
    y = []
    for _, batch_y in ds:
        y.extend(np.argmax(batch_y.numpy(), axis=1))
    return np.array(y)

y_train = extract_labels(train_ds_unshuffled)
y_test  = extract_labels(test_ds)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Setup ViT Fine-Tuning
# ─────────────────────────────────────────────────────────────────────────────
print("\n[v] Loading and fine-tuning ViT...")

# ViT preprocessing: typically scales to [-1, 1], so we normalize our [0, 255] images
def preprocess_for_vit(x, y):
    x = tf.cast(x, tf.float32)
    # The vit-base model expects normalization with mean=[0.5, 0.5, 0.5] and std=[0.5, 0.5, 0.5]
    x = (x / 127.5) - 1.0  
    # ViT expects channels first (B, C, H, W)
    x = tf.transpose(x, [0, 3, 1, 2])
    return x, y

vit_train_ds = train_ds.map(preprocess_for_vit).prefetch(tf.data.AUTOTUNE)
vit_test_ds  = test_ds.map(preprocess_for_vit).prefetch(tf.data.AUTOTUNE)

# Load ViT for image classification
vit_model_cls = TFViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
)

vit_model_cls.compile(
    optimizer=keras.optimizers.Adam(learning_rate=2e-5),
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# We fine-tune for a few epochs (3 is usually enough for a strong feature extractor)
print("\n--- Fine-Tuning ViT (3 Epochs) ---")
vit_model_cls.fit(vit_train_ds, validation_data=vit_test_ds, epochs=3)
vit_model_cls.save_pretrained(VIT_DIR)
print(f"    Saved fine-tuned ViT to {VIT_DIR}")

# Headless ViT (the inner `.vit` model outputs sequence, we take CLS token `[:, 0, :]`)
class HeadlessViT(keras.Model):
    def __init__(self, vit_base):
        super().__init__()
        self.vit = vit_base

    def call(self, inputs):
        # inputs are normalized images
        outputs = self.vit(inputs)
        return outputs.last_hidden_state[:, 0, :] # 768-D

headless_vit = HeadlessViT(vit_model_cls.vit)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Setup DenseNet121 Headless
# ─────────────────────────────────────────────────────────────────────────────
print("\n[v] Loading DenseNet121...")
densenet_base = keras.applications.DenseNet121(include_top=False, weights=None, input_shape=(224, 224, 3))
inputs = keras.Input(shape=(224, 224, 3), name='input_layer')
x = densenet_base(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D(name='gap')(x)
x = keras.layers.BatchNormalization(name='bn_top')(x)
x = keras.layers.Dense(512, activation='relu', name='fc_512')(x)
x = keras.layers.Dropout(0.4, name='dropout_1')(x)
x = keras.layers.Dense(256, activation='relu', name='fc_256')(x)
x = keras.layers.Dropout(0.3, name='dropout_2')(x)
outputs = keras.layers.Dense(4, activation='softmax', name='predictions')(x)

densenet_full = keras.Model(inputs=inputs, outputs=outputs, name='Encephlo_DenseNet121')
densenet_full.load_weights(DENSENET_PATH.replace('.keras', '.weights.h5'))

# The GAP layer output is usually 1024. Intrain.py, it was called 'gap'
dense_gap_layer = densenet_full.get_layer('gap').output
headless_densenet = keras.Model(inputs=densenet_full.input, outputs=dense_gap_layer)

def preprocess_densenet(x):
    return keras.applications.densenet.preprocess_input(tf.cast(x, tf.float32))

# ─────────────────────────────────────────────────────────────────────────────
# 4. Setup EfficientNetB0 Headless
# ─────────────────────────────────────────────────────────────────────────────
print("\n[v] Loading EfficientNetB0 (ImageNet weights)...")
effnet_base = keras.applications.EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
x = keras.layers.GlobalAveragePooling2D()(effnet_base.output)
headless_effnet = keras.Model(inputs=effnet_base.input, outputs=x)

def preprocess_effnet(x):
    return keras.applications.efficientnet.preprocess_input(tf.cast(x, tf.float32))

# ─────────────────────────────────────────────────────────────────────────────
# 5. Extract Features
# ─────────────────────────────────────────────────────────────────────────────
print("\n[v] Extracting features from DenseNet, EfficientNet, and ViT...")

def extract_all_features(dataset):
    feats_dense, feats_eff, feats_vit = [], [], []
    for x_batch, _ in dataset:
        # DenseNet
        xd = preprocess_densenet(x_batch)
        feats_dense.append(headless_densenet(xd, training=False).numpy())
        
        # EffNet
        xe = preprocess_effnet(x_batch)
        feats_eff.append(headless_effnet(xe, training=False).numpy())
        
        # ViT
        xv = (tf.cast(x_batch, tf.float32) / 127.5) - 1.0
        xv = tf.transpose(xv, [0, 3, 1, 2])
        feats_vit.append(headless_vit(xv, training=False).numpy())
        
    return (
        np.concatenate(feats_dense, axis=0), # 1024
        np.concatenate(feats_eff, axis=0),   # 1280
        np.concatenate(feats_vit, axis=0)    # 768
    )

print("    -> Train features...")
d_tr, e_tr, v_tr = extract_all_features(train_ds_unshuffled)
X_train_fusion = np.concatenate([d_tr, e_tr, v_tr], axis=1) # 1024+1280+768 = 3072

print("    -> Test features...")
d_te, e_te, v_te = extract_all_features(test_ds)
X_test_fusion = np.concatenate([d_te, e_te, v_te], axis=1) # 3072

print(f"    Fusion Train Shape: {X_train_fusion.shape}")
print(f"    Fusion Test Shape:  {X_test_fusion.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Train SVM fusion
# ─────────────────────────────────────────────────────────────────────────────
print("\n[v] Training Support Vector Machine on 3072-D Fusion Vectors...")
svm_clf = SVC(probability=True, kernel='rbf', C=1.0)
svm_clf.fit(X_train_fusion, y_train)

print("\n--- Evaluate SVM ---")
y_pred = svm_clf.predict(X_test_fusion)
acc = accuracy_score(y_test, y_pred)
print(f"SVM Accuracy: {acc * 100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=CLASSES, digits=4))

joblib.dump(svm_clf, SVM_PATH)
print(f"[v] Saved SVM weights to {SVM_PATH}")

print("=" * 60)
print("  API READY - RUN fusion_engine.py ")
print("=" * 60)
