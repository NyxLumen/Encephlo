import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

try:
    from transformers import TFAutoModelForImageClassification, AutoImageProcessor
except ImportError:
    print("\nFATAL: transformers not found or misconfigured for TensorFlow!")
    print("On Colab, you MUST run this first before running this script:")
    print("!pip install tensorflow==2.15.0 tf-keras transformers==4.35.0 huggingface_hub scikit-learn\n")
    raise

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
# Base dataset path
if os.path.exists("/content"):
    # Google Colab path
    BASE_DIR = "/content"
else:
    # Local Windows path
    BASE_DIR = r"c:\Users\Siddharth Gupta\Desktop\main_encephlo\Encephlo"

TRAIN_DIR = os.path.join(BASE_DIR, "MRI images", "Training")
TEST_DIR  = os.path.join(BASE_DIR, "MRI images", "Testing")

IMG_SIZE = 224
BATCH_SIZE = 16  # Smaller batch size for ViT
NUM_CLASSES = 4
SEED = 42

EPOCHS = 10
LR = 2e-5

print("=" * 60)
print("LOADING DATA FOR VISION TRANSFORMER (HuggingFace/TF)")
print("=" * 60)

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

def process_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    
    # Resize and standard ViT normalization
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    image = (image - 0.5) / 0.5
    
    # TF AutoModels expect channels last (NHWC) in standard Keras
    image = tf.transpose(image, [2, 0, 1])
    return image, label

def create_dataset(directory):
    classes = sorted(os.listdir(directory))
    class_indices = {name: idx for idx, name in enumerate(classes)}
    
    filepaths = []
    labels = []
    
    for class_name in classes:
        class_dir = os.path.join(directory, class_name)
        if not os.path.isdir(class_dir): continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepaths.append(os.path.join(class_dir, fname))
                labels.append(class_indices[class_name])
                
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    ds = ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds, class_indices

train_ds, class_indices = create_dataset(TRAIN_DIR)
test_ds, _ = create_dataset(TEST_DIR)

print(f"Classes: {class_indices}")

print("\n" + "=" * 60)
print("BUILDING ViT MODEL")
print("=" * 60)

model = TFAutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k", 
    num_labels=NUM_CLASSES
)

optimizer = keras.optimizers.Adam(learning_rate=LR)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

print("\n" + "=" * 60)
print("FINE-TUNING ViT")
print("=" * 60)

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS
)

print("Saving headless feature extractor as .keras ...")

# Wrap it in standard Keras for `.keras` serialization
class ViTFeatureExtractor(tf.keras.Model):
    def __init__(self, vit_model):
        super().__init__()
        self.vit = vit_model.vit
        
    def call(self, inputs):
        # outputs = self.vit(inputs)
        # return outputs.pooler_output # 768-D
        return self.vit(inputs).pooler_output

feature_extractor = ViTFeatureExtractor(model)

# Test a dummy input to build it
dummy_input = tf.zeros((1, 3, 224, 224))
out = feature_extractor(dummy_input)
print(f"Feature Extractor Output Shape: {out.shape}")  # Should be (1, 768)

# Save it using TF SavedModel
feature_extractor.save(OUTPUT_MODEL_PATH)
print(f"ViT Feature Extractor saved to {OUTPUT_MODEL_PATH}")
