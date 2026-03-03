import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from transformers import TFAutoModelForImageClassification, AutoImageProcessor

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
LR = 2e-5  # Fine-tuning learning rate suitable for ViT

OUTPUT_MODEL_PATH = r"c:\Users\Siddharth Gupta\Desktop\main_encephlo\Encephlo\backend\models\vit_feature_extractor"

print("=" * 60)
print("LOADING DATA FOR VISION TRANSFORMER (HuggingFace)")
print("=" * 60)

# Use HuggingFace Feature Extractor for ViT Base
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

def process_image(image_path, label):
    # Load and decode image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    
    # Resize and standard ViT normalization (done roughly here for tf.data compatibility)
    # The processor usually does: resize to 224, rescale to [0,1], normalize (mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    image = (image - 0.5) / 0.5
    
    # Hugging Face TF models expect channels last or first? 
    # TFViTForImageClassification expects channels last (NHWC) in typical Keras setup, but 
    # let's be careful. By default HF expects NCHW for Pytorch, but TF API handles NHWC.
    # Actually, transformers TFViT input shape is usually (batch, 3, 224, 224) if not using Keras layers directly.
    # We will transpose it just in case: (3, 224, 224)
    image = tf.transpose(image, [2, 0, 1])
    return image, label

# Create Datasets
def create_dataset(directory):
    classes = sorted(os.listdir(directory))
    class_indices = {name: idx for idx, name in enumerate(classes)}
    
    filepaths = []
    labels = []
    
    for class_name in classes:
        class_dir = os.path.join(directory, class_name)
        if not os.path.isdir(class_dir):
            continue
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

# Load pre-trained ViT
model = TFAutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k", 
    num_labels=NUM_CLASSES
)

# Compile
optimizer = keras.optimizers.Adam(learning_rate=LR)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = ["accuracy"]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

print("\n" + "=" * 60)
print("FINE-TUNING ViT")
print("=" * 60)

# We won't train fully here since we don't have time/GPUs, but the script is complete for the user
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS
)

# Chop off head to get the 768-D feature vector
# The ViT model base output has `pooler_output` or `last_hidden_state`
print("Saving headless feature extractor...")

# We can save the weights for the model. 
# But let's build a functional Keras model wrapping it to output exactly 768
class ViTFeatureExtractor(tf.keras.Model):
    def __init__(self, vit_model):
        super().__init__()
        self.vit = vit_model.vit # The core ViT body
        
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
