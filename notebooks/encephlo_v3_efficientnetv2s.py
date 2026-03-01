# %% [markdown]
# # Encephlo v3 — EfficientNetV2-S + Squeeze-and-Excitation
# **Clinical Decision Support for Brain Tumor Triage**
#
# Production-grade classifier optimized for:
# - Exceeding 96% validation accuracy
# - Maximizing tumor recall (minimizing false negatives)
# - Attention-enhanced GradCAM focus
#
# Architecture: EfficientNetV2-S → SE Block → GAP → Dense Head → 4-class softmax
# Loss: Focal Loss (gamma=2.0, balanced alpha)
# Strategy: Two-phase training (frozen backbone → fine-tuning last 40%)
#
# ── COLAB INSTRUCTIONS ──────────────────────────────────────
# 1. Upload your MRI dataset to Google Drive at:
#      My Drive/Encephlo/MRI images/Training/
#        ├── glioma/
#        ├── meningioma/
#        ├── notumor/
#        └── pituitary/
#
# 2. Run each cell sequentially.
#    Google Drive will be mounted automatically.
# ─────────────────────────────────────────────────────────────

# %%
# ============================================================
# SECTION 0A: ENVIRONMENT SETUP (Colab / Local auto-detect)
# ============================================================

import os, sys

# Detect Google Colab
IS_COLAB = "google.colab" in sys.modules or os.path.exists("/content")

if IS_COLAB:
    print("☁️  Google Colab detected — mounting Google Drive...")
    from google.colab import drive
    drive.mount("/content/drive")

    # ── Set Colab paths ─────────────────────────────────────
    # IMPORTANT: Update this if your Drive folder structure differs
    DRIVE_BASE = "/content/drive/MyDrive/Encephlo"
    DATA_DIR       = os.path.join(DRIVE_BASE, "MRI images", "Training")
    CHECKPOINT_DIR = os.path.join(DRIVE_BASE, "checkpoints")
    MODEL_SAVE_DIR = os.path.join(DRIVE_BASE, "models")

    print(f"  Data dir   : {DATA_DIR}")
    print(f"  Checkpoints: {CHECKPOINT_DIR}")
    print(f"  Models     : {MODEL_SAVE_DIR}")
else:
    print("💻 Local environment detected.")
    _SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR       = os.path.join(_SCRIPT_DIR, "..", "..", "MRI images", "Training")
    CHECKPOINT_DIR = os.path.join(_SCRIPT_DIR, "..", "checkpoints")
    MODEL_SAVE_DIR = os.path.join(_SCRIPT_DIR, "..", "models")

# %%
# ============================================================
# SECTION 0B: REPRODUCIBILITY & CONFIGURATION
# ============================================================

import json
import random
import pathlib
import warnings
import datetime

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

# ── Deterministic seeds ──────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass  # fallback for older TF versions

# ── Hyperparameter Registry ─────────────────────────────────
CONFIG = {
    "model_name": "EfficientNetV2S_SE",
    "image_size": (224, 224),
    "input_shape": (224, 224, 3),
    "batch_size": 32,
    "num_classes": 4,
    "class_names": ["glioma", "meningioma", "notumor", "pituitary"],
    # Split
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    # Phase 1
    "phase1_epochs": 10,
    "phase1_lr": 1e-4,
    # Phase 2
    "phase2_epochs": 50,
    "phase2_lr": 1e-5,
    "fine_tune_percent": 0.40,
    # Optimizer
    "weight_decay": 1e-4,
    # Focal loss
    "focal_gamma": 2.0,
    # Head
    "dense_units": 512,
    "dropout_rate": 0.4,
    # SE block
    "se_ratio": 16,
    # Callbacks
    "reduce_lr_patience": 5,
    "early_stop_patience": 10,
    # Paths (set from environment detection above)
    "data_dir": DATA_DIR,
    "checkpoint_dir": CHECKPOINT_DIR,
    "model_save_dir": MODEL_SAVE_DIR,
    "seed": SEED,
}

# Create output directories
os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
os.makedirs(CONFIG["model_save_dir"], exist_ok=True)

print("=" * 60)
print("ENCEPHLO v3 — EfficientNetV2-S + SE Attention")
print("=" * 60)
print(f"TensorFlow : {tf.__version__}")
print(f"GPU        : {tf.config.list_physical_devices('GPU')}")
print(f"Runtime    : {'Google Colab' if IS_COLAB else 'Local'}")
print(f"Seed       : {CONFIG['seed']}")
print(f"Data dir   : {CONFIG['data_dir']}")
print(f"Timestamp  : {datetime.datetime.now().isoformat()}")
print("=" * 60)

# Log full config
for k, v in CONFIG.items():
    print(f"  {k:25s} = {v}")

# %%
# ============================================================
# SECTION 1: DATA PIPELINE  (tf.data, 70/15/15 stratified split)
# ============================================================
# - Loads images from disk via file paths
# - Stratified split prevents data leakage
# - Augmentation applied ONLY to training set
# - EfficientNetV2 preprocessing
# - Caching + prefetching for GPU saturation

def collect_image_paths_and_labels(data_dir, class_names):
    """Walk the data directory and collect all image paths with labels."""
    paths, labels = [], []
    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for fname in os.listdir(class_dir):
            fpath = os.path.join(class_dir, fname)
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                paths.append(fpath)
                labels.append(label_idx)
    return paths, labels


# Collect all data
all_paths, all_labels = collect_image_paths_and_labels(
    CONFIG["data_dir"], CONFIG["class_names"]
)
print(f"\nTotal images found: {len(all_paths)}")
for i, name in enumerate(CONFIG["class_names"]):
    count = all_labels.count(i)
    print(f"  {name:15s}: {count:5d}")

# ── Stratified split: 70 / 15 / 15 ─────────────────────────
# First split: 70% train, 30% temp
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    all_paths, all_labels,
    test_size=(1 - CONFIG["train_ratio"]),
    stratify=all_labels,
    random_state=SEED,
)

# Second split: temp → 50/50 → 15% val, 15% test
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels,
    test_size=0.5,
    stratify=temp_labels,
    random_state=SEED,
)

print(f"\nSplit sizes:")
print(f"  Train : {len(train_paths)}")
print(f"  Val   : {len(val_paths)}")
print(f"  Test  : {len(test_paths)}")

# ── Compute class weights for focal loss ─────────────────────
from collections import Counter

train_counter = Counter(train_labels)
total_train = len(train_labels)
num_classes = CONFIG["num_classes"]

# Inverse frequency, normalized
class_weights = {}
for cls_idx in range(num_classes):
    class_weights[cls_idx] = total_train / (num_classes * train_counter[cls_idx])

# Convert to alpha array (normalized to sum=1)
alpha_raw = np.array([class_weights[i] for i in range(num_classes)], dtype=np.float32)
alpha_normalized = alpha_raw / alpha_raw.sum()
CONFIG["focal_alpha"] = alpha_normalized.tolist()

print(f"\nClass weights (alpha for focal loss):")
for i, name in enumerate(CONFIG["class_names"]):
    print(f"  {name:15s}: {alpha_normalized[i]:.4f}  (count={train_counter[i]})")


# ── tf.data loading functions ────────────────────────────────
IMG_SIZE = CONFIG["image_size"]
AUTOTUNE = tf.data.AUTOTUNE


def load_and_preprocess(path, label):
    """Load an image from disk, decode, resize, and apply EfficientNetV2 preprocessing."""
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.efficientnet_v2.preprocess_input(img)
    return img, label


def augment(img, label):
    """Apply data augmentation — training set ONLY.

    - Random horizontal flip
    - Random rotation ±10° (≈0.175 rad)
    - Random zoom ±10%
    - Random width/height shift ±5%
    """
    img = tf.image.random_flip_left_right(img)

    # Random rotation ±10 degrees
    angle = tf.random.uniform([], -10.0, 10.0) * (np.pi / 180.0)
    img = rotate_image(img, angle)

    # Random zoom ±10%  (crop + resize trick)
    zoom_factor = tf.random.uniform([], 0.9, 1.1)
    h, w = IMG_SIZE
    new_h = tf.cast(tf.cast(h, tf.float32) * zoom_factor, tf.int32)
    new_w = tf.cast(tf.cast(w, tf.float32) * zoom_factor, tf.int32)
    img = tf.image.resize(img, (new_h, new_w))
    img = tf.image.resize_with_crop_or_pad(img, h, w)

    # Random translate ±5%
    dx = tf.random.uniform([], -0.05, 0.05) * tf.cast(w, tf.float32)
    dy = tf.random.uniform([], -0.05, 0.05) * tf.cast(h, tf.float32)
    img = translate_image(img, dx, dy)

    return img, label


@tf.function
def rotate_image(image, angle):
    """Rotate image by angle (radians) using affine transform."""
    cos_a = tf.math.cos(angle)
    sin_a = tf.math.sin(angle)
    # Center of image
    h = tf.cast(tf.shape(image)[0], tf.float32)
    w = tf.cast(tf.shape(image)[1], tf.float32)
    cx, cy = w / 2.0, h / 2.0

    # Build 8-element flat transform vector for ImageProjectiveTransformV3
    # [a0, a1, a2, b0, b1, b2, c0, c1]
    # Maps output → input: x_src = a0*x + a1*y + a2, y_src = b0*x + b1*y + b2
    transform = tf.stack([
        cos_a, sin_a, cx - cos_a * cx - sin_a * cy,
        -sin_a, cos_a, cy + sin_a * cx - cos_a * cy,
        0.0, 0.0
    ])
    image = tf.expand_dims(image, 0)
    image = tf.raw_ops.ImageProjectiveTransformV3(
        images=image,
        transforms=tf.expand_dims(transform, 0),
        output_shape=tf.shape(image)[1:3],
        interpolation="BILINEAR",
        fill_mode="NEAREST",
        fill_value=0.0,
    )
    return tf.squeeze(image, 0)


@tf.function
def translate_image(image, dx, dy):
    """Translate image by (dx, dy) pixels."""
    transform = tf.stack([1.0, 0.0, -dx, 0.0, 1.0, -dy, 0.0, 0.0])
    image = tf.expand_dims(image, 0)
    image = tf.raw_ops.ImageProjectiveTransformV3(
        images=image,
        transforms=tf.expand_dims(transform, 0),
        output_shape=tf.shape(image)[1:3],
        interpolation="BILINEAR",
        fill_mode="NEAREST",
        fill_value=0.0,
    )
    return tf.squeeze(image, 0)


def build_dataset(paths, labels, is_training=False, batch_size=32):
    """Build a tf.data.Dataset with optional augmentation."""
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if is_training:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED)
    ds = ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    if is_training:
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    if not is_training:
        ds = ds.cache()
    return ds


# Build all three datasets
train_ds = build_dataset(train_paths, train_labels, is_training=True, batch_size=CONFIG["batch_size"])
val_ds   = build_dataset(val_paths,   val_labels,   is_training=False, batch_size=CONFIG["batch_size"])
test_ds  = build_dataset(test_paths,  test_labels,  is_training=False, batch_size=CONFIG["batch_size"])

# Quick sanity check
for images, labels in train_ds.take(1):
    print(f"\nBatch shape : {images.shape}")
    print(f"Label batch : {labels.numpy()[:8]}...")
    print(f"Pixel range : [{images.numpy().min():.2f}, {images.numpy().max():.2f}]")

# %%
# ============================================================
# SECTION 2: MODEL ARCHITECTURE
# ============================================================
# EfficientNetV2-S backbone → SE block → GAP → Dense head
#
# The SE (Squeeze-and-Excitation) block recalibrates channel-wise
# feature responses, sharpening attention on tumor-relevant features
# and improving GradCAM focus.


def squeeze_and_excitation_block(inputs, ratio=16):
    """Squeeze-and-Excitation block for channel attention.

    1. Squeeze: Global Average Pooling collapses spatial dims → (B, C)
    2. Excitation: Two FC layers learn channel-wise importance
       - Dense(C//ratio, relu) — bottleneck
       - Dense(C, sigmoid)     — per-channel gate [0, 1]
    3. Scale: Element-wise multiply to recalibrate feature map

    Args:
        inputs: 4D tensor (batch, H, W, C) — backbone feature map
        ratio:  reduction ratio for the bottleneck (default 16)

    Returns:
        Recalibrated 4D tensor (batch, H, W, C)
    """
    channels = inputs.shape[-1]

    # Squeeze: spatial → channel descriptor
    se = tf.keras.layers.GlobalAveragePooling2D(name="se_squeeze")(inputs)

    # Excitation: learn channel interdependencies
    se = tf.keras.layers.Dense(
        channels // ratio,
        activation="relu",
        kernel_initializer="he_normal",
        name="se_reduce",
    )(se)
    se = tf.keras.layers.Dense(
        channels,
        activation="sigmoid",
        kernel_initializer="he_normal",
        name="se_expand",
    )(se)

    # Reshape for broadcasting: (B, C) → (B, 1, 1, C)
    se = tf.keras.layers.Reshape((1, 1, channels), name="se_reshape")(se)

    # Scale: recalibrate
    return tf.keras.layers.Multiply(name="se_scale")([inputs, se])


def build_model(config):
    """Build the full EfficientNetV2-S + SE + classification head.

    Architecture:
        Input(224,224,3)
          → EfficientNetV2S(pretrained, no top)
          → SE Block (channel attention)
          → GlobalAveragePooling2D
          → Dense(512) → BN → ReLU → Dropout(0.4)
          → Dense(4, softmax)
    """
    inputs = tf.keras.Input(shape=config["input_shape"], name="input_image")

    # ── Backbone ─────────────────────────────────────────────
    backbone = tf.keras.applications.EfficientNetV2S(
        include_top=False,
        weights="imagenet",
        input_shape=config["input_shape"],
    )
    backbone.trainable = False  # Frozen for Phase 1
    backbone._name = "efficientnetv2s"

    x = backbone(inputs, training=False)

    # ── Squeeze-and-Excitation ───────────────────────────────
    x = squeeze_and_excitation_block(x, ratio=config["se_ratio"])

    # ── Classification Head ──────────────────────────────────
    x = tf.keras.layers.GlobalAveragePooling2D(name="head_gap")(x)
    x = tf.keras.layers.Dense(config["dense_units"], name="head_dense")(x)
    x = tf.keras.layers.BatchNormalization(name="head_bn")(x)
    x = tf.keras.layers.Activation("relu", name="head_relu")(x)
    x = tf.keras.layers.Dropout(config["dropout_rate"], name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(
        config["num_classes"], activation="softmax", name="predictions"
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="EfficientNetV2S_SE")
    return model, backbone


model, backbone = build_model(CONFIG)
try:
    model.summary(line_length=120, show_trainable=True, expand_nested=False)
except (ValueError, Exception) as e:
    print(f"[model.summary skipped: {e}]")
    print(f"Model: {model.name}, Params: {model.count_params():,}")

# %%
# ============================================================
# SECTION 3: FOCAL LOSS
# ============================================================
# Focal Loss down-weights easy examples and focuses training on
# hard misclassifications — critical for imbalanced clinical data.
#
# FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
#
# With balanced alpha, tumor classes receive proportionally higher
# weight, directly improving recall and reducing false negatives.


def focal_loss(gamma=2.0, alpha=None):
    """Create a focal loss function with class-balanced alpha.

    Args:
        gamma: focusing parameter. Higher gamma → more focus on hard examples.
               gamma=0 reduces to standard cross-entropy.
        alpha: per-class weight array of shape (num_classes,). If None, uniform.

    Returns:
        A loss function compatible with model.compile(loss=...).
    """
    alpha_tensor = tf.constant(alpha, dtype=tf.float32) if alpha is not None else None

    def _focal_loss(y_true, y_pred):
        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())

        # Convert integer labels to one-hot
        y_true_onehot = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])
        y_true_onehot = tf.cast(y_true_onehot, tf.float32)

        # If labels were already one-hot (shape mismatch fix)
        if len(y_true.shape) > 1 and y_true.shape[-1] == y_pred.shape[-1]:
            y_true_onehot = tf.cast(y_true, tf.float32)

        # Compute cross-entropy component
        cross_entropy = -y_true_onehot * tf.math.log(y_pred)

        # Compute focal weight: (1 - p_t)^gamma
        p_t = tf.reduce_sum(y_true_onehot * y_pred, axis=-1, keepdims=True)
        focal_weight = tf.pow(1.0 - p_t, gamma)

        # Apply per-class alpha
        if alpha_tensor is not None:
            alpha_weight = tf.reduce_sum(y_true_onehot * alpha_tensor, axis=-1, keepdims=True)
            focal_weight = alpha_weight * focal_weight

        loss = focal_weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

    return _focal_loss


# Instantiate with our computed class weights
loss_fn = focal_loss(gamma=CONFIG["focal_gamma"], alpha=CONFIG["focal_alpha"])
print(f"\nFocal Loss created:")
print(f"  gamma = {CONFIG['focal_gamma']}")
print(f"  alpha = {CONFIG['focal_alpha']}")

# %%
# ============================================================
# SECTION 4: PHASE 1 — TRAIN HEAD ONLY (backbone frozen)
# ============================================================
# - Backbone completely frozen
# - Only SE block + classification head are trainable
# - AdamW optimizer, LR = 1e-4
# - 10 epochs to warm up the head
# - Checkpoint on best val_recall

# ── Recall Callback (Keras 3 compatible) ─────────────────────
# Custom Metric subclasses with per-class variable tracking break in
# Keras 3 / TF 2.20 during symbolic graph tracing. Instead we compute
# per-class recall via a lightweight callback at each epoch end.

class RecallCallback(tf.keras.callbacks.Callback):
    """Compute and log per-class recall at each epoch end."""
    def __init__(self, val_ds, class_names, tumor_indices=None):
        super().__init__()
        self.val_ds = val_ds
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.tumor_indices = tumor_indices or [0, 1, 3]

    def on_epoch_end(self, epoch, logs=None):
        all_true, all_pred = [], []
        for images, labels in self.val_ds:
            preds = self.model(images, training=False)
            all_pred.extend(tf.argmax(preds, axis=-1).numpy())
            all_true.extend(labels.numpy())
        all_true, all_pred = np.array(all_true), np.array(all_pred)

        # Per-class recall
        recalls = []
        for i in range(self.num_classes):
            mask = all_true == i
            r = (all_pred[mask] == i).sum() / mask.sum() if mask.sum() > 0 else 0.0
            recalls.append(r)

        tumor_r = np.mean([recalls[i] for i in self.tumor_indices])
        overall_r = np.mean(recalls)

        # Inject into logs so ModelCheckpoint / EarlyStopping can see them
        if logs is not None:
            logs["val_recall"] = overall_r
            logs["val_tumor_recall"] = tumor_r

        print(f"  ↳ val_recall={overall_r:.4f}  val_tumor_recall={tumor_r:.4f}  "
              f"[{', '.join(f'{n}={recalls[i]:.3f}' for i, n in enumerate(self.class_names))}]")


# ── Compile ──────────────────────────────────────────────────
optimizer_phase1 = tf.keras.optimizers.AdamW(
    learning_rate=CONFIG["phase1_lr"],
    weight_decay=CONFIG["weight_decay"],
)

model.compile(
    optimizer=optimizer_phase1,
    loss=loss_fn,
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
)

# ── Callbacks ────────────────────────────────────────────────
recall_cb = RecallCallback(val_ds, CONFIG["class_names"])

best_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(CONFIG["checkpoint_dir"], "best_model.keras"),
    monitor="val_recall",
    mode="max",
    save_best_only=True,
    verbose=1,
)

epoch_snapshot = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(CONFIG["checkpoint_dir"], "epoch_{epoch:02d}_valAcc_{val_accuracy:.4f}.keras"),
    verbose=0,
)

phase1_callbacks = [recall_cb, best_checkpoint, epoch_snapshot]

print("\n" + "=" * 60)
print("PHASE 1: Training Head Only (backbone frozen)")
print("=" * 60)
print(f"  Trainable params : {sum(tf.keras.backend.count_params(w) for w in model.trainable_weights):,}")
print(f"  Epochs           : {CONFIG['phase1_epochs']}")
print(f"  Learning rate    : {CONFIG['phase1_lr']}")

history_phase1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=CONFIG["phase1_epochs"],
    callbacks=phase1_callbacks,
    verbose=1,
)

# %%
# ============================================================
# SECTION 5: PHASE 2 — FINE-TUNING (unfreeze last 40% of backbone)
# ============================================================
# - Unfreeze the last 40% of backbone layers
# - Lower learning rate (1e-5) to prevent catastrophic forgetting
# - ReduceLROnPlateau + EarlyStopping for optimal convergence

# ── Unfreeze backbone ────────────────────────────────────────
backbone.trainable = True
total_layers = len(backbone.layers)
freeze_until = int(total_layers * (1 - CONFIG["fine_tune_percent"]))

for layer in backbone.layers[:freeze_until]:
    layer.trainable = False

trainable_count = sum(1 for l in backbone.layers if l.trainable)
print(f"\n{'=' * 60}")
print(f"PHASE 2: Fine-Tuning")
print(f"{'=' * 60}")
print(f"  Total backbone layers   : {total_layers}")
print(f"  Frozen layers           : {freeze_until}")
print(f"  Unfrozen backbone layers: {trainable_count}")
print(f"  Learning rate           : {CONFIG['phase2_lr']}")

# ── Recompile with lower LR ─────────────────────────────────
optimizer_phase2 = tf.keras.optimizers.AdamW(
    learning_rate=CONFIG["phase2_lr"],
    weight_decay=CONFIG["weight_decay"],
)

model.compile(
    optimizer=optimizer_phase2,
    loss=loss_fn,
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
)

# ── Callbacks ────────────────────────────────────────────────
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=CONFIG["reduce_lr_patience"],
    min_lr=1e-7,
    verbose=1,
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_recall",
    mode="max",
    patience=CONFIG["early_stop_patience"],
    restore_best_weights=True,
    verbose=1,
)

phase2_callbacks = [recall_cb, best_checkpoint, epoch_snapshot, reduce_lr, early_stop]

history_phase2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=CONFIG["phase2_epochs"],
    initial_epoch=CONFIG["phase1_epochs"],
    callbacks=phase2_callbacks,
    verbose=1,
)

# %%
# ============================================================
# SECTION 6: EVALUATION
# ============================================================
# - Full test set evaluation
# - Per-class precision, recall, F1-score
# - Confusion matrix visualization
# - ROC-AUC per class
# - Tumor recall printed separately


def evaluate_model(model, test_ds, class_names):
    """Run full evaluation on the test set."""
    print("\n" + "=" * 60)
    print("EVALUATION ON TEST SET")
    print("=" * 60)

    # Gather predictions
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)
        all_probs.append(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.vstack(all_probs)

    # ── Overall Accuracy ─────────────────────────────────────
    accuracy = np.mean(all_preds == all_labels)
    print(f"\nOverall Test Accuracy: {accuracy * 100:.2f}%")

    # ── Classification Report ────────────────────────────────
    print("\nClassification Report:")
    print("-" * 60)
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report)

    # ── Tumor Recall (CLINICAL PRIORITY) ─────────────────────
    tumor_classes = {"glioma": 0, "meningioma": 1, "pituitary": 3}
    print("🔬 TUMOR RECALL (Clinical Priority):")
    print("-" * 40)
    for name, idx in tumor_classes.items():
        mask = all_labels == idx
        if mask.sum() > 0:
            recall = (all_preds[mask] == idx).sum() / mask.sum()
            print(f"  {name:15s}: {recall * 100:.2f}% ({(all_preds[mask] == idx).sum()}/{mask.sum()})")

    # Average tumor recall
    tumor_recalls = []
    for name, idx in tumor_classes.items():
        mask = all_labels == idx
        if mask.sum() > 0:
            tumor_recalls.append((all_preds[mask] == idx).sum() / mask.sum())
    avg_tumor_recall = np.mean(tumor_recalls)
    print(f"\n  Average Tumor Recall: {avg_tumor_recall * 100:.2f}%")

    # ── Confusion Matrix ─────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True Label",
        title="Confusion Matrix — EfficientNetV2-S + SE",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(os.path.join(CONFIG["checkpoint_dir"], "confusion_matrix.png"), dpi=150)
    plt.show()

    # ── ROC-AUC per class ────────────────────────────────────
    labels_bin = label_binarize(all_labels, classes=list(range(len(class_names))))

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    for i, (name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curves — Per Class")
    ax.legend(loc="lower right")
    fig.tight_layout()
    plt.savefig(os.path.join(CONFIG["checkpoint_dir"], "roc_curves.png"), dpi=150)
    plt.show()

    # Macro AUC
    try:
        macro_auc = roc_auc_score(labels_bin, all_probs, multi_class="ovr", average="macro")
        print(f"\nMacro ROC-AUC: {macro_auc:.4f}")
    except Exception as e:
        print(f"\nCould not compute macro AUC: {e}")

    return {
        "accuracy": float(accuracy),
        "report": report,
        "confusion_matrix": cm.tolist(),
        "avg_tumor_recall": float(avg_tumor_recall),
    }


eval_results = evaluate_model(model, test_ds, CONFIG["class_names"])

# %%
# ============================================================
# SECTION 7: GRAD-CAM VISUALIZATION
# ============================================================
# Generate Grad-CAM heatmaps from the final convolution layer
# to verify tumor region activation.
#
# Compares SE-enhanced model vs. what a baseline would highlight.


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate Grad-CAM heatmap for a given image and model.

    Handles nested models (backbone wrapped inside a functional model) by
    walking the layer graph to find the inner model that owns the target conv layer.
    """
    # Check if the model uses a nested backbone
    inner_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            try:
                layer.get_layer(last_conv_layer_name)
                inner_model = layer
                break
            except ValueError:
                continue

    if inner_model is not None:
        # ── Nested model path ────────────────────────────────
        last_conv_layer = inner_model.get_layer(last_conv_layer_name)
        grad_base_model = tf.keras.Model(
            inner_model.inputs,
            [last_conv_layer.output, inner_model.output],
        )

        with tf.GradientTape() as tape:
            conv_outputs, base_outputs = grad_base_model(img_array)
            tape.watch(conv_outputs)

            # Forward through remaining outer layers
            x = base_outputs
            inner_idx = model.layers.index(inner_model)
            for layer in model.layers[inner_idx + 1:]:
                x = layer(x)
            preds = x

            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
        return heatmap.numpy()
    else:
        # ── Flat model path ──────────────────────────────────
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output],
        )

        with tf.GradientTape() as tape:
            conv_outputs, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
        return heatmap.numpy()


def overlay_heatmap(img, heatmap, alpha=0.4):
    """Overlay a Grad-CAM heatmap onto the original image."""
    if hasattr(img, "size"):
        width, height = img.size
        img_array = np.array(img)
    else:
        height, width = img.shape[:2]
        img_array = img

    heatmap_resized = cv2.resize(np.uint8(255 * heatmap), (width, height))
    heatmap_bgr = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    superimposed = heatmap_rgb * alpha + img_array
    return np.clip(superimposed, 0, 255).astype("uint8")


# ── Identify the target conv layer ───────────────────────────
# EfficientNetV2-S last conv layer is typically 'top_conv'
GRADCAM_LAYER = "top_conv"

# Verify the layer exists
try:
    backbone.get_layer(GRADCAM_LAYER)
    print(f"GradCAM target layer: '{GRADCAM_LAYER}' found in backbone.")
except ValueError:
    # Fallback: find the last Conv2D layer in the backbone
    for layer in reversed(backbone.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            GRADCAM_LAYER = layer.name
            break
    print(f"GradCAM fallback layer: '{GRADCAM_LAYER}'")

# ── Generate GradCAM for sample images ───────────────────────
print("\nGenerating Grad-CAM visualizations...")

# Pick one sample per class from the test set
sample_images = {}
sample_raw_images = {}
for class_idx, class_name in enumerate(CONFIG["class_names"]):
    for images, labels in test_ds:
        mask = labels.numpy() == class_idx
        if mask.any():
            idx = np.where(mask)[0][0]
            sample_images[class_name] = images[idx:idx+1]

            # Get the raw (un-preprocessed) image for display
            raw_img = images[idx].numpy()
            # Reverse EfficientNetV2 preprocessing (approximate)
            raw_display = ((raw_img + 1.0) * 127.5).astype(np.uint8)
            sample_raw_images[class_name] = raw_display
            break

# Plot GradCAM for each class
fig, axes = plt.subplots(len(CONFIG["class_names"]), 3, figsize=(12, 4 * len(CONFIG["class_names"])))

for row, class_name in enumerate(CONFIG["class_names"]):
    img_tensor = sample_images[class_name]
    raw_img = sample_raw_images[class_name]

    # Prediction
    pred = model.predict(img_tensor, verbose=0)
    pred_class = CONFIG["class_names"][np.argmax(pred)]
    pred_conf = np.max(pred) * 100

    # Grad-CAM
    heatmap = make_gradcam_heatmap(img_tensor, model, GRADCAM_LAYER)
    superimposed = overlay_heatmap(raw_img, heatmap)

    # Plot: Original | Heatmap | Overlay
    axes[row, 0].imshow(raw_img)
    axes[row, 0].set_title(f"True: {class_name}", fontsize=11)
    axes[row, 0].axis("off")

    axes[row, 1].imshow(heatmap, cmap="jet")
    axes[row, 1].set_title("Grad-CAM Heatmap", fontsize=11)
    axes[row, 1].axis("off")

    axes[row, 2].imshow(superimposed)
    axes[row, 2].set_title(f"Pred: {pred_class} ({pred_conf:.1f}%)", fontsize=11)
    axes[row, 2].axis("off")

fig.suptitle("Grad-CAM — EfficientNetV2-S + SE Attention", fontsize=14, fontweight="bold")
fig.tight_layout()
plt.savefig(os.path.join(CONFIG["checkpoint_dir"], "gradcam_visualizations.png"), dpi=150)
plt.show()

print("✅ Grad-CAM visualizations saved.")

# %%
# ============================================================
# SECTION 8: MODEL EXPORT & HISTORY
# ============================================================
# - Save final model in Keras format
# - Save training history as JSON
# - Print final summary

# ── Save final model ─────────────────────────────────────────
final_model_path = os.path.join(CONFIG["model_save_dir"], "efficientnetv2s_se.keras")
model.save(final_model_path)
print(f"\n✅ Final model saved: {final_model_path}")

# ── Save training history ────────────────────────────────────
# Merge Phase 1 and Phase 2 histories
combined_history = {}
for key in history_phase1.history:
    combined_history[key] = [float(v) for v in history_phase1.history[key]]
for key in history_phase2.history:
    if key in combined_history:
        combined_history[key].extend([float(v) for v in history_phase2.history[key]])
    else:
        combined_history[key] = [float(v) for v in history_phase2.history[key]]

history_path = os.path.join(CONFIG["checkpoint_dir"], "training_history.json")
with open(history_path, "w") as f:
    json.dump(combined_history, f, indent=2)
print(f"✅ Training history saved: {history_path}")

# ── Plot training curves ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Accuracy
axes[0].plot(combined_history.get("accuracy", []), label="Train Acc", color="#3498db")
axes[0].plot(combined_history.get("val_accuracy", []), label="Val Acc", color="#e74c3c")
axes[0].axvline(x=CONFIG["phase1_epochs"] - 1, color="gray", linestyle="--", alpha=0.5, label="Phase 2 start")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(combined_history.get("loss", []), label="Train Loss", color="#3498db")
axes[1].plot(combined_history.get("val_loss", []), label="Val Loss", color="#e74c3c")
axes[1].axvline(x=CONFIG["phase1_epochs"] - 1, color="gray", linestyle="--", alpha=0.5, label="Phase 2 start")
axes[1].set_title("Focal Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Recall
axes[2].plot(combined_history.get("recall", []), label="Train Recall", color="#3498db")
axes[2].plot(combined_history.get("val_recall", []), label="Val Recall", color="#e74c3c")
if "tumor_recall" in combined_history:
    axes[2].plot(combined_history.get("tumor_recall", []), label="Train Tumor Recall", color="#2ecc71", linestyle="--")
    axes[2].plot(combined_history.get("val_tumor_recall", []), label="Val Tumor Recall", color="#f39c12", linestyle="--")
axes[2].axvline(x=CONFIG["phase1_epochs"] - 1, color="gray", linestyle="--", alpha=0.5, label="Phase 2 start")
axes[2].set_title("Recall (Overall & Tumor)")
axes[2].set_xlabel("Epoch")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

fig.suptitle("Training Curves — EfficientNetV2-S + SE", fontsize=14, fontweight="bold")
fig.tight_layout()
plt.savefig(os.path.join(CONFIG["checkpoint_dir"], "training_curves.png"), dpi=150)
plt.show()

# ── Save hyperparameters ─────────────────────────────────────
config_save = {k: str(v) if not isinstance(v, (int, float, str, list, bool)) else v
               for k, v in CONFIG.items()}
config_path = os.path.join(CONFIG["checkpoint_dir"], "hyperparameters.json")
with open(config_path, "w") as f:
    json.dump(config_save, f, indent=2)
print(f"✅ Hyperparameters saved: {config_path}")

# ── Final Summary ────────────────────────────────────────────
print("\n" + "=" * 60)
print("TRAINING COMPLETE — ENCEPHLO v3")
print("=" * 60)
print(f"  Model           : {CONFIG['model_name']}")
print(f"  Final model     : {final_model_path}")
print(f"  Best checkpoint : {os.path.join(CONFIG['checkpoint_dir'], 'best_model.keras')}")
print(f"  Test Accuracy   : {eval_results['accuracy'] * 100:.2f}%")
print(f"  Avg Tumor Recall: {eval_results['avg_tumor_recall'] * 100:.2f}%")
print(f"  GradCAM layer   : {GRADCAM_LAYER}")
print("=" * 60)
print("Ready for integration into Encephlo app.py")
