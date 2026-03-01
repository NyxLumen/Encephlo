"""
Encephlo v3 — External Test Set Evaluation
Evaluates the trained EfficientNetV2S_SE on the separate Testing dataset.
Shows random sample predictions + full accuracy report.
"""
import os, random, numpy as np, tensorflow as tf, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(SCRIPT_DIR, "..", "models", "efficientnetv2s_se.keras")
TEST_DIR    = os.path.join(SCRIPT_DIR, "..", "..", "MRI images", "Testing")
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE    = (224, 224)
SEED        = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Load model ───────────────────────────────────────────────
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print(f"  Model: {model.name}, params={model.count_params():,}")

# ── Collect all test images ──────────────────────────────────
all_paths, all_labels = [], []
for idx, cls in enumerate(CLASS_NAMES):
    cls_dir = os.path.join(TEST_DIR, cls)
    files = sorted([f for f in os.listdir(cls_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    for f in files:
        all_paths.append(os.path.join(cls_dir, f))
        all_labels.append(idx)
    print(f"  {cls:15s}: {len(files)} images")
print(f"  Total test images: {len(all_paths)}")

# ── Predict on ALL test images ───────────────────────────────
print("\nRunning inference on full test set...")

def load_and_preprocess(path):
    img = tf.keras.utils.load_img(path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    arr = tf.keras.applications.efficientnet_v2.preprocess_input(arr)
    return arr

all_preds = []
all_probs = []
batch = []
batch_size = 32

for i, path in enumerate(all_paths):
    batch.append(load_and_preprocess(path))
    if len(batch) == batch_size or i == len(all_paths) - 1:
        batch_arr = np.array(batch)
        probs = model.predict(batch_arr, verbose=0)
        preds = np.argmax(probs, axis=1)
        all_preds.extend(preds)
        all_probs.extend(probs)
        batch = []
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(all_paths)}...")

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# ── Overall Accuracy ─────────────────────────────────────────
accuracy = np.mean(all_preds == all_labels)
print(f"\n{'='*60}")
print(f"EXTERNAL TEST SET ACCURACY: {accuracy * 100:.2f}%")
print(f"{'='*60}")

# ── Per-class report ─────────────────────────────────────────
print("\nClassification Report:")
print("-" * 60)
print(classification_report(all_labels, all_preds,
                            target_names=CLASS_NAMES, digits=4))

# ── Tumor Recall ─────────────────────────────────────────────
tumor_classes = {"glioma": 0, "meningioma": 1, "pituitary": 3}
print("TUMOR RECALL (Clinical Priority):")
print("-" * 40)
tumor_recalls = []
for name, idx in tumor_classes.items():
    mask = all_labels == idx
    recall = (all_preds[mask] == idx).sum() / mask.sum()
    tumor_recalls.append(recall)
    print(f"  {name:15s}: {recall*100:.2f}% ({(all_preds[mask]==idx).sum()}/{mask.sum()})")
print(f"\n  Average Tumor Recall: {np.mean(tumor_recalls)*100:.2f}%")

# ── Random Samples (3 per class) ─────────────────────────────
print("\nGenerating random sample predictions...")
fig, axes = plt.subplots(4, 3, figsize=(12, 16))

for row, cls in enumerate(CLASS_NAMES):
    cls_dir = os.path.join(TEST_DIR, cls)
    files = [f for f in os.listdir(cls_dir)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    samples = random.sample(files, min(3, len(files)))

    for col, fname in enumerate(samples):
        path = os.path.join(cls_dir, fname)
        img = tf.keras.utils.load_img(path, target_size=IMG_SIZE)
        arr = tf.keras.utils.img_to_array(img)
        preprocessed = tf.keras.applications.efficientnet_v2.preprocess_input(arr.copy())
        probs = model.predict(np.expand_dims(preprocessed, 0), verbose=0)
        pred_idx = np.argmax(probs)
        pred_name = CLASS_NAMES[pred_idx]
        conf = probs[0, pred_idx] * 100
        correct = "✓" if pred_idx == row else "✗"

        axes[row, col].imshow(arr.astype(np.uint8))
        color = "green" if pred_idx == row else "red"
        axes[row, col].set_title(
            f"True: {cls}\nPred: {pred_name} ({conf:.1f}%) {correct}",
            fontsize=10, color=color
        )
        axes[row, col].axis("off")

fig.suptitle(f"Random Test Samples — Accuracy: {accuracy*100:.2f}%",
             fontsize=14, fontweight="bold")
fig.tight_layout()
out_path = os.path.join(SCRIPT_DIR, "..", "checkpoints", "test_samples.png")
plt.savefig(out_path, dpi=150)
print(f"✅ Sample predictions saved: {out_path}")
plt.close()

print(f"\n{'='*60}")
print(f"DONE — External Test Accuracy: {accuracy*100:.2f}%")
print(f"{'='*60}")
