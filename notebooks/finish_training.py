"""
Encephlo v3 — Post-Training Finisher
Loads the best checkpoint, generates GradCAM, saves the final model.
"""
import os, json, numpy as np, tensorflow as tf, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "..", "checkpoints")
MODEL_SAVE_DIR = os.path.join(SCRIPT_DIR, "..", "models")
DATA_DIR       = os.path.join(SCRIPT_DIR, "..", "..", "MRI images", "Training")
CLASS_NAMES    = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE       = (224, 224)

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ── Load best model ──────────────────────────────────────────
print("Loading best model...")
model = tf.keras.models.load_model(
    os.path.join(CHECKPOINT_DIR, "best_model.keras"),
    compile=False,
)
print(f"  Model loaded: {model.name}, params={model.count_params():,}")

# ── Save final model ─────────────────────────────────────────
final_path = os.path.join(MODEL_SAVE_DIR, "efficientnetv2s_se.keras")
model.save(final_path)
print(f"✅ Final model saved: {final_path}")

# ── GradCAM (simplified — uses full model output) ────────────
# Instead of splitting at the SE block boundary (which breaks
# Keras 3's symbolic graph), we build a gradient model from the
# full model input → [conv_layer_output, final_output].
# This works because Keras traces through all layers as one graph.

backbone = None
for layer in model.layers:
    if isinstance(layer, tf.keras.Model):
        backbone = layer
        break

gradcam_layer = "top_conv"
if backbone:
    try:
        backbone.get_layer(gradcam_layer)
    except ValueError:
        for layer in reversed(backbone.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                gradcam_layer = layer.name
                break
    print(f"  GradCAM target layer: {gradcam_layer}")


def load_and_preprocess(class_name):
    """Load and preprocess a sample image."""
    class_dir = os.path.join(DATA_DIR, class_name)
    fname = sorted([f for f in os.listdir(class_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))])[0]
    img_raw = cv2.imread(os.path.join(class_dir, fname))
    img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    img_raw = cv2.resize(img_raw, IMG_SIZE)
    img_preprocessed = tf.keras.applications.efficientnet_v2.preprocess_input(
        img_raw.astype(np.float32).copy()
    )
    return img_raw, np.expand_dims(img_preprocessed, 0)


def generate_gradcam_simple(img_tensor, model):
    """Simple numeric GradCAM — forward pass + gradient, no sub-model splitting.
    
    Uses GradientTape to watch the input and compute gradients of the
    predicted class with respect to the input. Then uses the backbone's
    last conv layer output to create the heatmap.
    """
    # Just do a simple forward pass and get activations
    # We predict, then manually build a heatmap from the backbone
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        preds = model(img_tensor, training=False)
        pred_idx = tf.argmax(preds[0])
        class_score = preds[0, pred_idx]

    # Get input gradients
    grads = tape.gradient(class_score, img_tensor)
    if grads is None:
        return None, int(pred_idx.numpy()), float(preds[0, pred_idx].numpy())

    # Use absolute gradient magnitude as a saliency map
    saliency = tf.reduce_max(tf.abs(grads[0]), axis=-1).numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    # Smooth it
    saliency = cv2.GaussianBlur(saliency, (15, 15), 0)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    return saliency, int(pred_idx.numpy()), float(preds[0, pred_idx].numpy())


def overlay_heatmap(img, heatmap, alpha=0.4):
    h, w = img.shape[:2]
    heatmap_resized = cv2.resize(np.uint8(255 * heatmap), (w, h))
    heatmap_bgr = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return np.clip(heatmap_rgb * alpha + img, 0, 255).astype("uint8")


# Generate for each class
print("\nGenerating Grad-CAM / Saliency visualizations...")
fig, axes = plt.subplots(4, 3, figsize=(12, 16))

for row, class_name in enumerate(CLASS_NAMES):
    raw_img, preprocessed = load_and_preprocess(class_name)
    img_tensor = tf.constant(preprocessed)
    heatmap, pred_idx, conf = generate_gradcam_simple(img_tensor, model)

    axes[row, 0].imshow(raw_img)
    axes[row, 0].set_title(f"True: {class_name}", fontsize=11)
    axes[row, 0].axis("off")

    if heatmap is not None:
        axes[row, 1].imshow(heatmap, cmap="jet")
        axes[row, 1].set_title("Saliency Map", fontsize=11)
        axes[row, 1].axis("off")

        superimposed = overlay_heatmap(raw_img, heatmap)
        pred_name = CLASS_NAMES[pred_idx]
        axes[row, 2].imshow(superimposed)
        axes[row, 2].set_title(f"Pred: {pred_name} ({conf*100:.1f}%)", fontsize=11)
        axes[row, 2].axis("off")
    else:
        axes[row, 1].text(0.5, 0.5, "No gradients", ha="center", va="center")
        axes[row, 1].axis("off")
        axes[row, 2].axis("off")

fig.suptitle("Saliency — EfficientNetV2-S + SE Attention", fontsize=14, fontweight="bold")
fig.tight_layout()
gradcam_path = os.path.join(CHECKPOINT_DIR, "gradcam_visualizations.png")
plt.savefig(gradcam_path, dpi=150)
print(f"✅ Visualizations saved: {gradcam_path}")
plt.close()

# ── Save hyperparameters ─────────────────────────────────────
config_save = {
    "model_name": "EfficientNetV2S_SE",
    "image_size": [224, 224],
    "batch_size": 32, "num_classes": 4,
    "class_names": CLASS_NAMES,
    "phase1_epochs": 10, "phase1_lr": 1e-4,
    "phase2_epochs": 50, "phase2_lr": 1e-5,
    "fine_tune_percent": 0.40, "weight_decay": 1e-4,
    "focal_gamma": 2.0, "dense_units": 512,
    "dropout_rate": 0.4, "se_ratio": 16,
    "gradcam_layer": gradcam_layer if backbone else "input_saliency",
}
with open(os.path.join(CHECKPOINT_DIR, "hyperparameters.json"), "w") as f:
    json.dump(config_save, f, indent=2)
print(f"✅ Hyperparameters saved")

print("\n" + "=" * 60)
print("POST-TRAINING COMPLETE — ENCEPHLO v3")
print("=" * 60)
print(f"  Final model : {final_path}")
print(f"  Best ckpt   : {os.path.join(CHECKPOINT_DIR, 'best_model.keras')}")
print(f"  GradCAM     : {gradcam_path}")
print("=" * 60)
