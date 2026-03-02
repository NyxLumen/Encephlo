import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from PIL import Image

# Setup Paths & Config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(SCRIPT_DIR, '..', 'MRI images', 'Testing')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'models', 'densenet121.keras')
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMG_SIZE = (224, 224)
TARGET_LAYER = 'conv5_block16_concat'

print("Loading DenseNet121 model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

# Quick Grad-CAM implementation
def get_gradcam_heatmap(img_array, mdl, layer_name):
    inner = next((l for l in mdl.layers if isinstance(l, tf.keras.Model)), None)
    if inner:
        lyr = inner.get_layer(layer_name)
        gm  = tf.keras.Model(inner.inputs, [lyr.output, inner.output])
        with tf.GradientTape() as tape:
            conv_out, base_out = gm(img_array)
            tape.watch(conv_out)
            x = base_out
            idx = mdl.layers.index(inner)
            for l in mdl.layers[idx+1:]:
                x = l(x)
            cls_ch = x[:, tf.argmax(x[0])]
        grads = tape.gradient(cls_ch, conv_out)
        pw    = tf.reduce_mean(grads, axis=(0,1,2))
        hm    = conv_out[0] @ pw[..., tf.newaxis]
        hm    = tf.squeeze(hm)
        hm    = tf.maximum(hm, 0) / tf.math.reduce_max(hm)
        return hm.numpy()
    return None

# Pick 1 random image from each class
selected_images = []
for cls in CLASS_NAMES:
    cls_folder = os.path.join(TEST_DIR, cls)
    files = os.listdir(cls_folder)
    sel = random.choice(files)
    selected_images.append({
        'path': os.path.join(cls_folder, sel),
        'true_class': cls
    })

# Prediction Plot
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Encephlo DenseNet121 - 4 Random Test Samples & Grad-CAM', fontsize=16, fontweight='bold')

for i, img_data in enumerate(selected_images):
    # Load Image
    img = Image.open(img_data['path']).convert('RGB').resize(IMG_SIZE)
    img_arr = np.array(img)
    prep_arr = preprocess_input(np.expand_dims(img_arr.copy(), axis=0))
    
    # Predict
    preds = model.predict(prep_arr, verbose=0)
    pred_idx = np.argmax(preds[0])
    pred_class = CLASS_NAMES[pred_idx]
    conf = preds[0][pred_idx] * 100
    is_correct = "✅" if pred_class == img_data['true_class'] else "❌"
    
    # Heatmap
    hm = get_gradcam_heatmap(prep_arr, model, TARGET_LAYER)
    hm_resized = cv2.resize(np.uint8(255 * hm), (224, 224))
    hm_bgr = cv2.applyColorMap(hm_resized, cv2.COLORMAP_JET)
    hm_rgb = cv2.cvtColor(hm_bgr, cv2.COLOR_BGR2RGB)
    overlay = np.clip(hm_rgb * 0.4 + img_arr, 0, 255).astype('uint8')
    
    # Display Original
    axes[0, i].imshow(img)
    axes[0, i].set_title(f"True: {img_data['true_class']}\nFile: {os.path.basename(img_data['path'])}", fontsize=10)
    axes[0, i].axis('off')
    
    # Display Overlay
    axes[1, i].imshow(overlay)
    axes[1, i].set_title(f"{is_correct} Pred: {pred_class}\nConf: {conf:.1f}%", fontsize=11, fontweight='bold')
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig('models/plots/4_random_samples.png', dpi=120)
print("\nSaved sample plot to models/plots/4_random_samples.png")

# Now, we should compute confusion matrix on entire testing set or use the saved one.
# For speed, I'll print the metrics from our previous test generator if needed, 
# but the easiest is loading the saved confusion matrix plot or generating it.
print("\nEvaluation metrics are covered fully in the saved confusion matrix plots.")
