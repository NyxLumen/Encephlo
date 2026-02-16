import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import sys
import os
import time

st.set_page_config(
    page_title="Encephlo | Neural Interface",
    page_icon="🤓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .block-container { padding-top: 75px; padding-bottom: 2rem; }
    div[data-testid="stMetric"] {
        margin-top: 15px;
        background-color: #1E2329;
        border: 1px solid #2B313A;
        padding: 15px;
        border-radius: 10px;
        color: #FFFFFF;
    }
    img { border-radius: 15px; }
    .stAlert { border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from utils import make_gradcam_heatmap, overlay_heatmap
    from report import create_pdf
except ImportError:
    st.error("⚠️ System Error: Missing modules (utils.py or report.py)")
    st.stop()

# --- CONFIGURATION (UPDATED FOR MULTI-MODEL SUPPORT) ---
# Each model has its own unique "eyes" (preprocessing) and "map" (layer name)
ENSEMBLE_CONFIG = {
    "EfficientNetB0": {
        "path": "models/best_model.keras",          # Your new high-acc model
        "layer": "efficientnetb0",                  # Layer name for *this* specific model structure
        "preprocess": tf.keras.applications.efficientnet.preprocess_input
    },
    "ResNet50": {
        "path": "models/resnet50.h5",               # Placeholder path
        "layer": "conv5_block3_out",                # Standard ResNet last conv layer
        "preprocess": tf.keras.applications.resnet50.preprocess_input
    },
    "MobileNetV2": {
        "path": "models/mobilenetv2.h5",            # Placeholder path
        "layer": "out_relu",                        # Standard MobileNetV2 last conv layer
        "preprocess": tf.keras.applications.mobilenet_v2.preprocess_input
    },
    "DenseNet121": {
        "path": "models/densenet121.h5",            # Placeholder path
        "layer": "relu",                            # Standard DenseNet last conv layer
        "preprocess": tf.keras.applications.densenet.preprocess_input
    }
}

@st.cache_resource
def load_ensemble():
    loaded_models = {}
    for name, config in ENSEMBLE_CONFIG.items():
        if os.path.exists(config["path"]):
            try:
                # Compile=False is safer for inference
                m = tf.keras.models.load_model(config["path"], compile=False)
                loaded_models[name] = {"model": m, "config": config}
            except Exception as e:
                st.error(f"Failed to load {name}: {e}")
        else:
            st.warning(f"Model file not found: {config['path']}")
    return loaded_models

def crop_brain_contour(image):
    image_np = np.array(image)
    if len(image_np.shape) == 3: gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else: gray = image_np
    
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: return image
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return Image.fromarray(gray[y:y+h, x:x+w])

# --- UI SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=60)
    st.title("ENCEPHLO")
    st.caption("v2.1 | Neural Diagnostic Suite")
    st.divider()
    
    uploaded_file = st.file_uploader("📂 Load MRI Scan", type=["jpg", "png", "jpeg"])
    
    st.divider()
    st.write("⚙️ **System Config**")
    
    models = load_ensemble()
    status_color = "🟢" if len(models) >= 1 else "🔴"
    st.write(f"System State: {status_color} {len(models)} Active")
    st.write("Engine: EfficientNet B0 (Optimized)")

# --- MAIN DASHBOARD ---
if uploaded_file is None:
    st.markdown("<h1 style='text-align: center; color: #555;'>Ready to Scan</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Upload a patient MRI to begin analysis.</p>", unsafe_allow_html=True)

else:
    original_image = Image.open(uploaded_file).convert('RGB')
    
    # 1. PREP (Otsu & Resize)
    cropped_image = crop_brain_contour(original_image)
    img_resized = cropped_image.resize((224, 224))
    img_converted = img_resized.convert('RGB')
    img_array = np.array(img_converted) # Kept 0-255 for standard prep
    
    # 2. ENSEMBLE INFERENCE (Soft Voting)
    start_time = time.time()
    predictions_list = []
    
    if not models:
        st.error("No models loaded. Please check model path in `app.py`.")
        st.stop()
        
    for name, data in models.items():
        # Apply model-specific preprocessing
        # Note: EfficientNet preprocess expects [0-255], ResNet usually [0-1] or specific mean subtraction
        # We use strict configuration from ENSEMBLE_CONFIG
        img_prepped = data["config"]["preprocess"](np.expand_dims(img_array.copy(), axis=0))
        preds = data["model"].predict(img_prepped, verbose=0)
        predictions_list.append(preds)
        
    end_time = time.time()
    inference_time = round((end_time - start_time) * 1000, 2)
    
    # Average the predictions
    final_preds = np.mean(predictions_list, axis=0) if predictions_list else np.zeros((1, 4))
    
    # Class names must match training (alphabetical usually for flow_from_directory)
    # ['glioma', 'meningioma', 'notumor', 'pituitary']
    class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    pred_idx = np.argmax(final_preds)
    result = class_names[pred_idx]
    confidence = final_preds[0][pred_idx] * 100

    # --- ROW 1: HUD ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Diagnosis", result, delta_color="inverse")
    c2.metric("Confidence", f"{confidence:.1f}%")
    c3.metric("Latency", f"{inference_time} ms")
    c4.metric("Active Models", str(len(models))) 
    
    st.divider()

    # --- ROW 2: THE COUNCIL VIEW ---
    st.subheader("🧠 Explainable AI (Grad-CAM)")
    
    col_input, col_m1 = st.columns([1, 1])
    
    with col_input:
        st.image(img_converted, use_container_width=True, caption="Processed Input")
    
    # Generate Heatmaps
    anchor_heatmap_img = None # Save one for the PDF report
    
    for i, (name, data) in enumerate(models.items()):
        # We only show the first model's heatmap to keep UI clean if multiple identical models
        if i > 0 and name.startswith("Model_"): continue 
        
        with col_m1:
            try:
                img_prepped = data["config"]["preprocess"](np.expand_dims(img_array.copy(), axis=0))
                
                # Grad-CAM
                heatmap = make_gradcam_heatmap(img_prepped, data["model"], data["config"]["layer"])
                final_img = overlay_heatmap(img_converted, heatmap)
                
                st.image(final_img, use_container_width=True, caption=f"{name} Attention Map")
                anchor_heatmap_img = final_img
                
            except Exception as e:
                st.warning(f"Heatmap failed for {name}: {e}")

    # --- ROW 3: ACTION BAR ---
    st.divider()
    
    if result == "No Tumor":
        st.success(f"✅ CONSENSUS REACHED: No anomalies detected ({confidence:.1f}%)")
    else:
        st.error(f"⚠️ CRITICAL: {result.upper()} detected ({confidence:.1f}%)")

    # PDF Button
    col_btn, _ = st.columns([1, 4])
    with col_btn:
        if st.button("📄 Generate Report"):
            with st.spinner("Compiling..."):
                img_converted.save("temp_input.jpg")
                heatmap_path = None
                
                if anchor_heatmap_img is not None:
                    Image.fromarray(anchor_heatmap_img).save("temp_heatmap.jpg")
                    heatmap_path = "temp_heatmap.jpg"
                
                pdf_bytes = create_pdf(result, confidence, "temp_input.jpg", heatmap_path)
                
                if os.path.exists("temp_input.jpg"): os.remove("temp_input.jpg")
                if heatmap_path and os.path.exists(heatmap_path): os.remove(heatmap_path)
                
                st.download_button("📥 Download PDF", pdf_bytes, "Encephlo_Report.pdf", "application/pdf")
