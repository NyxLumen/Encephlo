import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils import make_gradcam_heatmap, overlay_heatmap
except ImportError:
    st.error("⚠️ Error: 'utils.py' not found. Please create it in the src folder.")
    st.stop()

st.set_page_config(
    page_title="Encephlo | AI Diagnostic",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1f2937; border-radius: 5px; padding: 10px; border-left: 5px solid #3b82f6; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_prediction_model():
    model_path = 'models/model.h5'
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)

def crop_brain_contour(image):
    image_np = np.array(image)
    
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image
        
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    cropped = gray[y:y+h, x:x+w]
    
    return Image.fromarray(cropped)

with st.sidebar:
    st.title("🧠 Encephlo")
    st.caption("v1.0.2 (Otsu Corrected)")
    uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "png", "jpeg"])
    st.divider()
    st.info("Classes:\n- Glioma\n- Meningioma\n- No Tumor\n- Pituitary")

st.title("Neural Tumor Detection Interface")

model = load_prediction_model()
if model is None:
    st.error("🚨 Model missing! Place 'model.h5' in the models folder.")
    st.stop()

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    original_image = Image.open(uploaded_file).convert('RGB')
    
    with col1:
        st.header("Original Scan")
        st.image(original_image, use_container_width=True)

    with col2:
        st.header("AI Analysis")
        
        with st.spinner('Preprocessing & Scanning...'):
            cropped_image = crop_brain_contour(original_image)
            
            img_resized = cropped_image.resize((224, 224))
            img_converted = img_resized.convert('RGB')
            
            img_array = np.array(img_converted)
            img_array = img_array / 255.0
            img_batch = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_batch)
            class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
            
            predicted_index = np.argmax(predictions)
            result = class_names[predicted_index]
            confidence = predictions[0][predicted_index] * 100

            if result == "No Tumor":
                st.success(f"DIAGNOSIS: {result}")
            else:
                st.error(f"DIAGNOSIS: {result}")
            
            st.metric("Confidence", f"{confidence:.2f}%")
            
            with st.expander("Debug Info"):
                st.image(img_converted, caption="AI Input (What the model actually saw)", width=150)
                st.write(f"Raw Probabilities: {predictions}")

            st.divider()
            st.subheader("Visual Explanation")
            try:
                heatmap = make_gradcam_heatmap(img_batch, model, "conv5_block3_out")
                final_img = overlay_heatmap(img_converted, heatmap)
                st.image(final_img, caption="Grad-CAM Attention Map", use_container_width=True)
            except Exception as e:
                st.warning(f"Heatmap Error: {e}")

else:
    st.info("Waiting for MRI Scan...")