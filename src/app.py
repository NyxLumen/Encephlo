import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import make_gradcam_heatmap, overlay_heatmap

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

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image
        
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    cropped = image_np[y:y+h, x:x+w]
    return Image.fromarray(cropped)

with st.sidebar:
    st.title("🧠 Encephlo")
    st.write("### AI Diagnostic Suite")
    uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "png", "jpeg"])
    
    st.divider()
    st.info("**Model Classes:**\n1. Glioma\n2. Meningioma\n3. No Tumor\n4. Pituitary")

st.title("Neural Tumor Detection Interface")

model = load_prediction_model()

if model is None:
    st.error("🚨 Model not found!")
    st.write("Please ensure your model file is at: `Encephlo/models/model.h5`")
    st.stop()

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    original_image = Image.open(uploaded_file).convert('RGB')
    
    with col1:
        st.header("Original Scan")
        st.image(original_image, use_container_width=True, caption="Patient Input")

    with col2:
        st.header("AI Analysis")
        
        with st.spinner('Isolating Brain Region & Analyzing...'):
            cropped_image = crop_brain_contour(original_image)
            
            with st.expander("Show AI Input (Preprocessed)"):
                st.image(cropped_image, caption="Cropped Brain Region", width=150)
            
            img_resized = cropped_image.resize((224, 224))
            
            img_array = np.array(img_resized)
            img_array = img_array / 255.0
            img_batch = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_batch)
            
            class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
            predicted_index = np.argmax(predictions)
            predicted_label = class_names[predicted_index]
            confidence = predictions[0][predicted_index] * 100

            if predicted_label == "No Tumor":
                st.success(f"DIAGNOSIS: {predicted_label}")
            else:
                st.error(f"DIAGNOSIS: {predicted_label}")
            
            st.metric("Confidence Score", f"{confidence:.2f}%")
            
            st.divider()
            st.subheader("Visual Explanation")
            try:
                last_conv_layer = "conv5_block3_out" 
                
                heatmap = make_gradcam_heatmap(
                    img_batch, 
                    model, 
                    last_conv_layer_name=last_conv_layer
                )
                
                if heatmap is not None:
                    final_overlay = overlay_heatmap(img_resized, heatmap)
                    st.image(
                        final_overlay, 
                        caption="Grad-CAM Attention Map", 
                        use_container_width=True
                    )
                else:
                    st.warning("Heatmap generation returned None.")
                    
            except Exception as e:
                st.warning(f"Could not generate heatmap: {e}")
                st.caption("Common Error: Check 'last_conv_layer' name in app.py")

else:
    st.info("Waiting for MRI Scan... Please upload a file.")
