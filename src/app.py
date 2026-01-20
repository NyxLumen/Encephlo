import streamlit as st
from PIL import Image
import time

# 1. Page Config
st.set_page_config(page_title="Encephlo | MRI Diagnostic", layout="wide")

# 2. Sidebar
with st.sidebar:
    st.title("🧠 Encephlo")
    uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "png", "jpeg"])

# 3. Main Area
st.title("Neural Tumor Detection Interface")

if uploaded_file is not None:
    # This just displays the image you uploaded
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Original Scan")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        
    with col2:
        st.header("AI Analysis")
        st.info("⚠️ Model not connected yet. This is just the UI.")

else:
    st.info("Upload an MRI scan from your data folder to test the interface.")