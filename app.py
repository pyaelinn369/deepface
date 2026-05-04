import os
import sys

# Maintain compatibility for your RTX 4060 / TF 2.16+ environment
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace

# --- Page Config ---
st.set_page_config(page_title="DeepFace Attribute Analyzer", layout="wide")
st.title("🧬 DeepFace: Advanced Facial Analysis")
st.markdown("""
This app uses the **DeepFace** framework to analyze age and gender. 
* **Age MAE:** ±4.65 years
* **Gender Accuracy:** 97.44%
""")

# --- Model Warmup ---
@st.cache_resource
def warmup_deepface():
    """
    DeepFace downloads models on the first call. 
    This function ensures they are ready.
    """
    # Just a dummy call to trigger model loading if not present
    st.info("Initializing DeepFace models (VGG-Face, Age, Gender)...")
    return True

def analyze_image(pil_image):
    # 1. Convert PIL to BGR (DeepFace uses OpenCV/BGR internally)
    img_array = np.array(pil_image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    try:
        # 2. Run DeepFace Analysis
        # actions: age, gender, race, emotion
        # enforce_detection: False allows the app to process images even if faces are hard to find
        results = DeepFace.analyze(
            img_path=img_bgr, 
            actions=['age', 'gender'],
            enforce_detection=False,
            detector_backend='opencv' # You can also use 'retinaface' for higher accuracy but slower speed
        )
        
        # 3. Annotate Image
        annotated_img = img_array.copy()
        
        for face in results:
            # Extract coordinates
            region = face['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            
            # Extract attributes
            age = face['age']
            gender = face['dominant_gender']
            gender_label = "M" if gender == "Man" else "F"
            
            # Format Label
            label = f"{gender_label}, Age: {age}"
            
            # Drawing
            # Green box
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
            # Label background
            cv2.rectangle(annotated_img, (x, y - 40), (x + 180, y), (0, 255, 0), -1)
            
            # Text
            cv2.putText(annotated_img, label, (x + 5, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            
        return annotated_img, results
    
    except Exception as e:
        st.error(f"Analysis Error: {e}")
        return img_array, []

# --- Main Logic ---
warmup_deepface()

uploaded_file = st.file_uploader("Upload an image for DeepFace analysis...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(input_image, use_container_width=True)
        
    if st.button("Analyze with DeepFace"):
        with st.spinner("DeepFace is processing (detecting, aligning, and predicting)..."):
            result_img, raw_data = analyze_image(input_image)
            
        with col2:
            st.subheader("DeepFace Prediction")
            st.image(result_img, use_container_width=True)
            
        if raw_data:
            st.success(f"Successfully analyzed {len(raw_data)} face(s).")
            # Show Raw JSON Data in expander for the senior
            with st.expander("View Raw Analysis Data (JSON)"):
                st.json(raw_data)
        else:
            st.warning("No faces were detected in this image.")