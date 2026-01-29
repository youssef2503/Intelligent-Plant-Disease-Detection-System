import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import os
from PIL import Image, ImageEnhance

st.set_page_config(
    page_title="EcoVision AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = 'models/final_finetuned_model.keras'
MAPPING_PATH = 'class_indices.json'

TREATMENTS = {
    "Tomato_Spider_mites_Two_spotted_spider_mite": "🕷️ **Diagnosis:** Spider Mites.\n💊 **Treatment:** Use Miticides like Abamectin or Neem Oil. Increase humidity around plants.",
    "Tomato_Late_blight": "🍄 **Diagnosis:** Late Blight.\n💊 **Treatment:** Serious fungal disease! Remove infected parts immediately. Apply Copper-based fungicides.",
    "Tomato_YellowLeaf_Curl_Virus": "virus **Diagnosis:** Yellow Leaf Curl Virus.\n💊 **Treatment:** Transmitted by Whiteflies. Use insect nets and remove infected plants to stop spread.",
    "Tomato_Bacterial_spot": "microbe **Diagnosis:** Bacterial Spot.\n💊 **Treatment:** Apply Copper bactericides. Avoid overhead watering.",
    "Tomato_healthy": "✅ **Diagnosis:** Healthy Plant.\n🌟 **Advice:** Keep maintaining regular watering and good soil nutrition.",
    "Pepper__bell___Bacterial_spot": "microbe **Diagnosis:** Pepper Bacterial Spot.\n💊 **Treatment:** Remove infected leaves. Spray Copper fungicide.",
    "Pepper__bell___healthy": "✅ **Diagnosis:** Healthy Pepper.\n🌟 **Advice:** Good job! Monitor for pests regularly."
}

def get_advice(class_name):
    return TREATMENTS.get(class_name, f"🔍 **Diagnosis:** {class_name}\n💊 **Advice:** Consult a local agricultural expert for specific fungicides.")

@st.cache_resource
def load_learner():
    if not os.path.exists(MODEL_PATH):
        return None
    model = load_model(MODEL_PATH)
    return model

@st.cache_data
def load_classes():
    if not os.path.exists(MAPPING_PATH):
        return None
    with open(MAPPING_PATH, 'r') as f:
        return json.load(f)

model = load_learner()
class_names = load_classes()

st.sidebar.title("🌿 EcoVision Dashboard")
app_mode = st.sidebar.selectbox("Choose Mode", ["🔍 Disease Detection", "📊 Performance Metrics", "ℹ️ About Project"])

# (Disease Detection)
if app_mode == "🔍 Disease Detection":
    st.title("🔍 AI Plant Disease Detector")
    st.markdown("Upload a leaf image to detect diseases instantly.")

    st.sidebar.header("⚙️ Image Enhancement")
    use_enhancement = st.sidebar.checkbox("Apply High Contrast (For Dark Images)", value=False)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.info("📸 Uploaded Image")
            img_pil = Image.open(uploaded_file)
            
            if use_enhancement:
                enhancer = ImageEnhance.Contrast(img_pil)
                img_pil = enhancer.enhance(2.0) 
                st.caption("✨ Enhanced Mode Active")
            
            st.image(img_pil, use_container_width=True)

        with col2:
            st.info("🧠 AI Analysis")
            if model is None or class_names is None:
                st.error("❌ Error: Model or Mapping file not found!")
            else:
                if st.button("Analyze Leaf 🚀"):
                    with st.spinner('Analyzing texture and patterns...'):
                        img = img_pil.resize((224, 224))
                        img_array = image.img_to_array(img)
                        img_array = np.expand_dims(img_array, axis=0)
                        
                        predictions = model.predict(img_array)
                        score = predictions[0]
                        
                        top_idx = np.argmax(score)
                        top_class = class_names[top_idx]
                        confidence = score[top_idx] * 100

                        if confidence > 60:
                            if "healthy" in top_class.lower():
                                st.success(f"### 🌱 Result: {top_class}")
                            else:
                                st.error(f"### 🦠 Result: {top_class}")
                            
                            st.progress(int(confidence))
                            st.caption(f"Confidence: {confidence:.2f}%")
                            
                            st.markdown("---")
                            st.markdown("#### 💊 Expert Advice")
                            st.info(get_advice(top_class))
                            
                            st.markdown("---")
                            st.write("📊 **Probability Distribution:**")
                            top_3_idx = score.argsort()[-3:][::-1]
                            top_3_values = score[top_3_idx]
                            top_3_names = [class_names[i] for i in top_3_idx]
                            
                            st.bar_chart(data=dict(zip(top_3_names, top_3_values)))

                        else:
                            st.warning(f"⚠️ Low Confidence ({confidence:.2f}%). The model is unsure.")
                            st.write(f"Best guess: {top_class}")

# Performance Metrics
elif app_mode == "📊 Performance Metrics":
    st.title("📊 Model Performance")
    st.markdown("Metrics demonstrating the robustness of our Fine-Tuned MobileNetV2.")

    tab1, tab2 = st.tabs(["Confusion Matrix", "Training Logs"])

    with tab1:
        st.subheader("Confusion Matrix")
        st.write("Visualizes classification accuracy across all classes.")
        if os.path.exists("outputs/confusion_matrix_final.png"):
            st.image("outputs/confusion_matrix_final.png", caption="Test Set Confusion Matrix", use_container_width=True)
        else:
            st.warning("Confusion matrix image not found in 'outputs/' folder.")

    with tab2:
        st.subheader("Accuracy & Loss Curves")
        if os.path.exists("outputs/accuracy_plot.png"):
            st.image("outputs/accuracy_plot.png", caption="Training vs Validation Accuracy", use_container_width=True)
        else:
            st.warning("Training plot image not found.")
            
    st.metric(label="Final Test Accuracy", value="95.23%")
    st.metric(label="Test Loss", value="0.22")


elif app_mode == "ℹ️ About Project":
    st.title("ℹ️ About EcoVision")
    st.markdown("""
    **EcoVision** is an advanced agricultural diagnostic tool designed to help farmers identify tomato and pepper diseases early.
    
    ### 🛠️ Technologies Used:
    * **Architecture:** MobileNetV2 (Transfer Learning).
    * **Training:** Fine-Tuned on 18,000+ images.
    * **Optimization:** Class Weighting & Test-Time Augmentation (TTA).
    
    ### 🎯 Goal:
    To provide a fast, offline-capable solution for food security protection.
    
    **Developed by:** [Youssef Hisham]
    **Supervised by:** [Dr.Essam Abdellatef]
    """)