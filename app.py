"""
Streamlit web app for Brain Tumor Classification.

Run with:
    streamlit run app.py
"""

import streamlit as st
from PIL import Image

from src.predict import predict

st.set_page_config(page_title="Brain Tumor Classification", page_icon="🧠", layout="centered")

st.title("🧠 Brain Tumor Classification using Deep Learning")
st.write(
    "Upload an MRI scan and the model will classify it as **glioma**, "
    "**meningioma**, **pituitary tumor**, or **no tumor**."
)

uploaded_file = st.file_uploader("Upload MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", use_container_width=True)

    if st.button("Detect Tumor"):
        with st.spinner("Analyzing image..."):
            predicted_class, confidence, class_probs = predict(image)

        st.success(f"**Prediction:** {predicted_class.capitalize()}  "
                   f"({confidence * 100:.1f}% confidence)")

        st.subheader("Class probabilities")
        st.bar_chart(class_probs)

st.markdown("---")
st.caption(
    "Note: This tool is intended as a supportive academic/demo project and "
    "is **not** a substitute for professional medical diagnosis."
)
