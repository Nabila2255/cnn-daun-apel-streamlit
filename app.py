import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import pandas as pd
import altair as alt
import os

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(
    page_title="Deteksi Penyakit Daun Apel",
    page_icon="🍏",
    layout="wide"
)

# ============================
# CUSTOM CSS
# ============================
st.markdown("""
<style>
.title-box {
    background-color: #e1f5d8;
    padding: 18px;
    border-radius: 10px;
    font-size: 28px;
    font-weight: bold;
    color: #2e7d32;
    text-align: center;
    border: 2px solid #8bc34a;
}
.info-card {
    background-color: #f0f8ff;
    padding: 15px;
    border-left: 6px solid #2196f3;
    border-radius: 6px;
    margin-bottom: 15px;
}
.prediction-box {
    background-color: #e8f5e9;
    border-left: 8px solid #43a047;
    padding: 18px;
    border-radius: 8px;
    margin-top: 10px;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

# ============================
# JUDUL
# ============================
st.markdown(
    '<div class="title-box">🍏 Sistem Deteksi Penyakit Daun Apel - CNN (MobileNetV2) 🌿</div>',
    unsafe_allow_html=True
)
st.write("")
st.markdown("""
Aplikasi ini menggunakan **Convolutional Neural Network (CNN)**  
dengan arsitektur **MobileNetV2** untuk mengklasifikasi penyakit daun apel  
berdasarkan citra yang diunggah.
""")

# ============================
# MODEL CONFIG
# ============================
MODEL_PATH = "apple_leaf_cnn_best.h5"

CLASS_NAMES = [
    "Alternaria leaf spot",
    "Brown spot",
    "Gray spot",
    "Healthy leaf",
    "Rust"
]

DISEASE_INFO = {
    "Alternaria leaf spot": "🟤 Disebabkan oleh *Alternaria mali*.",
    "Brown spot": "🟤 Bercak coklat akibat infeksi jamur.",
    "Gray spot": "⚪ Bercak abu-abu dengan tepi gelap.",
    "Healthy leaf": "🌿 Daun sehat tanpa penyakit.",
    "Rust": "🧡 Bercak jingga akibat jamur karat."
}

# ============================
# LOAD MODEL (AMAN)
# ============================
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

if model is None:
    st.error("❌ Model tidak ditemukan! Pastikan file apple_leaf_cnn_best.h5 ada di repo.")
    st.stop()

# ============================
# UPLOAD IMAGE
# ============================
st.markdown("## 📤 Upload Gambar Daun Apel")
uploaded_file = st.file_uploader(
    "Unggah gambar (.jpg, .jpeg, .png)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", width=300)

    # Preprocessing
    img = image.resize((224, 224))
    img = np.array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    # Predict
    preds = model.predict(img)[0]

    sorted_idx = np.argsort(preds)[::-1]
    sorted_labels = [CLASS_NAMES[i] for i in sorted_idx]
    sorted_probs = preds[sorted_idx]

    predicted_label = sorted_labels[0]
    confidence = sorted_probs[0]

    # ============================
    # HASIL PREDIKSI
    # ============================
    st.markdown(
        f"""
        <div class="prediction-box">
            <h3>🎯 Prediksi: <b>{predicted_label}</b></h3>
            <h4>🔢 Tingkat Kepercayaan: <b>{confidence*100:.2f}%</b></h4>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ============================
    # INFO PENYAKIT
    # ============================
    st.markdown("## 🩺 Informasi Penyakit")
    st.markdown(
        f"<div class='info-card'>{DISEASE_INFO[predicted_label]}</div>",
        unsafe_allow_html=True
    )

    # ============================
    # GRAFIK PROBABILITAS (FIXED)
    # ============================
    st.markdown("## 📊 Grafik Probabilitas Kelas")

    df_probs = pd.DataFrame({
        "Kelas": sorted_labels,
        "Probabilitas": sorted_probs
    })

    chart = alt.Chart(df_probs).mark_bar().encode(
        x=alt.X("Probabilitas:Q", title="Nilai Probabilitas"),
        y=alt.Y("Kelas:N", sort="-x", title="Kelas Penyakit"),
        tooltip=["Kelas", "Probabilitas"]
    )

    st.altair_chart(chart, use_container_width=True)    
