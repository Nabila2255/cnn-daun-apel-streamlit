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
# CUSTOM CSS (WARNA TEGAS)
# ============================
st.markdown("""
<style>
body {
    background-color: #0f172a;
    color: white;
}

.header-box {
    background: linear-gradient(90deg, #1b5e20, #2e7d32);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 5px 15px rgba(0,0,0,0.4);
}

.header-box h1 {
    color: white;
    margin-bottom: 6px;
}

.subtitle {
    color: #dcedc8;
}

.upload-box {
    background-color: #111827;
    padding: 18px;
    border-radius: 10px;
    border-left: 6px solid #66bb6a;
}

.prediction-box {
    background-color: #064e3b;
    padding: 18px;
    border-radius: 10px;
    border-left: 6px solid #34d399;
}

.info-box {
    background-color: #1e293b;
    padding: 16px;
    border-radius: 8px;
    border-left: 5px solid #60a5fa;
}
</style>
""", unsafe_allow_html=True)

# ============================
# HEADER
# ============================
st.markdown("""
<div class="header-box">
    <h1>🍏 Sistem Deteksi Penyakit Daun Apel</h1>
    <div class="subtitle">
        Convolutional Neural Network (CNN – MobileNetV2)
    </div>
</div>
""", unsafe_allow_html=True)

st.write("")

# ============================
# MODEL CONFIG
# ============================
MODEL_PATH = "apple_leaf_cnn_final.h5"

CLASS_NAMES = [
    "Alternaria leaf spot",
    "Brown spot",
    "Gray spot",
    "Healthy leaf",
    "Rust"
]

DISEASE_INFO = {
    "Alternaria leaf spot": "Bercak coklat gelap akibat jamur *Alternaria mali*.",
    "Brown spot": "Bercak coklat pada daun akibat kelembapan tinggi.",
    "Gray spot": "Bercak abu-abu dengan tepi gelap.",
    "Healthy leaf": "Daun sehat tanpa tanda penyakit.",
    "Rust": "Bercak jingga akibat jamur *Gymnosporangium*."
}

# ============================
# LOAD MODEL (AMANKAN ERROR)
# ============================
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
            safe_mode=False
        )
    except Exception as e:
        st.error("❌ Model gagal dimuat")
        st.code(str(e))
        return None

model = load_model()

# ============================
# MAIN APP
# ============================
if model:

    st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
    st.subheader("📤 Upload Gambar Daun Apel")
    uploaded_file = st.file_uploader(
        "Upload gambar (.jpg / .png)",
        type=["jpg", "jpeg", "png"]
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📷 Gambar Diunggah", width=350)

        # ============================
        # PREPROCESS
        # ============================
        img = image.resize((224, 224))
        img_array = preprocess_input(np.array(img))
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)[0]

        idx_sorted = np.argsort(preds)[::-1]
        label = CLASS_NAMES[idx_sorted[0]]
        confidence = preds[idx_sorted[0]]

        # ============================
        # PREDICTION RESULT
        # ============================
        st.markdown(f"""
        <div class="prediction-box">
            <h3>🎯 Prediksi: <b>{label}</b></h3>
            <h4>🔢 Akurasi: <b>{confidence*100:.2f}%</b></h4>
        </div>
        """, unsafe_allow_html=True)

        # ============================
        # INFO
        # ============================
        st.markdown("## 🩺 Informasi Penyakit")
        st.markdown(f"""
        <div class="info-box">
            {DISEASE_INFO[label]}
        </div>
        """, unsafe_allow_html=True)

        # ============================
        # PROBABILITY CHART
        # ============================
        df = pd.DataFrame({
            "Kelas": [CLASS_NAMES[i] for i in idx_sorted],
            "Probabilitas": preds[idx_sorted]
        })

        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X("Probabilitas:Q", scale=alt.Scale(domain=[0,1])),
            y=alt.Y("Kelas:N", sort="-x"),
            color=alt.Color("Probabilitas:Q", scale=alt.Scale(scheme="greens"))
        ).properties(height=300)

        st.markdown("## 📊 Grafik Probabilitas")
        st.altair_chart(chart, use_container_width=True)

else:
    st.stop()

