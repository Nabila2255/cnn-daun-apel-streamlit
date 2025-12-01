import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import pandas as pd
import altair as alt
import os

# ============================
#       PAGE CONFIG
# ============================
st.set_page_config(
    page_title="Deteksi Penyakit Daun Apel",
    page_icon="🍏",
    layout="wide"
)

# Custom CSS agar lebih cantik
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
#       JUDUL APLIKASI
# ============================

st.markdown('<div class="title-box">🍏 Sistem Deteksi Penyakit Daun Apel - Convolutional Neural Network (CNN) 🌿</div>', unsafe_allow_html=True)
st.write("")
st.markdown("""
Aplikasi ini menggunakan **Convolutional Neural Network (CNN) dengan arsitektur MobileNetV2** untuk melakukan klasifikasi penyakit daun apel berdasarkan citra yang Anda unggah.  
Aplikasi ini bertujuan membantu identifikasi dini penyakit tanaman secara cepat dan akurat. 🌱🔍
""")

# ============================
#       LOAD MODEL
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
    "Alternaria leaf spot": "🟤 Disebabkan oleh *Alternaria mali*. Ditandai bercak coklat gelap berbentuk bulat/tidak beraturan.",
    "Brown spot": "🟤 Bercak coklat pada daun akibat infeksi jamur. Biasanya muncul saat kelembapan tinggi.",
    "Gray spot": "⚪ Bercak abu-abu dengan tepi gelap. Sering muncul akibat patogen daun.",
    "Healthy leaf": "🌿 Daun sehat tanpa tanda penyakit. Kondisi optimal!",
    "Rust": "🧡 Bercak jingga/merah akibat jamur *Gymnosporangium*. Umumnya menyerang daun muda."
}

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        # PERBAIKAN: Menambahkan compile=False untuk menghindari error optimizer/config
        try:
            return tf.keras.models.load_model(MODEL_PATH, compile=False)
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")
            return None
    return None

model = load_model()

if model is None:
    st.error("❌ Model tidak ditemukan atau rusak! Pastikan file .h5 ada di direktori yang sama.")
else:

    # ============================
    #       UPLOAD IMAGE
    # ============================

    st.write("## 📤 Upload Gambar Daun Apel")
    uploaded_file = st.file_uploader("Unggah file gambar (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Gambar Diunggah", width=350)

            # Preprocessing
            img = image.resize((224, 224))
            img = np.array(img)
            img = preprocess_input(img)
            img = np.expand_dims(img, axis=0)

            # Prediksi
            preds = model.predict(img)[0]

            # Sorting probabilitas
            sorted_idx = np.argsort(preds)[::-1]
            sorted_labels = [CLASS_NAMES[i] for i in sorted_idx]
            sorted_probs = preds[sorted_idx]

            predicted_label = sorted_labels[0]
            confidence = sorted_probs[0]

            # ============================
            #       BOX PREDIKSI
            # ============================

            st.markdown(
                f"""
                <div class="prediction-box">
                    <h3>🎯 Prediksi: <b>{predicted_label}</b></h3>
                    <h4>🔢 Tingkat kepercayaan: <b>{confidence*100:.2f}%</b></h4>
                </div>
                """,
                unsafe_allow_html=True
            )

            # ============================
            #       INFO PENYAKIT
            # ============================

            st.markdown("## 🩺 Informasi Penyakit")
            st.markdown(f"<div class='info-card'>{DISEASE_INFO[predicted_label]}</div>", unsafe_allow_html=True)

            # ============================
            #     GRAFIK PROBABILITAS
            # ============================

            st.markdown("## 📊 Grafik Probabilitas Kelas")

            df_probs = pd.DataFrame({
                "Kelas": sorted_labels,
                "Probabilitas": sorted_probs
            })

            chart = (
                alt.Chart(df_probs)
                .mark_bar()
                .encode(
                    x=alt.X("Probabilitas:Q", title="Nilai Probabilitas"),
                    y=alt.Y("Kelas:N", sort="-x", title="Kelas Penyakit"),
                    color=alt.Color("Probabilitas", scale=alt.Scale(scheme="greens")),
                    tooltip=["Kelas", "Probabilitas"]
                )
                .properties(width=700, height=300)
            )

            st.altair_chart(chart, use_container_width=True)
            
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses gambar: {e}")

