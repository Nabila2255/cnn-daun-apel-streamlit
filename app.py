import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
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

# ============================
#       CUSTOM CSS
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
#       JUDUL
# ============================
st.markdown(
    '<div class="title-box">🍏 Sistem Deteksi Penyakit Daun Apel - CNN (MobileNetV2) 🌿</div>',
    unsafe_allow_html=True
)

st.markdown("""
Aplikasi ini menggunakan **Convolutional Neural Network (CNN)** dengan arsitektur **MobileNetV2**
untuk mengklasifikasikan penyakit daun apel berdasarkan citra yang diunggah.
""")

# ============================
#       KONFIGURASI MODEL
# ============================
MODEL_WEIGHTS = "apple_leaf_cnn_best.h5"

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
    "Healthy leaf": "🌿 Daun sehat tanpa tanda penyakit.",
    "Rust": "🧡 Bercak jingga akibat jamur *Gymnosporangium*."
}

# ============================
#       BUILD MODEL (ULANG)
# ============================
def build_model(num_classes=5):
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None
    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    output = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model

# ============================
#       LOAD WEIGHTS
# ============================
@st.cache_resource
def load_model():
    model = build_model(num_classes=len(CLASS_NAMES))
    model.load_weights(MODEL_WEIGHTS)
    return model

model = load_model()

# ============================
#       UPLOAD & PREDIKSI
# ============================
st.write("## 📤 Upload Gambar Daun Apel")
uploaded_file = st.file_uploader(
    "Unggah gambar (.jpg, .jpeg, .png)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar Daun", width=350)

    img = image.resize((224, 224))
    img = preprocess_input(np.array(img))
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)[0]
    sorted_idx = np.argsort(preds)[::-1]

    predicted_label = CLASS_NAMES[sorted_idx[0]]
    confidence = preds[sorted_idx[0]]

    st.markdown(
        f"""
        <div class="prediction-box">
            <h3>🎯 Prediksi: <b>{predicted_label}</b></h3>
            <h4>🔢 Tingkat Kepercayaan: <b>{confidence*100:.2f}%</b></h4>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("## 🩺 Informasi Penyakit")
    st.markdown(
        f"<div class='info-card'>{DISEASE_INFO[predicted_label]}</div>",
        unsafe_allow_html=True
    )

    st.markdown("## 📊 Grafik Probabilitas")
    df_probs = pd.DataFrame({
        "Kelas": CLASS_NAMES,
        "Probabilitas": preds
    })

    chart = (
        alt.Chart(df_probs)
        .mark_bar()
        .encode(
