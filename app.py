import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import pandas as pd
import altair as alt
import os

st.set_page_config(
    page_title="Deteksi Penyakit Daun Apel",
    page_icon="🍏",
    layout="wide"
)

st.markdown("""
<h2 style='background:#e1f5d8;padding:16px;border-radius:8px;text-align:center'>
🍏 Sistem Deteksi Penyakit Daun Apel (CNN - MobileNetV2)
</h2>
""", unsafe_allow_html=True)

MODEL_PATH = "apple_leaf_cnn_best.h5"

CLASS_NAMES = [
    "Alternaria leaf spot",
    "Brown spot",
    "Gray spot",
    "Healthy leaf",
    "Rust"
]

def build_model():
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    outputs = Dense(len(CLASS_NAMES), activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.load_weights(MODEL_PATH)
    return model

if not os.path.exists(MODEL_PATH):
    st.error("❌ File model tidak ditemukan")
    st.stop()

model = build_model()

st.subheader("📤 Upload Gambar Daun Apel")
file = st.file_uploader("Upload gambar (.jpg/.png)", type=["jpg", "png", "jpeg"])

if file:
    image = Image.open(file).convert("RGB")
    st.image(image, width=300)

    img = image.resize((224, 224))
    img = np.array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)[0]

    idx = np.argmax(preds)
    st.success(f"🎯 Prediksi: **{CLASS_NAMES[idx]}**")
    st.write(f"🔢 Probabilitas: **{preds[idx]*100:.2f}%**")

    df = pd.DataFrame({
        "Kelas": CLASS_NAMES,
        "Probabilitas": preds
    })

    chart = alt.Chart(df).mark_bar().encode(
        x="Probabilitas",
        y=alt.Y("Kelas", sort="-x")
    )

    st.altair_chart(chart, use_container_width=True)
