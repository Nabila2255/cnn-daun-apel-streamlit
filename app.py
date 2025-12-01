import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import altair as alt
import random

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

CLASS_NAMES = [
    "Alternaria leaf spot",
    "Brown spot",
    "Gray spot",
    "Healthy leaf",
    "Rust"
]

DISEASE_INFO = {
    "Alternaria leaf spot": "Bercak coklat akibat jamur Alternaria.",
    "Brown spot": "Infeksi jamur dengan bercak coklat.",
    "Gray spot": "Bercak abu-abu pada daun.",
    "Healthy leaf": "Daun sehat tanpa penyakit.",
    "Rust": "Bercak jingga akibat jamur karat."
}

st.subheader("📤 Upload Gambar Daun Apel")
file = st.file_uploader("Upload gambar daun (.jpg/.png)", type=["jpg", "png", "jpeg"])

if file:
    image = Image.open(file)
    st.image(image, width=300)

    # SIMULASI OUTPUT CNN
    probs = np.random.dirichlet(np.ones(len(CLASS_NAMES)), size=1)[0]
    idx = np.argmax(probs)

    st.success(f"🎯 Prediksi: **{CLASS_NAMES[idx]}**")
    st.write(f"🔢 Probabilitas: **{probs[idx]*100:.2f}%**")
    st.info(DISEASE_INFO[CLASS_NAMES[idx]])

    df = pd.DataFrame({
        "Kelas": CLASS_NAMES,
        "Probabilitas": probs
    })

    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("Probabilitas:Q"),
        y=alt.Y("Kelas:N", sort="-x")
    )

    st.altair_chart(chart, use_container_width=True)

st.markdown("""
📌 **Catatan Akademik**  
Model CNN MobileNetV2 telah dilatih secara offline.  
Pada versi deployment ini digunakan mode simulasi karena keterbatasan kompatibilitas TensorFlow di environment server.
""")
