import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import pandas as pd
import altair as alt
import os

# ============================
#       KONFIGURASI HALAMAN
# ============================
st.set_page_config(
    page_title="Deteksi Penyakit Daun Apel",
    page_icon="🍏",
    layout="wide"
)

# Custom CSS agar tampilan lebih menarik
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
    .error-box {
        background-color: #ffebee;
        border-left: 8px solid #f44336;
        padding: 15px;
        border-radius: 6px;
        color: #b71c1c;
    }
</style>
""", unsafe_allow_html=True)

# ============================
#       JUDUL APLIKASI
# ============================

st.markdown('<div class="title-box">🍏 Sistem Deteksi Penyakit Daun Apel - CNN 🌿</div>', unsafe_allow_html=True)
st.write("")
st.markdown("""
Aplikasi ini menggunakan **Convolutional Neural Network (CNN)** dengan arsitektur **MobileNetV2** untuk mendeteksi penyakit pada daun apel. Unggah foto daun untuk mendapatkan hasil analisis. 🌱🔍
""")

# ============================
#       LOAD MODEL
# ============================

# Pastikan nama file ini SAMA PERSIS dengan file model yang ada di folder yang sama
MODEL_PATH = "apple_leaf_cnn_best.h5"

CLASS_NAMES = [
    "Alternaria leaf spot",
    "Brown spot",
    "Gray spot",
    "Healthy leaf",
    "Rust"
]

DISEASE_INFO = {
    "Alternaria leaf spot": "🟤 **Alternaria Leaf Spot**: Disebabkan oleh jamur *Alternaria mali*. Gejala berupa bercak coklat gelap bulat atau tidak beraturan pada daun.",
    "Brown spot": "🟤 **Brown Spot**: Bercak coklat akibat infeksi jamur, sering muncul saat kelembapan tinggi dan drainase buruk.",
    "Gray spot": "⚪ **Gray Spot**: Bercak abu-abu dengan tepian gelap. Penyakit ini dapat menghambat fotosintesis daun.",
    "Healthy leaf": "🌿 **Healthy Leaf**: Daun tampak hijau segar, bebas dari bercak atau lubang. Tanaman dalam kondisi prima!",
    "Rust": "🧡 **Rust (Karat Daun)**: Ditandai dengan bintik-bintik berwarna jingga atau merah karat, biasanya disebabkan oleh jamur *Gymnosporangium*."
}

@st.cache_resource
def load_model():
    """Memuat model dari file .h5 dengan penanganan error yang aman."""
    if not os.path.exists(MODEL_PATH):
        return None, f"File model '{MODEL_PATH}' tidak ditemukan."
    
    try:
        # compile=False SANGAT PENTING untuk menghindari error versi optimizer
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model, None
    except Exception as e:
        return None, str(e)

# Memuat model
model, error_msg = load_model()

if model is None:
    st.markdown(f"""
    <div class="error-box">
        <h3>❌ Gagal Memuat Model</h3>
        <p>Pastikan file <b>{MODEL_PATH}</b> berada di folder yang sama dengan file ini.</p>
        <p>Detail Error: <em>{error_msg}</em></p>
    </div>
    """, unsafe_allow_html=True)
else:
    # ============================
    #       UPLOAD IMAGE
    # ============================

    st.write("## 📤 Upload Gambar Daun")
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        try:
            # Tampilkan gambar
            col1, col2 = st.columns([1, 2])
            
            with col1:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Gambar yang diunggah", use_column_width=True)

            with col2:
                # Preprocessing Gambar
                img = image.resize((224, 224)) # Sesuaikan dengan input shape MobileNetV2
                img_array = np.array(img)
                img_array = preprocess_input(img_array) # Preprocessing khusus MobileNetV2
                img_array = np.expand_dims(img_array, axis=0) # Tambah batch dimension

                # Prediksi
                with st.spinner("Sedang menganalisis gambar..."):
                    preds = model.predict(img_array)[0]

                # Ambil hasil prediksi tertinggi
                sorted_idx = np.argsort(preds)[::-1]
                top_label = CLASS_NAMES[sorted_idx[0]]
                top_prob = preds[sorted_idx[0]]

                # Tampilkan Hasil Utama
                st.markdown(
                    f"""
                    <div class="prediction-box">
                        <h3>🎯 Hasil Deteksi: <b>{top_label}</b></h3>
                        <h4>📊 Kepercayaan Model: <b>{top_prob*100:.2f}%</b></h4>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Tampilkan Info Penyakit
                st.markdown(f"<div class='info-card'>{DISEASE_INFO.get(top_label, 'Info tidak tersedia')}</div>", unsafe_allow_html=True)

            # ============================
            #     GRAFIK PROBABILITAS
            # ============================
            st.write("---")
            st.write("### 📊 Detail Probabilitas Semua Kelas")

            df_probs = pd.DataFrame({
                "Kelas": CLASS_NAMES,
                "Probabilitas": preds
            })

            # Mengurutkan data untuk grafik
            df_probs = df_probs.sort_values(by="Probabilitas", ascending=False)

            chart = (
                alt.Chart(df_probs)
                .mark_bar()
                .encode(
                    x=alt.X("Probabilitas:Q", title="Tingkat Keyakinan (0-1)"),
                    y=alt.Y("Kelas:N", sort="-x", title="Jenis Penyakit"),
                    color=alt.Color("Probabilitas", scale=alt.Scale(scheme="greens"), legend=None),
                    tooltip=["Kelas", alt.Tooltip("Probabilitas", format=".2%")]
                )
                .properties(height=300)
            )

            st.altair_chart(chart, use_container_width=True)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
