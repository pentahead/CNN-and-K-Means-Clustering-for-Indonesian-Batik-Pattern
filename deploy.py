import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image
from collections import Counter
from sklearn.cluster import KMeans
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt

# --- Config UI
st.set_page_config(
    page_title="Batik Pattern Analyzer",
    page_icon="icon.png"
)
st.markdown("""
    <style>
    .header-title {
        font-size: 2.2rem;
        font-weight: bold;
        color: #d97706;
    }
    .subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-top: -8px;
        margin-bottom: 10px;
    }
    .badge {
        display:inline-block; padding:0.3em 0.6em;
        border-radius:0.4em; font-size:0.8rem;
        margin-right:0.5rem; margin-top:0.2rem;
    }
    .badge.green { background:#d1fae5; color:#059669; }
    .badge.blue { background:#dbeafe; color:#2563eb; }
    .badge.yellow { background:#fef3c7; color:#d97706; }
    </style>
""", unsafe_allow_html=True)

# --- Header
st.markdown('<div class="header-title">🔍 AI Batik Pattern Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">CNN Feature Extraction + K-Means Clustering for Traditional Batik Pattern Recognition</div>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;'>
    <span class="badge green">✅ CNN Model Ready</span>
    <span class="badge blue">📊 Clustering Active</span>
    <span class="badge yellow">🎨 Color Analyzer Enabled</span>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# === Load Model & Data  ===
@st.cache_resource
def load_models_and_data():
    feature_extractor = load_model('batik_feature_extractor.keras')
    scaler = joblib.load('batik_scaler.pkl')
    pca = joblib.load('batik_pca.pkl')
    kmeans = joblib.load('batik_kmeans.pkl')
    df = pd.read_csv('batik_dataset_final.csv')
    return feature_extractor, scaler, pca, kmeans, df

feature_extractor, scaler, pca, kmeans, df = load_models_and_data()

cluster_mapping = {
    0: "Geometris Tajam Berulang: Polanya cenderung geometris dengan bentuk tegas, berulang secara simetris.",
    1: "Padat & Ornamental: Motif batik padat dengan ornamen rapat, detail penuh.",
    2: "Objek Dominan Tunggal: Didominasi satu objek utama seperti hewan/figur, latar minimal.",
    3: "Ringan dengan Objek Kecil Terbuka: Pola ringan, ornamen kecil tersebar dengan banyak area kosong.",
}

# === Fungsi Ekstraksi ===
def extract_feature(img):
    # Pastikan ukuran dan RGB
    img = img.convert("RGB").resize((224, 224))

    # Konversi ke array dan ubah shape
    x = keras_image.img_to_array(img)  # (224, 224, 3)
    x = np.expand_dims(x, axis=0)      # (1, 224, 224, 3)

    # Preprocessing sesuai model MobileNetV2
    x = preprocess_input(x)

    # Prediksi fitur dari CNN
    features = feature_extractor.predict(x, verbose=0)
    return features.flatten().astype(np.float64)


def extract_top_colors(img, k=5, top_n=3, resize_dim=(100, 100)):
    img = img.convert('RGB').resize(resize_dim)
    arr = np.array(img).reshape(-1, 3)
    kmeans_color = KMeans(n_clusters=k, random_state=42).fit(arr)
    counts = Counter(kmeans_color.labels_)
    most_common = counts.most_common(top_n)
    colors = [tuple(kmeans_color.cluster_centers_[label].astype(int)) for label, _ in most_common]
    while len(colors) < top_n:
        colors.append((0, 0, 0))
    return colors

def plot_top_colors(color_list):
    fig, ax = plt.subplots(1, len(color_list), figsize=(len(color_list)*2, 2))
    if len(color_list) == 1:
        ax = [ax]
    for a, (r, g, b) in zip(ax, color_list):
        a.imshow([[(r/255, g/255, b/255)]])
        a.axis('off')
    st.pyplot(fig)

# === Streamlit UI ===
left, center, right = st.columns([2, 4, 2])
with center:
    st.markdown("## Upload Gambar Batik", unsafe_allow_html=True)
    st.write("Upload gambar batik, dan sistem akan menampilkan klaster serta batik miripnya.")
    uploaded_file = st.file_uploader("📤 Upload gambar batik", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="🖼️ Gambar input", use_container_width=False, width=350)

        with st.spinner("⏳ Memproses..."):
            feature = extract_feature(img)
            feature_scaled = scaler.transform([feature]).astype(np.float64)
            feature_pca = pca.transform(feature_scaled).astype(np.float64)
            kmeans_dtype = kmeans.cluster_centers_.dtype
            feature_pca = feature_pca.astype(kmeans_dtype)
            cluster = kmeans.predict(feature_pca)[0]
            cluster_label = cluster_mapping[cluster]
            similar_batiks = df[df['predicted_cluster'] == cluster].sample(n=3, random_state=42)
            dominant_colors = extract_top_colors(img)
            hex_colors = ['#{:02x}{:02x}{:02x}'.format(*c) for c in dominant_colors]

        st.success(f"✅ Gambar diprediksi termasuk klaster: **{cluster_label}**")

        st.markdown("### 🎨 Palet warna dominan:")
        color_cols = st.columns(len(hex_colors))
        for i, col in enumerate(color_cols):
            col.markdown(f"<div style='width:100%;height:50px;background-color:{hex_colors[i]};margin:auto;'></div>", unsafe_allow_html=True)
            col.markdown(f"<center>{hex_colors[i]}</center>", unsafe_allow_html=True)

        st.markdown("### 📚 Contoh batik dari klaster pola serupa:")
        for _, row in similar_batiks.iterrows():
            st.markdown(f"**{row['nama_batik']}** - _{row['provinsi']}_")
            img_path = row['filepath']
            if img_path.startswith('/content/'):
                img_path = img_path[1:]
            if not os.path.exists(img_path):
                img_path = os.path.join('Batik_Analizer', img_path.lstrip('/'))
            if os.path.exists(img_path):
                simg = Image.open(img_path)
                st.image(simg, width=200)
            else:
                st.warning(f"File tidak ditemukan: {img_path}")