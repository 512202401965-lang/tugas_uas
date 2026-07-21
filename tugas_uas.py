import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Konfigurasi halaman web
st.set_page_config(page_title="Analisis Cacat Produk", layout="wide")

st.title("🏭 Analisis Clustering Cacat Produk Industri Manufaktur")
st.markdown("Aplikasi ini dibuat untuk memenuhi tugas **Deployment & Streamlit Application**.")
st.markdown("---")

# 1. Memuat Data
st.sidebar.header("Pengaturan Data")
st.sidebar.info("Pastikan file 'defects_data.csv' berada di folder yang sama dengan app.py")

try:
    # Memuat dataset
    df = pd.read_csv('defects_data.csv')
    
    st.subheader("📋 Cuplikan Data Laporan Cacat")
    st.dataframe(df.head())

    # 2. Data Preprocessing
    # Mapping Data Ordinal
    severity_mapping = {'Minor': 1, 'Moderate': 2, 'Critical': 3}
    df['severity_score'] = df['severity'].map(severity_mapping)

    # Isolasi dan Standardisasi Fitur
    X = df[['repair_cost', 'severity_score']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. K-Means Clustering (K=3 sesuai hasil Elbow Method)
    optimal_k = 3
    kmeans_model = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
    df['cluster_kmeans'] = kmeans_model.fit_predict(X_scaled)

    # 4. Visualisasi Hasil Clustering
    st.markdown("---")
    st.subheader(f"📊 Visualisasi Hasil Segmentasi (K-Means dengan K={optimal_k})")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        x=df['repair_cost'],
        y=df['severity_score'],
        hue=df['cluster_kmeans'],
        palette='Set1',
        s=90,
        alpha=0.85,
        edgecolor='black',
        linewidth=0.5,
        ax=ax
    )
    ax.set_title('Hasil Segmentasi Kasus Cacat', fontsize=13, fontweight='bold')
    ax.set_xlabel('Biaya Perbaikan Produk ($ / Repair Cost)', fontsize=11)
    ax.set_ylabel('Tingkat Keparahan Cacat (Severity Score)', fontsize=11)
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['1 (Minor)', '2 (Moderate)', '3 (Critical)'])
    
    st.pyplot(fig)

    # 5. Interpretasi Hasil & Business Insights
    st.markdown("---")
    st.subheader("💡 Interpretasi Hasil & Rekomendasi Bisnis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**Segmen Risiko Finansial Tinggi (High-Cost Defects)**\n\nKlaster ini memiliki tingkat keparahan tinggi (*Critical*) dan pengeluaran biaya perbaikan yang masif. **Rekomendasi:** Manajemen QA wajib menempatkan sistem kendali mutu otomatis (*Automated Testing*) pada jalur ini.")
        
    with col2:
        st.success("**Segmen Minoritas Efisien (Low-Cost / Minor Defects)**\n\nKlaster dengan kerugian finansial kecil dan keparahan rendah (*Minor*). **Rekomendasi:** Penanganan dapat dilakukan melalui inspeksi berkala reguler tanpa perlu intervensi biaya operasional besar.")

except FileNotFoundError:
    st.error("Dataset 'defects_data.csv' tidak ditemukan! Silakan pastikan file tersebut ada di direktori yang sama.")
