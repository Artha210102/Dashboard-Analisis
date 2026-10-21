import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import nltk
import textwrap
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- PENGATURAN HALAMAN ---
st.set_page_config(page_title="Sentimen JAKI | Pemprov DKI", page_icon="🏛️", layout="wide")

# --- CUSTOM CSS LANJUTAN ---
st.markdown("""
    <style>
    /* Latar Belakang Dotted (Titik-titik) */
    .stApp {
        background-color: #f8f9fa;
        background-image: radial-gradient(#d1d5db 1px, transparent 1px);
        background-size: 24px 24px;
    }
    
    /* Styling Top Banner Gelap */
    .top-banner {
        background-color: #0F172A;
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        gap: 15px;
        font-family: sans-serif;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .badge-ilmiah {
        background-color: #1E3A8A;
        color: #60A5FA;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid #3B82F6;
    }
    .banner-text {
        color: #94A3B8;
        font-size: 0.9rem;
    }

    /* Styling Header Title */
    .main-logo-container {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 30px;
    }
    .logo-box {
        background-color: #0F172A;
        color: white;
        padding: 8px 12px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .title-text {
        font-size: 1.5rem;
        font-weight: 800;
        color: #111827;
        margin: 0;
    }
    .version-badge {
        background-color: #F3F4F6;
        color: #6B7280;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        border: 1px solid #E5E7EB;
    }

    /* Modifikasi Tampilan Tab Bawaan Streamlit Menjadi Button */
    div[data-baseweb="tab-list"] {
        gap: 15px;
        justify-content: center;
        margin-bottom: 30px;
    }
    div[data-baseweb="tab"] {
        background-color: white;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 10px 20px;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        color: #4B5563;
        font-weight: 600;
    }
    div[data-baseweb="tab"][aria-selected="true"] {
        background-color: #0F172A; /* Warna biru gelap / hitam */
        color: white !important;
        border-color: #0F172A;
    }
    div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {
        display: none;
    }

    /* Container Putih untuk Konten */
    .content-box {
        background-color: white;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border: 1px solid #E5E7EB;
        margin-bottom: 20px;
    }

    /* Alert Boxes */
    .success-box {padding: 20px; border-radius: 8px; background-color: #F0FDF4; border: 1px solid #BBF7D0; color: #166534; margin-top: 20px;}
    .error-box {padding: 20px; border-radius: 8px; background-color: #FEF2F2; border: 1px solid #FECACA; color: #991B1B; margin-top: 20px;}
    
    /* KPI Cards */
    .kpi-container {display: flex; justify-content: space-between; gap: 20px; margin-bottom: 25px;}
    .kpi-card {flex: 1; padding: 20px; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; background-color: white; border: 1px solid #E5E7EB;}
    .kpi-total {border-top: 4px solid #6B7280;}
    .kpi-positif {border-top: 4px solid #10B981;}
    .kpi-negatif {border-top: 4px solid #EF4444;}
    .kpi-title {font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 10px; color: #6B7280;}
    .kpi-value {font-size: 2.2rem; font-weight: bold; margin: 0; color: #111827;}
    </style>
""", unsafe_allow_html=True)

# --- FUNGSI INISIALISASI SASTRAWI & NLTK ---
@st.cache_resource
def load_text_processor():
    try:
        nltk.download('stopwords', quiet=True)
    except:
        pass
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    list_stopwords = stopwords.words('indonesian')
    kata_negasi_baku = ['tidak', 'bukan', 'belum', 'jangan']
    list_stopwords = [kata for kata in list_stopwords if kata not in kata_negasi_baku]
    kata_tambahan = ['yg', 'dg', 'rt', 'dgn', 'ny', 'd', 'klo', 'kalo', 'amp', 'biar', 'bikin', 'bilang', 'krn', 'nya', 'nih', 'sih', 'si', 'tau', 'tuh', 'utk', 'ya', 'jd', 'sdh', 'aja', 'n', 't', 'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'pas', 'sgt', 'sngt', 'y', 'bgt', 'banget', 'deh']
    list_stopwords.extend(kata_tambahan)
    return stemmer, list_stopwords

stemmer, list_stopwords = load_text_processor()

kamus_normalisasi = {
    'ga': 'tidak', 'gak': 'tidak', 'enggak': 'tidak', 'ngga': 'tidak', 'nggak': 'tidak', 'gk': 'tidak', 'g': 'tidak', 'tdk': 'tidak', 'ndak': 'tidak', 'jgn': 'jangan', 'blm': 'belum', 'bkn': 'bukan', 'lemot': 'lambat', 'lmot': 'lambat','lelet': 'lambat', 'ngelag': 'lambat', 'ngeleg': 'lambat', 'lag': 'lambat', 'error': 'rusak', 'eror': 'rusak','cacat': 'rusak', 'bug': 'rusak','blank': 'rusak', 'bagus': 'bagus', 'goks': 'bagus', 'gokil': 'bagus', 'keren': 'bagus', 'oon': 'bodoh', 'gblk': 'bodoh', 'bego': 'bodoh', 'tolol': 'bodoh', 'jelek': 'buruk', 'cemen': 'buruk'
}

# --- FUNGSI PEMBERSIHAN TEKS ---
def pembersihan_teks(text):
    text = str(text).lower()
    text = re.sub(r'([a-zA-Z])\1+', r'\1', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@[a-zA-Z0-9_]+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    kata_terpisah = text.split()
    kata_bersih = []
    for kata in kata_terpisah:
        if kata in kamus_normalisasi:
            kata = kamus_normalisasi[kata]
        if kata not in list_stopwords:
            kata_dasar = stemmer.stem(kata)
            if kata_dasar != '':
                kata_bersih.append(kata_dasar)
    return " ".join(kata_bersih)

# --- FUNGSI LOAD MODEL ---
@st.cache_resource
def load_models():
    model = joblib.load('model_svm_jaki.pkl')
    vectorizer = joblib.load('tfidf_jaki.pkl')
    return model, vectorizer

try:
    svm_model, tfidf_vect = load_models()
except Exception as e:
    st.error("Gagal memuat file .pkl. Pastikan model_svm_jaki.pkl dan tfidf_jaki.pkl ada di folder yang sama!")
    st.stop()

# --- HEADER APLIKASI ---
st.markdown("""
    <div class="top-banner">
        <span class="badge-ilmiah">• Penelitian Ilmiah</span>
        <span class="banner-text">Arsitektur Klasifikasi Sentimen Ulasan Layanan Publik JAKI (Sastrawi + TF-IDF + Support Vector Machine)</span>
    </div>
    
    <div class="main-logo-container">
        <div class="logo-box">JAKI</div>
        <h1 class="title-text">JAKIVision</h1>
        <span class="version-badge">v1.0 SVM</span>
    </div>
""", unsafe_allow_html=True)

# Inisialisasi session state untuk menyimpan dataframe hasil prediksi batch
if 'df_hasil' not in st.session_state:
    st.session_state.df_hasil = None

# --- TAB NAVIGASI ---
tab1, tab2, tab3 = st.tabs(["Teks Tunggal", "Pemrosesan CSV", "Visualisasi Data"])

# ==========================================
# TAB 1: UJI TEKS TUNGGAL DENGAN CONTENT-BOX
# ==========================================
with tab1:
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.markdown("### 📝 Analisis Teks Tunggal & Diagnosis Sentimen")
    st.write("Masukkan kalimat ulasan pengguna aplikasi JAKI secara manual di bawah ini untuk melihat hasil prediksi sentimennya.")
    
    user_input = st.text_area("Teks Ulasan Masukan:", placeholder="Contoh: Aplikasi JAKI lemot banget pas mau daftar antrean puskesmas...", height=120)
    
    col1, col2 = st.columns([1, 4])
    with col1:
        submit_btn = st.button("🚀 Analisis Sentimen", use_container_width=True)

    if submit_btn:
        if user_input.strip() == "":
            st.warning("⚠️ Silakan masukkan teks ulasan terlebih dahulu!")
        else:
            with st.spinner('Mendiagnosis teks menggunakan model SVM...'):
                teks_bersih = pembersihan_teks(user_input)
                
                if teks_bersih == "":
                    st.warning("Teks tidak memiliki makna setelah dibersihkan (hanya berisi tanda baca/stopword).")
                else:
                    vektor_teks = tfidf_vect.transform([teks_bersih])
                    prediksi = svm_model.predict(vektor_teks)[0]
                    
                    if prediksi.lower() == 'positif' or prediksi == 1:
                        st.markdown(f"""
                            <div class="success-box">
                                <h3 style='margin-top:0; color:#166534;'>✅ Sentimen: POSITIF</h3>
                                <hr style='border-color: #BBF7D0; margin:10px 0;'>
                                <strong>Teks Asli:</strong> {user_input} <br><br>
                                <strong>Hasil Preprocessing (Sastrawi):</strong> {teks_bersih}
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div class="error-box">
                                <h3 style='margin-top:0; color:#991B1B;'>❌ Sentimen: NEGATIF</h3>
                                <hr style='border-color: #FECACA; margin:10px 0;'>
                                <strong>Teks Asli:</strong> {user_input} <br><br>
                                <strong>Hasil Preprocessing (Sastrawi):</strong> {teks_bersih}
                            </div>
                        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 2: PEMROSESAN BATCH CSV
# ==========================================
with tab2:
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.markdown("### 📁 Pemrosesan Massal (Upload CSV)")
    st.write("Unggah dataset ulasan berformat CSV. Pastikan terdapat kolom bernama **'ulasan'** atau **'text'**.")
    
    uploaded_file = st.file_uploader("Upload File CSV", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Cari kolom teks
        col_text = None
        for col in ['ulasan', 'text', 'ulasan_jaki', 'content', 'review']:
            if col in df.columns:
                col_text = col
                break
                
        if col_text is None:
            st.error("❌ Tidak ditemukan kolom teks yang sesuai. Pastikan ada kolom bernama 'ulasan' atau 'text'.")
        else:
            if st.button("Mulai Proses Prediksi Massal"):
                with st.spinner('Sedang membersihkan dan memprediksi ratusan teks... Harap tunggu...'):
                    # Proses pembersihan dan prediksi
                    df['Teks_Bersih'] = df[col_text].apply(pembersihan_teks)
                    vektor_batch = tfidf_vect.transform(df['Teks_Bersih'])
                    df['Prediksi_Sentimen'] = svm_model.predict(vektor_batch)
                    
                    st.session_state.df_hasil = df
                    st.success("✅ Pemrosesan massal selesai!")
    
    # Jika sudah ada hasil, tampilkan KPI & Tabel
    if st.session_state.df_hasil is not None:
        df_res = st.session_state.df_hasil
        
        total_data = len(df_res)
        total_pos = len(df_res[df_res['Prediksi_Sentimen'].str.lower() == 'positif'])
        total_neg = len(df_res[df_res['Prediksi_Sentimen'].str.lower() == 'negatif'])
        
        # Tampilkan Custom KPI Cards
        st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-card kpi-total">
                    <p class="kpi-title">Total Ulasan</p>
                    <p class="kpi-value">{total_data}</p>
                </div>
                <div class="kpi-card kpi-positif">
                    <p class="kpi-title">Sentimen Positif</p>
                    <p class="kpi-value" style="color: #10B981;">{total_pos}</p>
                </div>
                <div class="kpi-card kpi-negatif">
                    <p class="kpi-title">Sentimen Negatif</p>
                    <p class="kpi-value" style="color: #EF4444;">{total_neg}</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.dataframe(df_res[[col_text, 'Teks_Bersih', 'Prediksi_Sentimen']].head(50), use_container_width=True)
        
        csv_download = df_res.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Unduh Hasil Prediksi (CSV)",
            data=csv_download,
            file_name="Hasil_Prediksi_JAKI.csv",
            mime="text/csv",
        )
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 3: VISUALISASI DATA
# ==========================================
with tab3:
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.markdown("### 📊 Dashboard Visualisasi")
    
    if st.session_state.df_hasil is None:
        st.info("ℹ️ Silakan lakukan pemrosesan dataset CSV pada tab 'Pemrosesan CSV' terlebih dahulu untuk melihat visualisasi.")
    else:
        df_vis = st.session_state.df_hasil
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Proporsi Sentimen")
            sentimen_count = df_vis['Prediksi_Sentimen'].value_counts().reset_index()
            sentimen_count.columns = ['Sentimen', 'Jumlah']
            fig_pie = px.pie(sentimen_count, values='Jumlah', names='Sentimen', 
                             color='Sentimen', color_discrete_map={'Positif':'#10B981', 'Negatif':'#EF4444'},
                             hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col2:
            st.markdown("#### WordCloud Sentimen Negatif")
            neg_text = " ".join(df_vis[df_vis['Prediksi_Sentimen'].str.lower() == 'negatif']['Teks_Bersih'])
            if neg_text.strip():
                wordcloud = WordCloud(width=800, height=500, background_color='white', colormap='Reds').generate(neg_text)
                fig, ax = plt.subplots(figsize=(6,4))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
            else:
                st.write("Tidak ada teks negatif untuk ditampilkan.")
                
    st.markdown('</div>', unsafe_allow_html=True)
