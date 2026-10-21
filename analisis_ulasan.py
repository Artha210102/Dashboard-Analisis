import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import plotly.express as px

# --- PENGATURAN HALAMAN ---
st.set_page_config(page_title="Sentimen JAKI | Pemprov DKI", page_icon="🏛️", layout="wide")

# --- CUSTOM CSS LANJUTAN ---
# Menghapus karakter spasi non-standar (non-breaking space) penyebab SyntaxError
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
        background-color: #0F172A;
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
        border: 1px solid #F3F4F6;
        margin-bottom: 20px;
    }

    /* Alert Boxes */
    .success-box {padding: 15px; border-radius: 8px; background-color: #F0FDF4; border: 1px solid #BBF7D0; color: #166534; margin-top: 10px;}
    .error-box {padding: 15px; border-radius: 8px; background-color: #FEF2F2; border: 1px solid #FECACA; color: #991B1B; margin-top: 10px;}
    
    /* KPI Cards */
    .kpi-container {display: flex; justify-content: space-between; gap: 20px; margin-bottom: 25px;}
    .kpi-card {flex: 1; padding: 20px; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; background-color: white;}
    .kpi-total {border-top: 4px solid #6B7280;}
    .kpi-positif {border-top: 4px solid #3B82F6;}
    .kpi-negatif {border-top: 4px solid #EF4444;}
    .kpi-title {font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 10px;}
    .kpi-value {font-size: 2.2rem; font-weight: bold; margin: 0;}
    </style>
""", unsafe_allow_html=True)

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

# --- INISIALISASI SESSION STATE ---
if 'df_hasil' not in st.session_state:
    st.session_state.df_hasil = pd.DataFrame()

# --- FUNGSI NLP & ML ---
@st.cache_resource
def load_text_processor():
    try:
        nltk.download('stopwords', quiet=True)
    except Exception:
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
    'ga': 'tidak', 'gak': 'tidak', 'enggak': 'tidak', 'ngga': 'tidak', 'nggak': 'tidak', 'gk': 'tidak', 'g': 'tidak', 'tdk': 'tidak', 'ndak': 'tidak', 'jgn': 'jangan', 'blm': 'belum', 'bkn': 'bukan', 'lemot': 'lambat', 'lmot': 'lambat','lelet': 'lambat', 'ngelag': 'lambat', 'ngeleg': 'lambat', 'lag': 'lambat', 'error': 'rusak', 'eror': 'rusak','cacat': 'rusak', 'bug': 'rusak','blank': 'rusak', 'bagus': 'bagus', 'goks': 'bagus', 'gokil': 'bagus', 'keren': 'bagus', 'oon': 'bodoh', 'gblk': 'bodoh', 'bego': 'bodoh', 'tolol': 'bodoh', 'jelek': 'buruk', 'cKesalahan pertama (`SyntaxError: invalid decimal literal`) terjadi karena adanya karakter spasi tidak kasatmata (non-breaking space / `\xa0`) yang terbawa saat proses *copy-paste* ke dalam blok CSS. Kesalahan kedua (`KeyError: 'Prediksi_Sentimen'`) terjadi karena sistem mencoba memanggil kolom yang belum ada atau memiliki nama berbeda di dalam DataFrame hasil pemrosesan CSV.

Berikut adalah kode lengkap yang telah diperbaiki sepenuhnya. Karakter spasi bermasalah telah dibersihkan, dan logika pemrosesan CSV telah ditambahkan untuk memastikan kolom `Prediksi_Sentimen` di-generate dengan benar sebelum dihitung.

```python
import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import plotly.express as px
from wordcloud import WordCloud

# --- PENGATURAN HALAMAN ---
st.set_page_config(page_title="Sentimen JAKI | Pemprov DKI", page_icon="🏛️", layout="wide")

# --- CUSTOM CSS LANJUTAN ---
# Catatan: Semua non-breaking space telah dibersihkan untuk mencegah SyntaxError
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
        background-color: #0F172A;
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
        border: 1px solid #F3F4F6;
        margin-bottom: 20px;
    }

    /* Alert Boxes */
    .success-box {padding: 15px; border-radius: 8px; background-color: #F0FDF4; border: 1px solid #BBF7D0; color: #166534; margin-top: 10px;}
    .error-box {padding: 15px; border-radius: 8px; background-color: #FEF2F2; border: 1px solid #FECACA; color: #991B1B; margin-top: 10px;}
    
    /* KPI Cards */
    .kpi-container {display: flex; justify-content: space-between; gap: 20px; margin-bottom: 25px;}
    .kpi-card {flex: 1; padding: 20px; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; background-color: white;}
    .kpi-total {border-top: 4px solid #6B7280;}
    .kpi-positif {border-top: 4px solid #3B82F6;}
    .kpi-negatif {border-top: 4px solid #EF4444;}
    .kpi-title {font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 10px; color: #4B5563;}
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
    # Ganti dengan path model yang sesuai jika diperlukan
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

# Inisialisasi Session State
if 'df_hasil' not in st.session_state:
    st.session_state.df_hasil = None

# --- TAB NAVIGASI ---
tab1, tab2, tab3 = st.tabs(["Uji Teks", "Pemrosesan CSV", "Visualisasi Data"])

# === TAB 1: UJI TEKS ===
with tab1:
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.markdown("### Analisis Teks Tunggal & Diagnosis Sentimen")
    st.write("Masukkan kalimat ulasan pengguna JAKI secara manual untuk melihat hasil prediksi sentimen.")
    
    user_input = st.text_area("TEKS ULASAN MASUKAN:", placeholder="Contoh: Aplikasi JAKI lemot banget pas mau daftar vaksin...")
    
    if st.button("Analisis Sentimen"):
        if user_input.strip() == "":
            st.warning("Silakan masukkan teks ulasan terlebih dahulu!")
        else:
            with st.spinner("Memproses teks..."):
                teks_bersih = pembersihan_teks(user_input)
                vektor_teks = tfidf_vect.transform([teks_bersih])
                prediksi = svm_model.predict(vektor_teks)[0]
                
                st.write("**Teks Bersih (Preprocessed):**")
                st.info(teks_bersih if teks_bersih else "[Teks kosong setelah dibersihkan]")
                
                if prediksi.lower() == 'positif':
                    st.markdown(f'<div class="success-box"><strong>Hasil Prediksi: POSITIF</strong></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="error-box"><strong>Hasil Prediksi: NEGATIF</strong></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# === TAB 2: PEMROSESAN CSV ===
with tab2:
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.markdown("### Pemrosesan Dataset Massal (CSV)")
    st.write("Unggah file CSV yang berisi ulasan. Pastikan terdapat kolom bernama **'Ulasan'** atau **'Text'**.")
    
    uploaded_file = st.file_uploader("Pilih file CSV", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Pratinjau Data Awal:")
        st.dataframe(df.head())
        
        # Cari kolom teks
        kolom_target = None
        if 'Ulasan' in df.columns:
            kolom_target = 'Ulasan'
        elif 'Text' in df.columns:
            kolom_target = 'Text'
        elif 'text' in df.columns:
            kolom_target = 'text'
        else:
            kolom_target = df.columns[0] # Gunakan kolom pertama sebagai default
            st.warning(f"Kolom 'Ulasan' tidak ditemukan. Menggunakan kolom pertama: '{kolom_target}'")

        if st.button("Proses Seluruh Data"):
            with st.spinner("Sedang memproses dataset... Ini mungkin memakan waktu beberapa saat."):
                # 1. Cleaning
                df['Teks_Bersih'] = df[kolom_target].astype(str).apply(pembersihan_teks)
                # 2. Hapus baris kosong setelah cleaning
                df = df[df['Teks_Bersih'].str.strip() != '']
                # 3. Vektorisasi
                vektor_fitur = tfidf_vect.transform(df['Teks_Bersih'])
                # 4. Prediksi (Pastikan nama kolom tepat 'Prediksi_Sentimen')
                df['Prediksi_Sentimen'] = svm_model.predict(vektor_fitur)
                
                # Simpan ke session state
                st.session_state.df_hasil = df
                st.success("Pemrosesan selesai! Silakan periksa hasil di tab 'Visualisasi Data'.")
                st.dataframe(df[[kolom_target, 'Teks_Bersih', 'Prediksi_Sentimen']].head(10))
    st.markdown('</div>', unsafe_allow_html=True)


# === TAB 3: VISUALISASI DATA ===
with tab3:
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.markdown("### Dashboard Metrik & Visualisasi")
    
    df_res = st.session_state.df_hasil
    
    if df_res is not None and 'Prediksi_Sentimen' in df_res.columns:
        # Menghitung Metrik untuk mencegah KeyError
        total_data = len(df_res)
        total_pos = len(df_res[df_res['Prediksi_Sentimen'].str.lower() == 'positif'])
        total_neg = len(df_res[df_res['Prediksi_Sentimen'].str.lower() == 'negatif'])
        
        # Menampilkan KPI Cards
        st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-card kpi-total">
                    <div class="kpi-title">Total Ulasan</div>
                    <div class="kpi-value">{total_data}</div>
                </div>
                <div class="kpi-card kpi-positif">
                    <div class="kpi-title">Sentimen Positif</div>
                    <div class="kpi-value" style="color: #3B82F6;">{total_pos}</div>
                </div>
                <div class="kpi-card kpi-negatif">
                    <div class="kpi-title">Sentimen Negatif</div>
                    <div class="kpi-value" style="color: #EF4444;">{total_neg}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Visualisasi Chart
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Proporsi Sentimen")
            df_pie = df_res['Prediksi_Sentimen'].value_counts().reset_index()
            df_pie.columns = ['Sentimen', 'Jumlah']
            fig_pie = px.pie(df_pie, values='Jumlah', names='Sentimen', 
                             color='Sentimen', color_discrete_map={'positif':'#3B82F6', 'negatif':'#EF4444'})
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col2:
            st.markdown("#### Wordcloud Teks Bersih")
            teks_gabungan = " ".join(df_res['Teks_Bersih'].tolist())
            if teks_gabungan.strip():
                wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(teks_gabungan)
                st.image(wordcloud.to_array(), use_column_width=True)
            else:
                st.info("Tidak ada kata yang bisa divisualisasikan.")
    else:
        st.info("Belum ada data yang diproses. Silakan unggah dan proses CSV di tab 'Pemrosesan CSV'.")
        
    st.markdown('</div>', unsafe_allow_html=True)
