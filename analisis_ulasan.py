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

# --- PENGATURAN HALAMAN ---
st.set_page_config(page_title="JAKIvision | Analisis Sentimen", page_icon="🏛️", layout="wide")

# --- CUSTOM CSS LANJUTAN (UI MODERN SEPERTI DESAIN TARGET) ---
st.markdown("""
    <style>
    /* Background Dotted pattern */
    .stApp {
        background-color: #f8fafc;
        background-image: radial-gradient(#cbd5e1 1px, transparent 0);
        background-size: 20px 20px;
    }
    
    /* Font & Text Overrides */
    h1, h2, h3, p, span, div {
        font-family: 'Inter', sans-serif;
    }
    
    /* Top Dark Banner */
    .dark-banner {
        background-color: #0B1120;
        color: #94A3B8;
        padding: 12px 20px;
        border-radius: 8px;
        font-size: 0.85rem;
        display: flex;
        align-items: center;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .dark-banner-badge {
        background-color: #1E3A8A;
        color: white;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
        margin-right: 15px;
    }
    .dark-banner strong { color: #F8FAFC; }

    /* Nav Bar Header */
    .nav-header {
        background-color: white;
        padding: 15px 25px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        border: 1px solid #E2E8F0;
        margin-bottom: 30px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    .nav-logo-box { background-color: #0F172A; color: white; padding: 5px 12px; border-radius: 6px; font-weight: bold; margin-right: 10px;}
    .nav-title { font-size: 1.2rem; font-weight: 800; color: #0F172A; margin-right: 10px; }
    .nav-version { background-color: #F1F5F9; color: #64748B; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600;}

    /* Section Titles */
    .section-subtitle { font-size: 0.75rem; font-weight: 700; color: #3B82F6; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 5px;}
    .section-title { font-size: 1.8rem; font-weight: 800; color: #0F172A; margin-bottom: 5px; margin-top: 0;}
    .section-desc { color: #64748B; font-size: 0.95rem; margin-bottom: 25px; }

    /* Panel Styling */
    .panel-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        height: 100%;
    }
    .panel-header {
        background-color: #0F172A;
        color: white;
        padding: 5px 12px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 15px;
        letter-spacing: 0.5px;
    }

    /* Result Boxes */
    .success-box {padding: 15px; border-radius: 8px; background-color: #4C1D95; color: white; margin-top: 15px; text-align: center;}
    .error-box {padding: 15px; border-radius: 8px; background-color: #7F1D1D; color: white; margin-top: 15px; text-align: center;}
    .result-text { font-size: 1.2rem; font-weight: bold; margin:0;}
    .result-sub { font-size: 0.85rem; opacity: 0.8; margin:0;}

    /* Styling Custom KPI Cards */
    .kpi-container {display: flex; gap: 20px; margin-bottom: 25px;}
    .kpi-card {flex: 1; padding: 20px; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); background: white;}
    .kpi-total { border-left: 6px solid #64748B; }
    .kpi-positif { border-left: 6px solid #3B82F6; }
    .kpi-negatif { border-left: 6px solid #EF4444; }
    .kpi-title {font-size: 0.85rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin: 0 0 10px 0;}
    .kpi-value {font-size: 2.2rem; font-weight: 800; color: #0F172A; margin: 0;}
    
    /* Hide default Streamlit elements for cleaner UI */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- FUNGSI INISIALISASI SASTRAWI & NLTK (Tetap Sama) ---
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

# Kamus Normalisasi (Tetap Sama)
kamus_normalisasi = {
    'ga': 'tidak', 'gak': 'tidak', 'enggak': 'tidak', 'ngga': 'tidak', 'nggak': 'tidak', 'gk': 'tidak', 'g': 'tidak', 'tdk': 'tidak', 'ndak': 'tidak', 'jgn': 'jangan', 'blm': 'belum', 'bkn': 'bukan', 'lemot': 'lambat', 'lmot': 'lambat','lelet': 'lambat', 'ngelag': 'lambat', 'ngeleg': 'lambat', 'lag': 'lambat', 'error': 'rusak', 'eror': 'rusak','cacat': 'rusak', 'bug': 'rusak','blank': 'rusak', 'bagus': 'bagus', 'goks': 'bagus', 'gokil': 'bagus', 'keren': 'bagus', 'oon': 'bodoh', 'gblk': 'bodoh', 'bego': 'bodoh', 'tolol': 'bodoh', 'jelek': 'buruk', 'cemen': 'buruk'
}

# --- FUNGSI PEMBERSIHAN TEKS (Tetap Sama) ---
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

# --- FUNGSI LOAD MODEL (Tetap Sama) ---
@st.cache_resource
def load_models():
    model = joblib.load('model_svm_jaki.pkl')
    vectorizer = joblib.load('tfidf_jaki.pkl')
    return model, vectorizer

try:
    svm_model, tfidf_vect = load_models()
except Exception as e:
    st.error("Gagal memuat file .pkl. Pastikan model_svm_jaki.pkl dan tfidf_jaki.pkl ada di folder ini!")
    st.stop()

# --- HEADER APLIKASI (UI BARU) ---
st.markdown("""
<div class="dark-banner">
    <div class="dark-banner-badge">Penelitian Ilmiah</div>
    <div>Arsitektur Klasifikasi Sentimen Ulasan Layanan Publik JAKI (Sastrawi + TF-IDF + Support Vector Machine) &Berikut adalah kode lengkap yang telah dirombak untuk merapikan UI Anda agar semirip mungkin dengan target desain pada gambar referensi. 

Perubahan utama yang dilakukan:
1.  **Latar Belakang & Tipografi:** Menambahkan *dot-grid background* (motif titik-titik) dan styling font yang lebih modern.
2.  **Banner Header Kustom:** Mengganti header teks biasa dengan elemen HTML/CSS khusus yang menyerupai *dark mode banner* dengan badge/tag (seperti pada gambar referensi).
3.  **Layout Tab 1 (Analisis Tunggal):** Membagi antarmuka menjadi 2 kolom (Kiri untuk Input, Kanan untuk Output) dan menambahkan *Galeri Sampel Ulasan* menggunakan mekanisme `st.session_state`.
4.  **Layout Tab 2 (Pemrosesan Massal):** Membagi antarmuka menjadi 2 kolom (Kiri untuk Upload, Kanan untuk panduan Mekanisme Validasi Dataset).
5.  **Styling Komponen:** Menambahkan container khusus dengan border, bayangan halus (*subtle shadow*), dan header berlatar belakang gelap untuk setiap panel.

Silakan salin dan ganti seluruh kode Anda dengan kode di bawah ini:

```python
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

# --- PENGATURAN HALAMAN ---
st.set_page_config(page_title="JAKIvision | Analisis Sentimen", page_icon="🏛️", layout="wide")

# --- CUSTOM CSS LANJUTAN ---
st.markdown("""
    <style>
    /* Dot Grid Background */
    .stApp {
        background-color: #ffffff;
        background-image: radial-gradient(#e5e7eb 1px, transparent 1px);
        background-size: 20px 20px;
    }
    
    /* Custom Top Banner */
    .top-banner-container {
        background-color: #0B1120;
        border-radius: 12px;
        padding: 20px 30px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .badge-blue {
        background-color: #1E3A8A;
        color: #93C5FD;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin-right: 10px;
        border: 1px solid #2563EB;
    }
    .banner-text-small { font-size: 0.85rem; color: #9CA3AF; margin-left: 10px;}
    .brand-title {
        font-size: 1.5rem; font-weight: 800; display: flex; align-items: center; margin-top: 15px;
    }
    .jaki-logo { background-color: white; color: #0B1120; padding: 2px 8px; border-radius: 6px; margin-right: 10px; font-weight: 900;}
    
    /* Section Headers */
    .section-title { font-size: 2rem; font-weight: bold; color: #111827; margin-bottom: 5px; }
    .section-subtitle { font-size: 1rem; color: #6B7280; margin-bottom: 30px; }
    
    /* Custom Panels */
    .panel-header {
        background-color: #0B1120; color: white; padding: 8px 15px; 
        border-top-left-radius: 8px; border-top-right-radius: 8px;
        font-weight: bold; font-size: 0.9rem; letter-spacing: 1px;
    }
    .panel-body {
        background-color: #ffffff; border: 1px solid #E5E7EB; border-top: none;
        border-bottom-left-radius: 8px; border-bottom-right-radius: 8px;
        padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); min-height: 250px;
    }
    
    /* Result Boxes */
    .result-box-neg { background-color: #1E1B4B; border: 1px solid #3730A3; padding: 15px; border-radius: 8px; color: white; margin-top:15px; }
    .result-box-pos { background-color: #064E3B; border: 1px solid #065F46; padding: 15px; border-radius: 8px; color: white; margin-top:15px; }
    
    /* Styling Custom KPI Cards */
    .kpi-container {display: flex; justify-content: space-between; gap: 20px; margin-bottom: 25px;}
    .kpi-card {flex: 1; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); text-align: center; background-color: white; border: 1px solid #f3f4f6;}
    .kpi-total {border-left: 6px solid #64748b;}
    .kpi-positif {border-left: 6px solid #3b82f6;}
    .kpi-negatif {border-left: 6px solid #ef4444;}
    .kpi-title {font-size: 1.1rem; font-weight: 600; margin: 0; padding-bottom: 5px; text-transform: uppercase; letter-spacing: 0.5px;}
    .kpi-value {font-size: 2.5rem; font-weight: bold; margin: 0;}
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
    'ga': 'tidak', 'gak': 'tidak', 'enggak': 'tidak', 'ngga': 'tidak', 'nggak': 'tidak', 'gk': 'tidak', 'g': 'tidak', 'tdk': 'tidak', 'ndak': 'tidak', 'jgn': 'jangan', 'blm': 'belum', 'bkn': 'bukan', 'lemot': 'lambat', 'lmot': 'lambat','lelet': 'lambat', 'ngelag': 'lambat', 'ngeleg': 'lambat', 'lag': 'lambat', 'error': 'rusak', 'eror': 'rusak','cacat': 'rusak', 'bug': 'rusak','blank': 'rusak', 'bagus': 'bagus', 'goks': 'bagus', 'gokil': 'bagus', 'keren': 'bagus', 'oon': 'bodoh', 'gblk': 'bodTentu, saya bisa membantu merapikan UI Streamlit Anda agar lebih mirip dengan gambar referensi awal yang memiliki desain lebih elegan dan terstruktur (seperti yang terlihat pada `image_5e75e1.jpg` hingga `image_5ecc9a.jpg`), dibandingkan dengan tampilan default Streamlit pada kode Anda saat ini (`image_5ed362.jpg` dkk).

Berdasarkan gambar referensi, UI yang Anda inginkan memiliki elemen-elemen berikut yang perlu ditambahkan/dimodifikasi dalam kode Anda:

1.  **Header dengan Background Gelap:** Header utama tidak hanya teks, tetapi dibungkus dalam *container* berwarna biru gelap dengan sudut melengkung.
2.  **Badge/Tag di Header:** Ada badge seperti "Penelitian Ilmiah", "Metode: Linear SVM", dll.
3.  **Navigasi Kustom (Bukan Tab Bawaan Streamlit):** Gambar referensi menggunakan semacam tombol navigasi horizontal ("Ringkasan", "Pemrosesan CSV", "Visualisasi Data", dll) alih-alih `st.tabs` bawaan yang terlihat pada kode Anda.
4.  **Layout Card yang Lebih Elegan:** Kartu untuk "Analisis Sentimen Ulasan Layanan Publik" dan "Klasifikasi Margin Hyperplane SVM" memiliki *shadow* dan *border-radius* yang jelas, serta penataan teks yang lebih rapi (menggunakan font sans-serif modern).
5.  **Kotak Input & Hasil (Terminal Style):** Pada bagian "Analisis Teks Tunggal", kotak input dan hasil prediksi digayakan seperti *terminal* atau *code block* berwarna biru tua dengan teks monospaced.
6.  **Card Metrik Bawah:** Terdapat tiga kartu metrik kecil di bawah ("4,075 Dimensi Kosakata", "Linear Kernel", dll) yang sejajar rata.

**Pendekatan untuk Memperbaiki UI di Streamlit:**

Mengingat keterbatasan kustomisasi UI bawaan Streamlit, Anda perlu sangat bergantung pada injeksi CSS dan HTML melalui `st.markdown(..., unsafe_allow_html=True)`.

Berikut adalah panduan dan potongan kode untuk merapikan UI Anda menuju desain yang diinginkan:

### 1. Perbaiki Header & Navigasi

Ganti bagian header di kode Anda dengan blok HTML/CSS yang meniru tampilan gelap. Untuk navigasi, Anda bisa menggunakan `st.radio` horizontal (atau komponen *third-party* seperti `streamlit-option-menu`, tetapi mari gunakan CSS dasar dulu agar simpel).

```python
# --- CUSTOM CSS UNTUK HEADER & CARD ---
st.markdown("""
    <style>
    /* Hilangkan padding default top */
    .block-container { padding-top: 2rem; }

    /* Header Gelap */
    .dark-header {
        background-color: #0B192C; /* Warna biru gelap */
        color: white;
        padding: 15px 25px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .badge {
        background-color: rgba(255,255,255,0.1);
        padding: 4px 10px;
        border-radius: 15px;
        font-size: 0.8rem;
        margin-right: 10px;
        border: 1px solid rgba(255,255,255,0.2);
    }
    .header-logo { font-weight: bold; font-size: 1.2rem; }
    
    /* Layout Utama */
    .content-card {
        background-color: white;
        padding: 30px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .section-title {
        color: #1f2937;
        font-weight: 800;
        font-size: 2rem;
        margin-bottom: 5px;
    }
    .section-subtitle {
        color: #6Untuk mengubah UI Anda agar sesuai dengan desain yang lebih modern dan profesional seperti pada referensi gambar awal (dengan latar belakang pola titik-titik, banner biru gelap, dan tab berbentuk *pill button*), kita perlu memaksimalkan penggunaan custom CSS pada Streamlit.

Berikut adalah pembaruan pada bagian **PENGATURAN HALAMAN**, **CUSTOM CSS**, dan **HEADER APLIKASI** yang bisa langsung Anda *copy-paste* untuk menggantikan bagian atas kode Anda. Bagian logika pemrosesan data (Python) tetap sama.

### 1. Update Bagian CSS dan Header

Ganti bagian kode dari `# --- PENGATURAN HALAMAN ---` sampai `# --- HEADER APLIKASI ---` dengan kode berikut:

```python
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
    /* Sembunyikan garis merah bawaan tab Streamlit */
    div[data-baseweb="tab-highlight"] {
        display: none;
    }
    div[data-baseweb="tab-border"] {
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

if 'df_hasil' not in st.session_state:
    st.session_state.df_hasil = None

# --- TAB NAVIGASI ---
tab1, tab2, tab3 = st.tabs(["Uji Teks", "Pemrosesan CSV", "Visualisasi Data"])
