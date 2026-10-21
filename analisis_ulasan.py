import os
import time
import base64
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
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# =====================================================================
# 1. PENGATURAN HALAMAN & FUNGSI PEMBACA LOGO .WEBP LOKAL GITHUB
# =====================================================================
st.set_page_config(
    page_title="JAKIVision | Analisis Sentimen Aplikasi JAKI",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def load_local_webp_logo():
    kandidat_nama = ['logo_jaki.webp', 'jaki.webp', 'logo.webp', 'logo_jaki.png']
    for nama_file in kandidat_nama:
        if os.path.exists(nama_file):
            with open(nama_file, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
                mime = "image/png" if nama_file.endswith(".png") else "image/webp"
                return f"data:{mime};base64,{encoded}"
    for file in os.listdir('.'):
        if file.lower().endswith(('.webp', '.png')):
            with open(file, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
                mime = "image/png" if file.lower().endswith(".png") else "image/webp"
                return f"data:{mime};base64,{encoded}"
    return None

logo_base64 = load_local_webp_logo()

# =====================================================================
# 2. CSS KUSTOM: FLOATING NAVBAR PERSIS MUSAVISION + DOT GRID BACKGROUND
# =====================================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Newsreader:ital,opsz,wght@1,6..72,500;1,6..72,600&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* Latar Belakang Dot-Matrix Halus persis MusaVision */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #F8FAFC;
        background-image: radial-gradient(#CBD5E1 1px, transparent 1px);
        background-size: 22px 22px;
        color: #0F172A;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 0rem !important;
    }
    .block-container {
        padding-top: 0.8rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1240px !important;
    }

    /* 1. TOP DARK STATUS BAR (Satu Baris Ramping ala MusaVision) */
    .mv-topbar {
        background-color: #070E1E;
        color: #94A3B8;
        padding: 8px 22px;
        border-radius: 10px 10px 0 0;
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: center;
        font-size: 0.72rem;
        border-bottom: 1px solid #1E293B;
    }
    .mv-topbar-left {
        display: flex;
        align-items: center;
        gap: 10px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .mv-badge-research {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: rgba(37, 99, 235, 0.2);
        color: #60A5FA;
        border: 1px solid rgba(96, 165, 250, 0.35);
        padding: 2px 10px;
        border-radius: 99px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.66rem;
        font-weight: 600;
    }
    .mv-topbar-right {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #94A3B8;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* 2. FLOATING NAVBAR CONTAINER (Mengubah st.columns menjadi 1 Baris Navbar Utuh) */
    div[data-testid="stHorizontalBlock"]:has(#mv-nav-brand) {
        background-color: rgba(255, 255, 255, 0.96);
        backdrop-filter: blur(10px);
        border: 1px solid #E2E8F0;
        border-top: none;
        border-radius: 0 0 12px 12px;
        padding: 10px 20px !important;
        margin-top: -1rem !important;
        margin-bottom: 24px !important;
        align-items: center !important;
        box-shadow: 0 8px 20px -6px rgba(15, 23, 42, 0.06);
    }

    /* Brand Kiri di dalam Navbar */
    #mv-nav-brand {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .mv-logo-box {
        height: 32px;
        width: 32px;
        border-radius: 8px;
        background-color: #0B1F47;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 0.72rem;
        letter-spacing: 0.03em;
        flex-shrink: 0;
    }
    .mv-logo-img {
        height: 32px;
        width: auto;
        border-radius: 8px;
        object-fit: contain;
    }
    .mv-brand-name {
        font-weight: 800;
        font-size: 1.05rem;
        color: #0F172A;
        letter-spacing: -0.02em;
    }
    .mv-brand-ver {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.64rem;
        font-weight: 600;
        color: #475569;
        background-color: #F1F5F9;
        border: 1px solid #E2E8F0;
        padding: 2px 7px;
        border-radius: 5px;
    }

    /* Styling Tombol di dalam Navbar agar persis MusaVision (Ghost Links vs Pill Aktif) */
    div[data-testid="stHorizontalBlock"]:has(#mv-nav-brand) div.stButton > button[kind="secondary"] {
        background-color: transparent !important;
        color: #475569 !important;
        border: 1px solid transparent !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        padding: 0.42rem 0.75rem !important;
        box-shadow: none !important;
    }
    div[data-testid="stHorizontalBlock"]:has(#mv-nav-brand) div.stButton > button[kind="secondary"]:hover {
        color: #0F172A !important;
        background-color: #F1F5F9 !important;
    }
    div[data-testid="stHorizontalBlock"]:has(#mv-nav-brand) div.stButton > button[kind="primary"] {
        background-color: #0B1736 !important;
        color: #FFFFFF !important;
        border: 1px solid #0B1736 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        padding: 0.42rem 0.9rem !important;
        box-shadow: 0 2px 6px rgba(11, 23, 54, 0.15) !important;
    }
    div[data-testid="stHorizontalBlock"]:has(#mv-nav-brand) div.stButton > button[kind="primary"]:hover {
        background-color: #1D4ED8 !important;
        border-color: #1D4ED8 !important;
    }

    /* 3. HERO LANDING SECTION */
    .mv-hero-box {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 30px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.02);
    }
    .mv-domain-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
        font-size: 0.68rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 99px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 14px;
    }
    .mv-main-title {
        font-size: 2.25rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.16;
        letter-spacing: -0.025em;
        margin-bottom: 14px;
    }
    .mv-main-title .serif-italic {
        font-family: 'Newsreader', Georgia, serif !important;
        font-style: italic;
        font-weight: 500;
        color: #1D4ED8;
    }
    .mv-hero-lead {
        font-size: 0.9rem;
        color: #475569;
        line-height: 1.65;
        margin-bottom: 20px;
    }
    .mv-inline-code {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem;
        font-weight: 600;
        background-color: #F1F5F9;
        color: #0F172A;
        padding: 2px 6px;
        border-radius: 4px;
        border: 1px solid #E2E8F0;
    }

    /* 4 METRIC BOXES DI HERO */
    .mv-metrics-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
        margin-top: 18px;
        padding-top: 18px;
        border-top: 1px solid #F1F5F9;
    }
    .mv-metric-item {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px 14px;
    }
    .mv-metric-num {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.2rem;
        font-weight: 700;
        color: #0F172A;
    }
    .mv-metric-num.blue { color: #1D4ED8; }
    .mv-metric-label {
        font-size: 0.68rem;
        color: #64748B;
        margin-top: 2px;
        font-weight: 500;
    }

    /* KARTU PREVIEW KANAN DI HERO */
    .mv-hero-preview {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.02);
    }
    .mv-preview-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 10px;
        margin-bottom: 12px;
        border-bottom: 1px solid #F1F5F9;
        font-size: 0.74rem;
        font-weight: 700;
        color: #1E293B;
    }
    .mv-preview-canvas {
        background: linear-gradient(180deg, #0B1F47 0%, #071126 100%);
        border-radius: 10px;
        padding: 18px;
        color: #F8FAFC;
        border: 1px solid #1E3A8A;
    }

    /* SECTION HEADERS */
    .mv-sec-header {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: flex-end;
        margin-bottom: 18px;
        gap: 12px;
    }
    .mv-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #1D4ED8;
        margin-bottom: 4px;
    }
    .mv-h2 {
        font-size: 1.45rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.015em;
        margin: 0;
    }
    .mv-sec-note {
        font-size: 0.82rem;
        color: #64748B;
        max-width: 440px;
        line-height: 1.5;
    }

    /* CARDS (Pipeline & Spesifikasi) */
    .mv-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px 18px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .mv-step-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        font-weight: 700;
        color: #1D4ED8;
        background-color: #EFF6FF;
        border: 1px solid #DBEAFE;
        padding: 2px 8px;
        border-radius: 4px;
        display: inline-block;
        margin-bottom: 12px;
        width: fit-content;
    }
    .mv-card-h3 {
        font-size: 0.95rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 6px;
    }
    .mv-card-p {
        font-size: 0.79rem;
        color: #475569;
        line-height: 1.55;
        margin-bottom: 14px;
    }
    .mv-card-foot {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.67rem;
        color: #64748B;
        padding-top: 10px;
        border-top: 1px solid #F1F5F9;
        display: flex;
        justify-content: space-between;
    }

    /* STUDIO CONTAINERS */
    .mv-studio-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 12px;
    }
    .mv-studio-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 10px;
        margin-bottom: 6px;
        border-bottom: 1px solid #F1F5F9;
    }
    .mv-step-num {
        background-color: #071126;
        color: #FFFFFF;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 6px;
    }
    .mv-waiting-panel {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 52px 24px;
        text-align: center;
        color: #64748B;
    }

    /* RESULT BOXES & BEDAH TOKEN */
    .mv-res-pos {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-left: 5px solid #16A34A;
        border-radius: 10px;
        padding: 16px 18px;
        color: #14532D;
        margin-bottom: 12px;
    }
    .mv-res-neg {
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
        border-left: 5px solid #DC2626;
        border-radius: 10px;
        padding: 16px 18px;
        color: #7F1D1D;
        margin-bottom: 12px;
    }
    .bedah-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 18px;
        margin-top: 12px;
    }
    .token-pill {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.73rem;
        font-weight: 600;
        background-color: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
        padding: 3px 9px;
        border-radius: 6px;
        margin: 3px 4px 3px 0;
    }
    .change-pill {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.71rem;
        background-color: #FFFBEB;
        color: #B45309;
        border: 1px solid #FDE68A;
        padding: 2px 8px;
        border-radius: 6px;
        margin: 2px 4px 2px 0;
    }

    /* GENERAL BUTTONS OUTSIDE NAVBAR */
    div.stButton > button[kind="primary"] {
        background-color: #071126 !important;
        color: #FFFFFF !important;
        border: 1px solid #071126 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        padding: 0.5rem 1rem !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #1D4ED8 !important;
        border-color: #1D4ED8 !important;
    }
    div.stButton > button[kind="secondary"] {
        background-color: #FFFFFF !important;
        color: #334155 !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        padding: 0.5rem 0.95rem !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        border-color: #1D4ED8 !important;
        color: #1D4ED8 !important;
        background-color: #EFF6FF !important;
    }

    /* KPI CARDS */
    .kpi-container {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        gap: 16px;
        margin-bottom: 22px;
    }
    .kpi-card {
        flex: 1;
        min-width: 220px;
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
    }
    .kpi-total { border-left: 5px solid #64748B; }
    .kpi-positif { border-left: 5px solid #4E79A7; }
    .kpi-negatif { border-left: 5px solid #E15759; }
    .kpi-title {
        font-size: 0.76rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin: 0 0 6px 0;
    }
    .kpi-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.95rem;
        font-weight: 800;
        margin: 0;
    }

    /* FOOTER */
    .mv-footer {
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #E2E8F0;
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: center;
        font-size: 0.76rem;
        color: #64748B;
    }
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# 3. INISIALISASI SASTRAWI, STOPWORDS, & MODEL SVM
# =====================================================================
@st.cache_resource
def load_text_processor():
    try:
        nltk.download('stopwords', quiet=True)
    except Exception:
        pass

    factory = StemmerFactory()
    stemmer_obj = factory.create_stemmer()

    try:
        stopwords_list = stopwords.words('indonesian')
    except Exception:
        stopwords_list = [
            'yang', 'di', 'dan', 'itu', 'dengan', 'untuk', 'pada', 'ke', 'dari',
            'ini', 'dalam', 'akan', 'juga', 'saya', 'ada', 'bisa', 'sudah', 'mau'
        ]

    kata_negasi_baku = ['tidak', 'bukan', 'belum', 'jangan']
    stopwords_list = [kata for kata in stopwords_list if kata not in kata_negasi_baku]

    kata_tambahan = [
        'yg', 'dg', 'rt', 'dgn', 'ny', 'd', 'klo', 'kalo', 'amp', 'biar',
        'bikin', 'bilang', 'krn', 'nya', 'nih', 'sih', 'si', 'tau', 'tuh',
        'utk', 'ya', 'jd', 'sdh', 'aja', 'n', 't', 'nyg', 'hehe', 'pen',
        'u', 'nan', 'loh', 'pas', 'sgt', 'sngt', 'y', 'bgt', 'banget', 'deh'
    ]
    stopwords_list.extend(kata_tambahan)
    return stemmer_obj, set(stopwords_list)

stemmer, list_stopwords = load_text_processor()

kamus_normalisasi = {
    'ga': 'tidak', 'gak': 'tidak', 'enggak': 'tidak', 'ngga': 'tidak',
    'nggak': 'tidak', 'gk': 'tidak', 'g': 'tidak', 'tdk': 'tidak', 'ndak': 'tidak',
    'jgn': 'jangan', 'blm': 'belum', 'bkn': 'bukan',
    'lemot': 'lambat', 'lmot': 'lambat', 'lelet': 'lambat',
    'ngelag': 'lambat', 'ngeleg': 'lambat', 'lag': 'lambat',
    'error': 'rusak', 'eror': 'rusak', 'cacat': 'rusak',
    'bug': 'rusak', 'blank': 'rusak',
    'bagus': 'bagus', 'goks': 'bagus', 'gokil': 'bagus', 'keren': 'bagus',
    'oon': 'bodoh', 'gblk': 'bodoh', 'bego': 'bodoh', 'tolol': 'bodoh',
    'jelek': 'buruk', 'cemen': 'buruk'
}

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

def bedah_perubahan_teks(text):
    """
    Membedah rincian perubahan kata demi kata (Normalisasi, Stopword, Stemming Sastrawi)
    beserta perhitungan jumlah token yang masuk ke SVM.
    """
    teks_awal = str(text).lower()
    teks_awal = re.sub(r'([a-zA-Z])\1+', r'\1', teks_awal)
    teks_awal = re.sub(r'http\S+|@[a-zA-Z0-9_]+|#\w+|\d+', '', teks_awal)
    teks_awal = re.sub(r'[^\w\s]', ' ', teks_awal)
    teks_awal = re.sub(r'\s+', ' ', teks_awal).strip()

    token_awal = teks_awal.split()
    token_masuk_svm = []
    perubahan_kata = []
    terbuang_stopwords = []

    for kata_asli in token_awal:
        kata_norm = kamus_normalisasi.get(kata_asli, kata_asli)
        if kata_norm in list_stopwords:
            terbuang_stopwords.append(kata_asli)
        else:
            kata_dasar = stemmer.stem(kata_norm)
            if kata_dasar != '':
                token_masuk_svm.append(kata_dasar)
                if kata_asli != kata_dasar:
                    perubahan_kata.append(f"{kata_asli} → {kata_dasar}")

    return {
        "jml_awal": len(token_awal),
        "jml_akhir": len(token_masuk_svm),
        "token_svm": token_masuk_svm,
        "teks_bersih": " ".join(token_masuk_svm),
        "perubahan": perubahan_kata,
        "stopwords_terbuang": terbuang_stopwords
    }

@st.cache_resource
def load_models():
    model = joblib.load('model_svm_jaki.pkl')
    vectorizer = joblib.load('tfidf_jaki.pkl')
    return model, vectorizer

try:
    svm_model, tfidf_vect = load_models()
    jml_vocab = len(tfidf_vect.vocabulary_) if hasattr(tfidf_vect, 'vocabulary_') else 4075
except Exception:
    st.error("Gagal memuat file .pkl. Pastikan model_svm_jaki.pkl dan tfidf_jaki.pkl berada di direktori yang sama.")
    st.stop()

# =====================================================================
# 4. STATE MANAGEMENT
# =====================================================================
if 'df_hasil' not in st.session_state:
    st.session_state.df_hasil = None
if 'halaman_aktif' not in st.session_state:
    st.session_state.halaman_aktif = "landing"
if 'input_sampel' not in st.session_state:
    st.session_state.input_sampel = ""

def navigasi_ke(nama_halaman):
    st.session_state.halaman_aktif = nama_halaman

# =====================================================================
# 5. TOP DARK STATUS BAR + FLOATING NAVBAR MENYATU ALA MUSAVISION
# =====================================================================
if logo_base64:
    elemen_logo = f'<img src="{logo_base64}" class="mv-logo-img" alt="Logo JAKI">'
else:
    elemen_logo = '<div class="mv-logo-box">JAKI</div>'

# Top Status Bar
st.markdown(f"""
    <div class="mv-topbar">
        <div class="mv-topbar-left">
            <span class="mv-badge-research">● Penelitian Ilmiah</span>
            <span>Arsitektur Klasifikasi Sentimen Ulasan Layanan Publik JAKI (Sastrawi + TF-IDF + Support Vector Machine)</span>
        </div>
        <div class="mv-topbar-right">
            <span>Fitur: <strong style="color:#FFFFFF;">{jml_vocab:,} Token</strong></span>
            <span>&bull;</span>
            <span>Metode: <strong style="color:#60A5FA;">Linear SVM</strong></span>
            <span>&bull;</span>
            <span>Peneliti: <strong style="color:#FFFFFF;">Artha Nugraha</strong></span>
        </div>
    </div>
""", unsafe_allow_html=True)

# Floating Navbar Menyatu (Logo Kiri | Menu Tengah | CTA Kanan) persis MusaVision
nb_brand, nb_m1, nb_m2, nb_m3, nb_m4, nb_cta = st.columns([2.2, 1.05, 1.25, 1.25, 1.25, 1.15])

with nb_brand:
    st.markdown(f"""
        <div id="mv-nav-brand">
            {elemen_logo}
            <span class="mv-brand-name">JAKIVision</span>
            <span class="mv-brand-ver">v1.0 SVM</span>
        </div>
    """, unsafe_allow_html=True)

with nb_m1:
    if st.button(
        "Ringkasan",
        type="primary" if st.session_state.halaman_aktif == "landing" else "secondary",
        use_container_width=True
    ):
        navigasi_ke("landing")
        st.rerun()

with nb_m2:
    if st.button(
        "Pemrosesan CSV",
        type="primary" if st.session_state.halaman_aktif == "massal" else "secondary",
        use_container_width=True
    ):
        navigasi_ke("massal")
        st.rerun()

with nb_m3:
    lbl_dash = "Visualisasi ●" if st.session_state.df_hasil is not None else "Visualisasi Data"
    if st.button(
        lbl_dash,
        type="primary" if st.session_state.halaman_aktif == "visualisasi" else "secondary",
        use_container_width=True
    ):
        navigasi_ke("visualisasi")
        st.rerun()

with nb_m4:
    if st.button(
        "Arsitektur & Pipeline",
        type="primary" if st.session_state.halaman_aktif == "pipeline" else "secondary",
        use_container_width=True
    ):
        navigasi_ke("pipeline")
        st.rerun()

with nb_cta:
    if st.button(
        "Uji Teks →",
        type="primary" if st.session_state.halaman_aktif == "tunggal" else "primary",
        use_container_width=True,
        key="btn_nav_cta"
    ):
        navigasi_ke("tunggal")
        st.rerun()


# =====================================================================
# 6. FUNGSI KOMPONEN HALAMAN
# =====================================================================
def render_hero_section():
    hero_kiri, hero_kanan = st.columns([7, 5], gap="medium")

    with hero_kiri:
        st.markdown(f"""
            <div class="mv-hero-box">
                <div class="mv-domain-pill">● Natural Language Processing &bull; Text Classification</div>
                <div class="mv-main-title">
                    Analisis Sentimen Ulasan<span class="serif-italic">(Aplikasi JAKI)</span>
                </div>
                <div class="mv-hero-lead">
                    Sistem klasifikasi dua kategori ulasan sentimen opini masyarakat pada aplikasi <strong>Jakarta Kini (JAKI)</strong>
                    yang diambil dari Google Play Store, menggunakan normalisasi leksikal bahasa tidak baku, ekstraksi kata dasar
                    <span class="mv-inline-code">Sastrawi</span>, pembobotan matriks frekuensi kata
                    <span class="mv-inline-code">TF-IDF</span>, dan pemisahan margin maksimum
                    <span class="mv-inline-code">Support Vector Machine</span>.
                </div>
                <div class="mv-metrics-row">
                    <div class="mv-metric-item">
                        <div class="mv-metric-num">{jml_vocab:,}</div>
                        <div class="mv-metric-label">Dimensi Kosakata</div>
                    </div>
                    <div class="mv-metric-item">
                        <div class="mv-metric-num blue">Linear</div>
                        <div class="mv-metric-label">Kernel Hyperplane</div>
                    </div>
                    <div class="mv-metric-item">
                        <div class="mv-metric-num">2 Kelas</div>
                        <div class="mv-metric-label">Positif & Negatif</div>
                    </div>
                    <div class="mv-metric-item">
                        <div class="mv-metric-num blue">~12 ms</div>
                        <div class="mv-metric-label">Waktu Inferensi</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        b1, b2 = st.columns(2)
        with b1:
            if st.button("Buka Studio Analisis Teks →", type="primary", use_container_width=True, key="cta_hero_1"):
                navigasi_ke("tunggal")
                st.rerun()
        with b2:
            if st.button("Unggah Dataset CSV JAKI →", type="secondary", use_container_width=True, key="cta_hero_2"):
                navigasi_ke("massal")
                st.rerun()

    with hero_kanan:
        st.markdown("""
            <div class="mv-hero-preview">
                <div class="mv-preview-bar">
                    <span style="color:#1D4ED8;">● Klasifikasi Margin Hyperplane SVM</span>
                    <span style="font-family:'JetBrains Mono', monospace; font-size:0.66rem; color:#64748B;">model_svm_jaki.pkl</span>
                </div>
                <div class="mv-preview-canvas">
                    <div style="font-family:'JetBrains Mono', monospace; font-size:0.66rem; color:#93C5FD; margin-bottom:5px;">
                        TEKS ULASAN MASUKAN:
                    </div>
                    <div style="font-size:0.82rem; color:#F8FAFC; line-height:1.5; background:rgba(255,255,255,0.06); padding:10px 12px; border-radius:6px; margin-bottom:12px; border:1px solid rgba(255,255,255,0.1);">
                        "Aplikasi JAKI lemot banget pas mau lapor jalan rusak di kelurahan, sering error!"
                    </div>
                    <div style="font-family:'JetBrains Mono', monospace; font-size:0.66rem; color:#93C5FD; margin-bottom:5px;">
                        HASIL NORMALISASI & STEMMING SASTRAWI (8 TOKEN):
                    </div>
                    <div style="font-family:'JetBrains Mono', monospace; font-size:0.74rem; color:#BFDBFE; background:rgba(29,78,216,0.28); padding:9px 12px; border-radius:6px; border:1px dashed #60A5FA; margin-bottom:14px;">
                        aplikasi jaki <strong>lambat</strong> lapor jalan rusak lurah sering <strong>rusak</strong>
                    </div>
                    <div style="display:flex; justify-content:space-between; align-items:center; background:rgba(220,38,38,0.22); border:1px solid #F87171; padding:10px 14px; border-radius:8px;">
                        <span style="font-size:0.74rem; font-weight:700; color:#FCA5A5;">PREDIKSI: SENTIMEN NEGATIF</span>
                        <span style="font-family:'JetBrains Mono', monospace; font-size:0.76rem; font-weight:700; color:#FFF;">Confidence: 94.6%</span>
                    </div>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.74rem; color:#64748B; padding-top:12px;">
                    <span>Kategori: <strong style="color:#0F172A;">Keluhan Teknis & Layanan</strong></span>
                    <span style="color:#1D4ED8; font-weight:700;">✓ Tervalidasi</span>
                </div>
            </div>
        """, unsafe_allow_html=True)


def render_studio_tunggal():
    st.markdown("""
        <div class="mv-sec-header">
            <div>
                <div class="mv-eyebrow">STUDIO PENGUJIAN MODEL</div>
                <h2 class="mv-h2">Analisis Teks Tunggal & Diagnosis Sentimen</h2>
            </div>
            <div class="mv-sec-note">
                Pilih salah satu sampel ulasan di bawah atau masukkan kalimat ulasan pengguna JAKI secara manual untuk menguji klasifikasi secara langsung.
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="mv-studio-card" style="padding:14px 18px; margin-bottom:12px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:0.8rem; font-weight:700; color:#0F172A;">1. Galeri Sampel Ulasan Layanan Publik JAKI</span>
                <span style="font-family:'JetBrains Mono', monospace; font-size:0.68rem; color:#64748B;">Klik sampel untuk memuat teks ke panel uji</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        if st.button("Respon JakLapor Cepat", use_container_width=True):
            st.session_state.input_sampel = "Aplikasi JAKI sangat membantu warga DKI, kemarin saya lapor lampu jalan mati di kelurahan langsung ditindak cepat oleh petugas!"
    with s2:
        if st.button("Kendala Server Lambat", use_container_width=True):
            st.session_state.input_sampel = "Aplikasi JAKI lemot banget pas mau daftar vaksin dan sering error blank putih, tolong segera diperbaiki servernya!"
    with s3:
        if st.button("Integrasi Layanan Pajak", use_container_width=True):
            st.session_state.input_sampel = "Keren banget aplikasi JAKI sekarang, cek informasi pajak PBB dan administrasi warga Jakarta jadi jauh lebih praktis dan bagus."
    with s4:
        if st.button("Keluhan Tindak Lanjut", use_container_width=True):
            st.session_state.input_sampel = "Percuma lapor parkir liar dan sampah lewat JAKI kalau respon petugasnya lelet banget, aplikasinya juga sering ngelag."
    with s5:
        if st.button("Kosongkan Masukan", use_container_width=True):
            st.session_state.input_sampel = ""

    st.write("")

    col_in, col_out = st.columns([5, 7], gap="medium")

    with col_in:
        st.markdown("""
            <div class="mv-studio-card">
                <div class="mv-studio-top">
                    <span class="mv-step-num">2. MASUKAN TEKS & PARAMETER</span>
                    <span style="font-family:'JetBrains Mono', monospace; font-size:0.68rem; color:#64748B;">Teks Bahasa Indonesia</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        user_input = st.text_area(
            "Masukkan teks ulasan pengguna aplikasi JAKI:",
            value=st.session_state.input_sampel,
            placeholder="Contoh: Aplikasi JAKI lemot banget pas mau daftar vaksin, tolong diperbaiki!",
            height=155
        )

        jalankan_prediksi = st.button("⚡ Jalankan Prediksi Sentimen", type="primary", use_container_width=True)

    with col_out:
        st.markdown("""
            <div class="mv-studio-card">
                <div class="mv-studio-top">
                    <span class="mv-step-num">3. KEPUTUSAN KLASIFIKASI SVM & BEDAH TOKEN</span>
                    <span style="font-family:'JetBrains Mono', monospace; font-size:0.68rem; color:#1D4ED8;">SVM Decision Function</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        if not jalankan_prediksi:
            st.markdown("""
                <div class="mv-waiting-panel">
                    <div style="font-weight:700; color:#0F172A; font-size:0.95rem; margin-bottom:6px;">Menunggu Masukan Teks Ulasan</div>
                    <div style="font-size:0.8rem; max-width:410px; margin:0 auto; line-height:1.55;">
                        Pilih salah satu sampel ulasan di atas atau ketik kalimat ulasan pada panel sebelah kiri, kemudian klik
                        <strong>Jalankan Prediksi Sentimen</strong> untuk memproses analisis.
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            if user_input.strip() == "":
                st.warning("Mohon masukkan teks ulasan terlebih dahulu.")
            else:
                # ANIMASI SPINNER & STATUS TAHAPAN KOMPUTASI INTERAKTIF
                with st.spinner("🔄 Mesin NLP sedang memproses normalisasi kata, stemming Sastrawi, dan margin SVM..."):
                    with st.status("⚙️ Menjalankan Pipeline Inferensi Model SVM...", expanded=True) as status:
                        st.write("1️⃣ **Normalisasi Leksikal & Case Folding:** Membakukan kata tidak baku dan membersihkan tanda baca...")
                        time.sleep(0.25)
                        detail_bedah = bedah_perubahan_teks(user_input)
                        teks_bersih = detail_bedah["teks_bersih"]

                        st.write(f"2️⃣ **Sastrawi Stemming & Stopword Removal:** Menghasilkan **{detail_bedah['jml_akhir']} token kata dasar** dari {detail_bedah['jml_awal']} kata awal...")
                        time.sleep(0.25)
                        teks_vektor = tfidf_vect.transform([teks_bersih])

                        st.write("3️⃣ **Evaluasi Hyperplane SVM:** Menghitung jarak margin dan tingkat keyakinan klasifikasi...")
                        time.sleep(0.25)
                        prediction = svm_model.predict(teks_vektor)[0]
                        jarak_margin = float(np.ravel(svm_model.decision_function(teks_vektor))[0])
                        probabilitas = 1 / (1 + np.exp(-jarak_margin))
                        keyakinan = probabilitas if probabilitas > 0.5 else (1 - probabilitas)
                        persentase_yakin = round(keyakinan * 100, 1)

                        status.update(label="✅ Analisis Selesai! Keputusan Klasifikasi SVM siap ditampilkan.", state="complete", expanded=False)

                # KARTU KEPUTUSAN KLASIFIKASI SVM
                if prediction == 'Positif':
                    st.markdown(f"""
                        <div class="mv-res-pos">
                            <div style="font-family:'JetBrains Mono', monospace; font-size:0.68rem; font-weight:700; text-transform:uppercase;">Keputusan Klasifikasi SVM</div>
                            <div style="font-size:1.4rem; font-weight:800; margin:4px 0;">Sentimen POSITIF (Tingkat Keyakinan: {persentase_yakin}%)</div>
                            <div style="font-size:0.82rem;">Ulasan terklasifikasi mengandung sentimen positif atau kepuasan terhadap layanan aplikasi JAKI.</div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="mv-res-neg">
                            <div style="font-family:'JetBrains Mono', monospace; font-size:0.68rem; font-weight:700; text-transform:uppercase;">Keputusan Klasifikasi SVM</div>
                            <div style="font-size:1.4rem; font-weight:800; margin:4px 0;">Sentimen NEGATIF (Tingkat Keyakinan: {persentase_yakin}%)</div>
                            <div style="font-size:0.82rem;">Ulasan terklasifikasi mengandung keluhan kendala teknis atau ketidakpuasan terhadap layanan.</div>
                        </div>
                    """, unsafe_allow_html=True)

                st.progress(
                    int(min(max(persentase_yakin, 1), 100)),
                    text=f"Confidence Score: {persentase_yakin}% | Jarak Margin Hyperplane: {jarak_margin:.3f}"
                )

                # SIAPKAN BADGE HTML UNTUK TOKEN & PERUBAHAN KATA
                if detail_bedah["token_svm"]:
                    pills_token_svm = "".join([f'<span class="token-pill">{t}</span>' for t in detail_bedah["token_svm"]])
                else:
                    pills_token_svm = '<span style="color:#64748B; font-size:0.78rem;">(Tidak ada token tersisa)</span>'

                if detail_bedah["perubahan"]:
                    pills_perubahan = "".join([f'<span class="change-pill">{p}</span>' for p in detail_bedah["perubahan"]])
                else:
                    pills_perubahan = '<span style="color:#64748B; font-size:0.75rem;">Seluruh kata sudah berupa kata dasar baku.</span>'

                stopwords_str = ", ".join(detail_bedah["stopwords_terbuang"]) if detail_bedah["stopwords_terbuang"] else "Tidak ada"

                # PANEL BEDAH PERUBAHAN TEKS, KATA DASAR MASUK SVM, & HITUNGAN TOKEN
                st.markdown(f"""
                    <div class="bedah-card">
                        <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #F1F5F9; padding-bottom:8px; margin-bottom:10px;">
                            <span style="font-weight:700; font-size:0.84rem; color:#0F172A;">🔍 Bedah Perubahan Teks & Hitungan Token Masuk SVM</span>
                            <span style="font-family:'JetBrains Mono', monospace; font-size:0.7rem; background:#EFF6FF; color:#1D4ED8; padding:2px 8px; border-radius:6px; border:1px solid #BFDBFE; font-weight:700;">
                                {detail_bedah['jml_awal']} Kata Awal → {detail_bedah['jml_akhir']} Token Masuk SVM
                            </span>
                        </div>
                        <div style="font-size:0.78rem; color:#475569; margin-bottom:8px;">
                            <strong>Kalimat Asli ({detail_bedah['jml_awal']} kata):</strong> "{user_input}"
                        </div>
                        <div style="font-size:0.78rem; color:#0F172A; font-weight:600; margin-bottom:4px;">
                            Hasil Pembersihan Teks & Stemming Sastrawi:
                        </div>
                        <div style="font-family:'JetBrains Mono', monospace; font-size:0.78rem; color:#1D4ED8; background:#F8FAFC; padding:8px 10px; border-radius:6px; border:1px solid #E2E8F0; margin-bottom:10px;">
                            {teks_bersih if teks_bersih else "(Teks habis terfilter stopwords)"}
                        </div>
                        <div style="font-size:0.76rem; color:#0F172A; font-weight:600; margin-bottom:4px;">
                            Daftar Kata Dasar Masuk ke Vektor SVM ({detail_bedah['jml_akhir']} Token):
                        </div>
                        <div style="margin-bottom:10px;">
                            {pills_token_svm}
                        </div>
                        <div style="font-size:0.76rem; color:#0F172A; font-weight:600; margin-bottom:4px;">
                            Transformasi Normalisasi & Stemming Sastrawi ({len(detail_bedah['perubahan'])} perubahan):
                        </div>
                        <div style="margin-bottom:8px;">
                            {pills_perubahan}
                        </div>
                        <div style="font-size:0.73rem; color:#64748B; border-top:1px solid #F1F5F9; padding-top:6px;">
                            <strong>Stopwords Tereliminasi ({len(detail_bedah['stopwords_terbuang'])} kata):</strong> <code>{stopwords_str}</code>
                        </div>
                    </div>
                """, unsafe_allow_html=True)


def render_pemrosesan_massal():
    st.markdown("""
        <div class="mv-sec-header">
            <div>
                <div class="mv-eyebrow">PEMROSESAN DATA SKALA BESAR</div>
                <h2 class="mv-h2">Klasifikasi Massal Dataset Ulasan JAKI (CSV / Excel)</h2>
            </div>
            <div class="mv-sec-note">
                Unggah file dataset hasil scraping ulasan aplikasi JAKI untuk dianalisis secara massal dan divisualisasikan pada Dashboard.
            </div>
        </div>
    """, unsafe_allow_html=True)

    c_left, c_right = st.columns([7, 5], gap="medium")

    with c_right:
        st.markdown("""
            <div class="mv-card">
                <div>
                    <div class="mv-step-tag">MEKANISME VALIDASI DATASET</div>
                    <div class="mv-card-h3">Validasi Konteks Berlapis & TF-IDF Overlap</div>
                    <div class="mv-card-p">
                        Sistem memverifikasi kesesuaian domain dataset menggunakan pembobotan kata kunci:
                        <ul style="margin-top:8px; padding-left:18px; line-height:1.6;">
                            <li>Mendeteksi kolom teks ulasan (<code>Ulasan</code>, <code>Review</code>, <code>Komentar</code>, atau <code>Text</code>).</li>
                            <li>Menghitung bobot kata kunci ekosistem JAKI (<em>jaki, jaklapor, jakwifi, pemprov dki, kelurahan</em>) dibanding kata kunci e-commerce/ride-hailing.</li>
                            <li>Memeriksa kecocokan kosakata dengan matriks <code>tfidf_jaki.pkl</code>.</li>
                        </ul>
                    </div>
                </div>
                <div class="mv-card-foot"><span>Proteksi Konteks</span><span>Keyword Heuristics</span></div>
            </div>
        """, unsafe_allow_html=True)

    with c_left:
        st.markdown("""
            <div class="mv-studio-card">
                <div class="mv-studio-top">
                    <span class="mv-step-num">UNGGAH FILE DATASET (.CSV / .XLSX)</span>
                    <span style="font-family:'JetBrains Mono', monospace; font-size:0.68rem; color:#64748B;">Batch Input</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Unggah File Dataset Ulasan JAKI (Drag and Drop)", type=['csv', 'xlsx'])

        if uploaded_file is None:
            st.session_state.df_hasil = None
        else:
            try:
                with st.spinner("Memvalidasi struktur kolom dan kesesuaian konteks dataset JAKI..."):
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)

                    kandidat_kolom = ['ulasan', 'review', 'reviews', 'komentar', 'comment', 'text', 'content']
                    kolom_terpilih = None

                    for col in df.columns:
                        if str(col).strip().lower() in kandidat_kolom:
                            kolom_terpilih = col
                            break

                if kolom_terpilih is None:
                    st.error("❌ Akses Ditolak: Kolom ulasan tidak ditemukan. Harap gunakan kolom bernama 'Ulasan', 'Review', 'Komentar', atau 'Text'.")
                    st.session_state.df_hasil = None
                else:
                    df = df.rename(columns={kolom_terpilih: 'Ulasan'})

                    teks_gabungan = " ".join(df['Ulasan'].dropna().astype(str).tolist()).lower()

                    kw_utama = ['jaki', 'jakarta kini', 'pemprov dki', 'dki jakarta', 'beritajakarta']
                    kw_fitur = ['jaklapor', 'jakpantau', 'jakwifi', 'jakwarta', 'jakrespon', 'kelurahan', 'kecamatan', 'rt/rw', 'lapor', 'crm', 'posyandu', 'pbb']

                    skor_utama = sum(teks_gabungan.count(w) for w in kw_utama) * 3
                    skor_fitur = sum(teks_gabungan.count(w) for w in kw_fitur) * 1
                    total_skor_jaki = skor_utama + skor_fitur

                    kw_domain_lain = ['driver', 'gojek', 'gofood', 'shopee', 'tokopedia', 'grab', 'pln', 'ongkir', 'pesanan', 'm-banking', 'seller']
                    skor_domain_lain = sum(teks_gabungan.count(w) for w in kw_domain_lain)

                    sampel_teks = df['Ulasan'].dropna().astype(str).head(100)
                    sampel_vektor = tfidf_vect.transform(sampel_teks)
                    jumlah_fitur_match = (sampel_vektor.sum(axis=0) > 0).sum()

                    if skor_domain_lain > total_skor_jaki and skor_domain_lain > 5:
                        st.error("❌ Akses Ditolak (Keamanan Konteks): Terdeteksi ulasan dari aplikasi e-commerce/ride-hailing, bukan konteks aplikasi JAKI.")
                        st.session_state.df_hasil = None
                    elif total_skor_jaki < 3 and jumlah_fitur_match < 5:
                        st.error("❌ Akses Ditolak (Keamanan Konteks): File yang diunggah tidak mengandung kosa kata/entitas terkait pelayanan aplikasi JAKI.")
                        st.session_state.df_hasil = None
                    else:
                        st.success(f"✅ File tervalidasi! Siap memproses data dari kolom '{kolom_terpilih}' ({df.shape[0]:,} baris).")

                        if st.button("⚡ Mulai Pembersihan & Analisis Massal", type="primary", use_container_width=True):
                            with st.spinner("Menjalankan Sastrawi Stemming dan klasifikasi SVM pada seluruh dataset..."):
                                with st.status("Memproses Data...", expanded=True) as status:
                                    df_bersih = df.dropna(subset=['Ulasan']).copy()
                                    df_bersih['Ulasan_Bersih'] = df_bersih['Ulasan'].apply(pembersihan_teks)
                                    df_bersih = df_bersih[df_bersih['Ulasan_Bersih'].str.strip() != '']

                                    vektor_massal = tfidf_vect.transform(df_bersih['Ulasan_Bersih'])
                                    df_bersih['Label_Sentimen'] = svm_model.predict(vektor_massal)

                                    st.session_state.df_hasil = df_bersih
                                    status.update(label="Pemrosesan Selesai!", state="complete", expanded=False)

                            st.success("🎉 Pemrosesan selesai! Unduh file hasil di bawah atau buka **Visualisasi Data**.")

            except Exception as e:
                st.error(f"Terjadi kesalahan teknis: {e}")
                st.session_state.df_hasil = None

    if st.session_state.df_hasil is not None:
        df_out = st.session_state.df_hasil
        with st.expander("📄 Klik untuk melihat 10 sampel data yang telah diproses", expanded=True):
            st.dataframe(df_out.head(10), use_container_width=True)

        d1, d2 = st.columns(2)
        with d1:
            csv_output = df_out.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Unduh Hasil Analisis (Hasil_JAKI.csv)",
                data=csv_output,
                file_name='Hasil_JAKI.csv',
                mime='text/csv',
                use_container_width=True
            )
        with d2:
            if st.button("📊 Buka Dashboard Visualisasi →", type="primary", use_container_width=True):
                navigasi_ke("visualisasi")
                st.rerun()


def render_pipeline_dan_spesifikasi():
    st.markdown("""
        <div class="mv-sec-header">
            <div>
                <div class="mv-eyebrow">ARSITEKTUR & METODOLOGI</div>
                <h2 class="mv-h2">Lima Tahap Pipeline Komputasi Analisis Sentimen</h2>
            </div>
            <div class="mv-sec-note">
                Alur pemrosesan sekuensial dari validasi domain teks hingga pemisahan kelas menggunakan Hyperplane Support Vector Machine.
            </div>
        </div>
    """, unsafe_allow_html=True)

    p1, p2, p3, p4, p5 = st.columns(5)
    with p1:
        st.markdown("""
            <div class="mv-card">
                <div>
                    <div class="mv-step-tag">01 / VALIDASI KONTEKS</div>
                    <div class="mv-card-h3">Input & Domain Guard</div>
                    <div class="mv-card-p">Pemeriksaan struktur atribut ulasan dan pembobotan kata kunci untuk memastikan kesesuaian topik aplikasi JAKI.</div>
                </div>
                <div class="mv-card-foot"><span>Heuristic Check</span><span>Filter Data</span></div>
            </div>
        """, unsafe_allow_html=True)
    with p2:
        st.markdown("""
            <div class="mv-card">
                <div>
                    <div class="mv-step-tag">02 / NORMALISASI TEKS</div>
                    <div class="mv-card-h3">Case Folding & Cleaning</div>
                    <div class="mv-card-p">Penyeragaman huruf kecil, pembersihan karakter non-alfabet/URL, dan konversi kata tidak baku ke kata standar.</div>
                </div>
                <div class="mv-card-foot"><span>Regex Filter</span><span>Kamus Leksikal</span></div>
            </div>
        """, unsafe_allow_html=True)
    with p3:
        st.markdown("""
            <div class="mv-card">
                <div>
                    <div class="mv-step-tag">03 / PREPROCESSING</div>
                    <div class="mv-card-h3">Stopwords & Sastrawi</div>
                    <div class="mv-card-p">Eliminasi kata umum tanpa menghapus negasi penting (<em>tidak, bukan, belum, jangan</em>) serta reduksi imbuhan ke kata dasar.</div>
                </div>
                <div class="mv-card-foot"><span>NLTK Corpus</span><span>Sastrawi Stemmer</span></div>
            </div>
        """, unsafe_allow_html=True)
    with p4:
        st.markdown("""
            <div class="mv-card">
                <div>
                    <div class="mv-step-tag">04 / EKSTRAKSI FITUR</div>
                    <div class="mv-card-h3">Pembobotan TF-IDF</div>
                    <div class="mv-card-p">Transformasi token kata dasar menjadi representasi vektor numerik berdasarkan frekuensi kata dalam dokumen.</div>
                </div>
                <div class="mv-card-foot"><span>Matriks Vektor</span><span>tfidf_jaki.pkl</span></div>
            </div>
        """, unsafe_allow_html=True)
    with p5:
        st.markdown("""
            <div class="mv-card">
                <div>
                    <div class="mv-step-tag">05 / KLASIFIKASI</div>
                    <div class="mv-card-h3">Hyperplane SVM</div>
                    <div class="mv-card-p">Penentuan kelas sentimen Positif atau Negatif berdasarkan posisi vektor terhadap bidang pemisah optimal (<em>margin</em>).</div>
                </div>
                <div class="mv-card-foot"><span>Linear Kernel</span><span>model_svm_jaki.pkl</span></div>
            </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("---")

    st.markdown("""
        <div class="mv-sec-header">
            <div>
                <div class="mv-eyebrow">SPESIFIKASI TEKNIS</div>
                <h2 class="mv-h2">Keunggulan Komputasi & Kapabilitas Sistem</h2>
            </div>
        </div>
    """, unsafe_allow_html=True)

    sp1, sp2, sp3 = st.columns(3)
    with sp1:
        st.markdown("""
            <div class="mv-card">
                <div>
                    <div class="mv-step-tag">01 / PRESERVASI NEGASI</div>
                    <div class="mv-card-h3">Akurasi Konteks Bahasa Indonesia</div>
                    <div class="mv-card-p">Mempertahankan kata negasi utama agar frasa seperti <em>"tidak lambat"</em> atau <em>"belum diperbaiki"</em> tidak kehilangan orientasi polaritas sentimennya.</div>
                </div>
                <div class="mv-card-foot"><span>Konteks Semantik</span><span style="color:#1D4ED8; font-weight:700;">Presisi Tinggi</span></div>
            </div>
        """, unsafe_allow_html=True)
    with sp2:
        st.markdown("""
            <div class="mv-card">
                <div>
                    <div class="mv-step-tag">02 / DOMAIN HEURISTICS</div>
                    <div class="mv-card-h3">Seleksi Otomatis Dataset JAKI</div>
                    <div class="mv-card-p">Mengombinasikan skor kata kunci layanan Pemprov DKI dan irisan kosakata TF-IDF untuk menolak dataset di luar topik penelitian.</div>
                </div>
                <div class="mv-card-foot"><span>Keamanan Data</span><span style="color:#1D4ED8; font-weight:700;">Multi-Layer Filter</span></div>
            </div>
        """, unsafe_allow_html=True)
    with sp3:
        st.markdown("""
            <div class="mv-card">
                <div>
                    <div class="mv-step-tag">03 / INFERENSI CEPAT</div>
                    <div class="mv-card-h3">Klasifikasi & Visualisasi Terpadu</div>
                    <div class="mv-card-p">Mendukung pengujian kalimat tunggal secara <em>real-time</em> maupun pemrosesan ribuan baris CSV lengkap dengan Peta Panas dan Word Cloud.</div>
                </div>
                <div class="mv-card-foot"><span>Analitik Visual</span><span style="color:#1D4ED8; font-weight:700;">Plotly & WordCloud</span></div>
            </div>
        """, unsafe_allow_html=True)


def render_dashboard_visualisasi():
    st.markdown("""
        <div class="mv-sec-header">
            <div>
                <div class="mv-eyebrow">DASHBOARD ANALITIK VISUAL</div>
                <h2 class="mv-h2">Visualisasi Distribusi Sentimen & Peta Panas Ulasan</h2>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if st.session_state.df_hasil is None:
        st.warning("⚠️ Silakan unggah dan proses dataset terlebih dahulu pada bagian **Pemrosesan CSV** untuk mengaktifkan Dashboard Visualisasi.")
        if st.button("Ke Pemrosesan CSV →", type="primary"):
            navigasi_ke("massal")
            st.rerun()
    else:
        with st.spinner("Memuat visualisasi grafik dan Word Cloud..."):
            df_viz = st.session_state.df_hasil
            total_ulasan = len(df_viz)
            jml_positif = len(df_viz[df_viz['Label_Sentimen'] == 'Positif'])
            jml_negatif = len(df_viz[df_viz['Label_Sentimen'] == 'Negatif'])

            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-card kpi-total">
                    <p class="kpi-title">Total Ulasan Diproses</p>
                    <p class="kpi-value" style="color:#0F172A;">{total_ulasan:,}</p>
                </div>
                <div class="kpi-card kpi-positif">
                    <p class="kpi-title" style="color:#4E79A7;">Sentimen Positif (Biru)</p>
                    <p class="kpi-value" style="color:#4E79A7;">{jml_positif:,} <span style="font-size:0.95rem; font-weight:600;">({round((jml_positif/total_ulasan)*100, 1)}%)</span></p>
                </div>
                <div class="kpi-card kpi-negatif">
                    <p class="kpi-title" style="color:#E15759;">Sentimen Negatif (Merah)</p>
                    <p class="kpi-value" style="color:#E15759;">{jml_negatif:,} <span style="font-size:0.95rem; font-weight:600;">({round((jml_negatif/total_ulasan)*100, 1)}%)</span></p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            color_map = {'Positif': '#4E79A7', 'Negatif': '#E15759'}

            col1, col2, col3 = st.columns(3)
            sentimen_counts = df_viz['Label_Sentimen'].value_counts().reset_index()
            sentimen_counts.columns = ['Sentimen', 'Jumlah']

            with col1:
                fig_pie = px.pie(
                    sentimen_counts, values='Jumlah', names='Sentimen',
                    title='Sentimen Keseluruhan',
                    color='Sentimen', color_discrete_map=color_map, hole=0.4
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(margin=dict(t=40, b=20, l=10, r=10), showlegend=True)
                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                fig_bar = px.bar(
                    sentimen_counts, x='Sentimen', y='Jumlah',
                    title='Distribusi Jumlah',
                    text='Jumlah', color='Sentimen', color_discrete_map=color_map
                )
                fig_bar.update_traces(texttemplate='%{text}', textposition='outside')
                fig_bar.update_layout(margin=dict(t=40, b=20, l=10, r=10), showlegend=True)
                st.plotly_chart(fig_bar, use_container_width=True)

            with col3:
                if 'Jumlah_Like' in df_viz.columns:
                    df_viz['Jumlah_Like'] = pd.to_numeric(df_viz['Jumlah_Like'], errors='coerce').fillna(0)
                    df_viz['Ulasan_Wrap'] = df_viz['Ulasan'].apply(lambda x: '<br>'.join(textwrap.wrap(str(x), width=50)))

                    fig_scatter = px.strip(
                        df_viz, x='Label_Sentimen', y='Jumlah_Like', color='Label_Sentimen',
                        title="Sebaran Interaksi Ulasan (Berdasarkan Like)",
                        custom_data=['Ulasan_Wrap'], color_discrete_map=color_map
                    )
                    fig_scatter.update_traces(
                        marker=dict(size=8, opacity=0.7, line=dict(width=1, color='DarkSlateGrey')),
                        hovertemplate="<b>Sentimen:</b> %{x}<br><b>Jumlah Like:</b> %{y}<br><br><b>Ulasan:</b><br>%{customdata[0]}<extra></extra>"
                    )
                    fig_scatter.update_layout(margin=dict(t=40, b=20, l=10, r=10), showlegend=True)
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.info("💡 Kolom 'Jumlah_Like' tidak ada di data.")

            st.write("---")

            col4, col5, col6 = st.columns(3)

            with col4:
                if 'Tanggal_dan_Jam' in df_viz.columns:
                    df_viz['Tanggal_dan_Jam_DT'] = pd.to_datetime(df_viz['Tanggal_dan_Jam'], errors='coerce')
                    hari_map = {0: 'Senin', 1: 'Selasa', 2: 'Rabu', 3: 'Kamis', 4: 'Jumat', 5: 'Sabtu', 6: 'Minggu'}
                    df_viz['Hari'] = df_viz['Tanggal_dan_Jam_DT'].dt.dayofweek.map(hari_map)
                    df_viz['Jam'] = df_viz['Tanggal_dan_Jam_DT'].dt.hour

                    heatmap_data = df_viz.groupby(['Hari', 'Jam']).size().reset_index(name='Jumlah')
                    pivot_data = heatmap_data.pivot(index='Hari', columns='Jam', values='Jumlah').fillna(0)
                    kategori_hari = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
                    pivot_data = pivot_data.reindex(kategori_hari[::-1])

                    for jam in range(24):
                        if jam not in pivot_data.columns:
                            pivot_data[jam] = 0
                    pivot_data = pivot_data[sorted(pivot_data.columns)]

                    fig_hm = px.imshow(
                        pivot_data, labels=dict(x="Jam Masuk", y="", color="Jumlah"),
                        x=pivot_data.columns, y=pivot_data.index,
                        color_continuous_scale="Blues", aspect="auto",
                        title="Kecepatan Ulasan Peta Panas"
                    )
                    fig_hm.update_traces(
                        hovertemplate='<b>Hari:</b> %{y}<br><b>Jam:</b> %{x}:00<br><b>Jumlah Ulasan Masuk:</b> %{z}<extra></extra>'
                    )
                    fig_hm.update_layout(margin=dict(t=30, b=20, l=10, r=10), coloraxis_showscale=False)
                    st.plotly_chart(fig_hm, use_container_width=True)
                else:
                    st.info("💡 Kolom 'Tanggal_dan_Jam' tidak ditemukan untuk membuat Heatmap.")

            def buat_wordcloud(teks_gabungan, title, colormap):
                if len(teks_gabungan) > 0:
                    wc = WordCloud(
                        width=500, height=350,
                        background_color='white',
                        colormap=colormap,
                        max_words=80
                    ).generate(teks_gabungan)
                    fig, ax = plt.subplots(figsize=(5, 3.5))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis("off")
                    ax.set_title(title, fontfamily='sans-serif', fontweight='normal', fontsize=14, pad=10, loc='left')
                    fig.tight_layout(pad=1)
                    return fig
                return None

            with col5:
                teks_pos = " ".join(review for review in df_viz[df_viz['Label_Sentimen'] == 'Positif']['Ulasan_Bersih'])
                fig_wc_pos = buat_wordcloud(teks_pos, "Kata Kunci Positif", "Blues")
                if fig_wc_pos:
                    st.pyplot(fig_wc_pos)

            with col6:
                teks_neg = " ".join(review for review in df_viz[df_viz['Label_Sentimen'] == 'Negatif']['Ulasan_Bersih'])
                fig_wc_neg = buat_wordcloud(teks_neg, "Kata Kunci Negatif", "Reds")
                if fig_wc_neg:
                    st.pyplot(fig_wc_neg)


# =====================================================================
# 7. ROUTING HALAMAN BERDASARKAN NAVBAR MENYATU DI ATAS
# =====================================================================
if st.session_state.halaman_aktif == "landing":
    render_hero_section()
    st.write("")
    st.write("---")
    render_studio_tunggal()
    st.write("")
    st.write("---")
    render_pipeline_dan_spesifikasi()

elif st.session_state.halaman_aktif == "tunggal":
    render_studio_tunggal()

elif st.session_state.halaman_aktif == "massal":
    render_pemrosesan_massal()

elif st.session_state.halaman_aktif == "visualisasi":
    render_dashboard_visualisasi()

elif st.session_state.halaman_aktif == "pipeline":
    render_pipeline_dan_spesifikasi()

# FOOTER ALA MUSAVISION
st.markdown("""
    <div class="mv-footer">
        <div>
            <strong style="color:#0F172A;">JAKIVision</strong> &nbsp;&bull;&nbsp; Analisis Sentimen Ulasan Aplikasi JAKI Berbasis Support Vector Machine (SVM)
        </div>
        <div style="font-family:'JetBrains Mono', monospace; font-size:0.7rem;">
            Sastrawi &bull; TF-IDF &bull; Linear SVM &bull; Artha Nugraha
        </div>
    </div>
""", unsafe_allow_html=True)
