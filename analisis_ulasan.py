import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import time

# --- PENGATURAN HALAMAN ---
st.set_page_config(page_title="Sentimen JAKI | Pemprov DKI", page_icon="🏛️", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main-header {font-size: 2.5rem; color: #1E3A8A; text-align: center; font-weight: bold;}
    .sub-header {text-align: center; color: #6B7280; margin-bottom: 30px;}
    .success-box {padding: 15px; border-radius: 10px; background-color: #D1FAE5; border-left: 5px solid #059669; margin-top: 10px;}
    .error-box {padding: 15px; border-radius: 10px; background-color: #FEE2E2; border-left: 5px solid #DC2626; margin-top: 10px;}
    /* Styling untuk KPI Cards */
    div[data-testid="stMetricValue"] {font-size: 2rem;}
    /* Styling untuk meniru layout card Tableau */
    .css-1r6slb0 {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 10px;
        border: 1px solid #e0e0e0;
    }
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
    
    kata_tambahan = ['yg', 'dg', 'rt', 'dgn', 'ny', 'd', 'klo', 'kalo', 'amp', 'biar',
                     'bikin', 'bilang', 'krn', 'nya', 'nih', 'sih', 'si', 'tau', 'tuh',
                     'utk', 'ya', 'jd', 'sdh', 'aja', 'n', 't', 'nyg', 'hehe', 'pen',
                     'u', 'nan', 'loh', 'pas', 'sgt', 'sngt', 'y', 'bgt', 'banget', 'deh']
    list_stopwords.extend(kata_tambahan)
    
    return stemmer, list_stopwords

stemmer, list_stopwords = load_text_processor()

# Kamus Normalisasi
kamus_normalisasi = {
    'ga': 'tidak', 'gak': 'tidak', 'enggak': 'tidak', 'ngga': 'tidak',
    'nggak': 'tidak', 'gk': 'tidak', 'g': 'tidak', 'tdk': 'tidak', 'ndak': 'tidak',
    'jgn': 'jangan', 'blm': 'belum', 'bkn': 'bukan',
    'lemot': 'lambat', 'lmot': 'lambat','lelet': 'lambat',
    'ngelag': 'lambat', 'ngeleg': 'lambat', 'lag': 'lambat',
    'error': 'rusak', 'eror': 'rusak','cacat': 'rusak',
    'bug': 'rusak','blank': 'rusak',
    'bagus': 'bagus', 'goks': 'bagus', 'gokil': 'bagus', 'keren': 'bagus',
    'oon': 'bodoh', 'gblk': 'bodoh', 'bego': 'bodoh', 'tolol': 'bodoh', 'jelek': 'buruk', 'cemen': 'buruk'
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
    st.error("Gagal memuat file .pkl. Pastikan model_svm_jaki.pkl dan tfidf_jaki.pkl ada di folder ini!")
    st.stop()

# --- HEADER APLIKASI ---
st.markdown('<p class="main-header">🏛️ Dashboard Analisis Sentimen Aplikasi JAKI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Berbasis Support Vector Machine (SVM) | Oleh: Artha Nugraha</p>', unsafe_allow_html=True)

# Inisialisasi Session State untuk menyimpan data agar bisa diakses antar Tab
if 'df_hasil' not in st.session_state:
    st.session_state.df_hasil = None

# --- TAB NAVIGASI ---
tab1, tab2, tab3 = st.tabs(["💬 Analisis Teks Tunggal", "📁 Pemrosesan Massal", "📊 Dashboard Visualisasi"])

# ==========================================
# TAB 1: ANALISIS TEKS TUNGGAL
# ==========================================
with tab1:
    st.subheader("Cek Ulasan Spesifik")
    user_input = st.text_area("Ketik keluhan atau ulasan secara manual:", placeholder="Contoh: Aplikasi JAKI lemot banget pas mau daftar vaksin, tolong diperbaiki!")
    
    if st.button("Prediksi Sentimen", type="primary", key="btn_prediksi_tunggal"):
        if user_input.strip() == "":
            st.warning("Mohon masukkan teks ulasan terlebih dahulu!")
        else:
            with st.spinner('Menganalisis teks...'):
                teks_bersih = pembersihan_teks(user_input)
                teks_vektor = tfidf_vect.transform([teks_bersih])
                prediction = svm_model.predict(teks_vektor)[0]
                jarak_margin = svm_model.decision_function(teks_vektor)[0]
                probabilitas = 1 / (1 + np.exp(-jarak_margin))
                keyakinan = probabilitas if probabilitas > 0.5 else (1 - probabilitas)
                persentase_yakin = round(keyakinan * 100, 1)
                
                st.write(f"**Teks setelah dibersihkan (Sastrawi):** *{teks_bersih}*")
                if prediction == 'Positif':
                    st.markdown(f'<div class="success-box"><h3>🔵 Sentimen POSITIF (Tingkat Keyakinan: {persentase_yakin}%)</h3></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="error-box"><h3>🔴 Sentimen NEGATIF (Tingkat Keyakinan: {persentase_yakin}%)</h3></div>', unsafe_allow_html=True)

# ==========================================
# TAB 2: PEMROSESAN MASSAL
# ==========================================
with tab2:
    st.subheader("Unggah Data Ulasan Baru")
    st.info("⚠️ **Catatan Sistem:** Sistem hanya menerima file CSV/Excel hasil scraping JAKI yang memuat minimal kolom **'Ulasan'**, dan idealnya beserta **'Rating'** & **'Jumlah_Like'**.")
    
    uploaded_file = st.file_uploader("Unggah File (Drag and Drop)", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
                
            if 'Ulasan' not in df.columns:
                st.error("❌ **AKSES DITOLAK:** Struktur file tidak dikenali! Harap pastikan ini adalah file yang memuat kolom 'Ulasan'.")
            else:
                st.success(f"✅ File tervalidasi! Menemukan {df.shape[0]} baris ulasan.")
                
                if st.button("Mulai Pembersihan & Analisis Massal", type="primary", key="btn_proses_massal"):
                    with st.status("Memproses Data...", expanded=True) as status:
                        st.write("Menginisialisasi engine pembersihan...")
                        
                        df_bersih = df.dropna(subset=['Ulasan']).copy()
                        
                        st.write("Membersihkan emoji, slang, kata hubung (Sastrawi)...")
                        df_bersih['Ulasan_Bersih'] = df_bersih['Ulasan'].apply(pembersihan_teks)
                        
                        df_bersih = df_bersih[df_bersih['Ulasan_Bersih'].str.strip() != '']
                        
                        st.write("Melakukan prediksi sentimen via SVM...")
                        vektor_massal = tfidf_vect.transform(df_bersih['Ulasan_Bersih'])
                        df_bersih['Label_Sentimen'] = svm_model.predict(vektor_massal)
                        
                        # Simpan ke session state
                        st.session_state.df_hasil = df_bersih
                        
                        status.update(label="Pemrosesan Selesai!", state="complete", expanded=False)
                    
                    # Tampilkan Sampel Data di Expander agar lebih rapi
                    with st.expander("📄 Klik untuk melihat sampel data yang telah diproses", expanded=True):
                        st.dataframe(df_bersih.head(10), use_container_width=True)
                    
                    csv_output = df_bersih.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Unduh Hasil Keseluruhan (CSV)",
                        data=csv_output,
                        file_name='Hasil_Analisis_JAKI_Terbaru.csv',
                        mime='text/csv',
                    )
                    
                    st.success("🎉 Pemrosesan sukses! Silakan buka **Tab 3: Dashboard Visualisasi** untuk melihat grafiknya.")

        except Exception as e:
            st.error(f"Terjadi kesalahan teknis: {e}")

# ==========================================
# TAB 3: DASHBOARD VISUALISASI
# ==========================================
with tab3:
    if st.session_state.df_hasil is None:
        st.warning("⚠️ Silakan lakukan Pemrosesan Massal terlebih dahulu di **Tab 2** untuk melihat visualisasi.")
    else:
        st.subheader("📊 Dashboard Visualisasi Analisis Sentimen")
        df_viz = st.session_state.df_hasil
        
        # --- KARTU METRIK (KPI) ---
        total_ulasan = len(df_viz)
        jml_positif = len(df_viz[df_viz['Label_Sentimen'] == 'Positif'])
        jml_negatif = len(df_viz[df_viz['Label_Sentimen'] == 'Negatif'])
        
        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
        col_kpi1.metric("Total Ulasan Diproses", f"{total_ulasan:,}")
        col_kpi2.metric("Sentimen Positif (Biru)", f"{jml_positif:,}", f"{round((jml_positif/total_ulasan)*100, 1)}%")
        col_kpi3.metric("Sentimen Negatif (Merah)", f"{jml_negatif:,}", f"{round((jml_negatif/total_ulasan)*100, 1)}%", delta_color="inverse")
        
        st.divider()
        
        # --- BARIS 1: PIE & BAR CHART ---
        sentimen_counts = df_viz['Label_Sentimen'].value_counts().reset_index()
        sentimen_counts.columns = ['Sentimen', 'Jumlah']
        
        # Mapping Warna (Positif: Biru Tableau, Negatif: Merah Tableau)
        color_map = {'Positif':'#4E79A7', 'Negatif':'#E15759'}
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pie = px.pie(sentimen_counts, values='Jumlah', names='Sentimen', 
                             title='Sentimen Keseluruhan',
                             color='Sentimen',
                             color_discrete_map=color_map,
                             hole=0.4) # Donut chart sedikit mirip tableau
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col2:
            fig_bar = px.bar(sentimen_counts, x='Sentimen', y='Jumlah', 
                             title='Distribusi Jumlah Sentimen',
                             text='Jumlah',
                             color='Sentimen',
                             color_discrete_map=color_map)
            fig_bar.update_traces(texttemplate='%{text}', textposition='outside')
            st.plotly_chart(fig_bar, use_container_width=True)
            
        st.divider()
        
        # --- BARIS 2: SCATTER PLOT IMPACT ULASAN ---
        if 'Rating' in df_viz.columns and 'Jumlah_Like' in df_viz.columns:
            st.write("### 📍 Impact Ulasan (Berdasarkan Rating & Jumlah Like)")
            # Mengganti nilai NaN atau null pada Jumlah Like menjadi 0 agar tidak error
            df_viz['Jumlah_Like'] = df_viz['Jumlah_Like'].fillna(0)
            
            fig_scatter = px.scatter(df_viz, x='Rating', y='Jumlah_Like', color='Label_Sentimen',
                                     color_discrete_map=color_map,
                                     size_max=15, # Batas ukuran bubble
                                     hover_data=['Ulasan'],
                                     title="Sebaran Interaksi Ulasan")
            
            # Membuat ukuran titik (bubble) statis agar mirip scatter plot biasa
            fig_scatter.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("💡 Grafik *Impact Ulasan* tidak dapat ditampilkan karena kolom 'Rating' atau 'Jumlah_Like' tidak ditemukan dalam file dataset yang diunggah.")
            
        st.divider()

        # --- BARIS 3: WORD CLOUD POSITIF & NEGATIF ---
        st.write("### ☁️ Kata Kunci Dominan (Word Cloud)")
        col_wc1, col_wc2 = st.columns(2)
        
        def buat_wordcloud(teks_gabungan, title, colormap):
            if len(teks_gabungan) > 0:
                wc = WordCloud(width=800, height=400, background_color='white', colormap=colormap, max_words=100).generate(teks_gabungan)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                ax.set_title(title, fontsize=20, pad=20)
                return fig
            return None

        with col_wc1:
            teks_positif = " ".join(review for review in df_viz[df_viz['Label_Sentimen'] == 'Positif']['Ulasan_Bersih'])
            fig_wc_pos = buat_wordcloud(teks_positif, "Positif Word Cloud", "Blues") # Colormap Biru
            if fig_wc_pos:
                st.pyplot(fig_wc_pos)
            else:
                st.write("Tidak cukup kata untuk Positif Word Cloud.")

        with col_wc2:
            teks_negatif = " ".join(review for review in df_viz[df_viz['Label_Sentimen'] == 'Negatif']['Ulasan_Bersih'])
            fig_wc_neg = buat_wordcloud(teks_negatif, "Negatif Word Cloud", "Reds") # Colormap Merah
            if fig_wc_neg:
                st.pyplot(fig_wc_neg)
            else:
                st.write("Tidak cukup kata untuk Negatif Word Cloud.")
