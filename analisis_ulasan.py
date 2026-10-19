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
    </style>
""", unsafe_allow_html=True)

# --- FUNGSI INISIALISASI SASTRAWI & NLTK (Di-cache agar tidak lambat) ---
@st.cache_resource
def load_text_processor():
    try:
        nltk.download('stopwords', quiet=True)
    except:
        pass # Handle jika sudah terdownload
        
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

# --- FUNGSI PEMBERSIHAN TEKS (Langsung dari Colab Anda) ---
def pembersihan_teks(text):
    text = str(text).lower()
    text = re.sub(r'([a-zA-Z])\1+', r'\1', text) # Regex Elongation
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

# --- TAB NAVIGASI ---
tab1, tab2 = st.tabs(["💬 Analisis Teks Tunggal", "📁 Analisis Massal & Dashboard"])

# ==========================================
# FITUR 1: ANALISIS TEKS TUNGGAL
# ==========================================
with tab1:
    st.subheader("Cek Ulasan Spesifik")
    user_input = st.text_area("Ketik keluhan atau ulasan secara manual:", placeholder="Contoh: Aplikasi JAKI lemot banget pas mau daftar vaksin, tolong diperbaiki!")
    
    if st.button("Prediksi Sentimen", type="primary"):
        if user_input.strip() == "":
            st.warning("Mohon masukkan teks ulasan terlebih dahulu!")
        else:
            with st.spinner('Menganalisis teks...'):
                # 1. Bersihkan teks
                teks_bersih = pembersihan_teks(user_input)
                
                # 2. Transformasi ke angka
                teks_vektor = tfidf_vect.transform([teks_bersih])
                
                # 3. Prediksi
                prediction = svm_model.predict(teks_vektor)[0]
                
                # 4. Hitung Tingkat Keyakinan (Pseudo-Probability dari Jarak Margin SVM)
                jarak_margin = svm_model.decision_function(teks_vektor)[0]
                probabilitas = 1 / (1 + np.exp(-jarak_margin)) # Fungsi Sigmoid
                keyakinan = probabilitas if probabilitas > 0.5 else (1 - probabilitas)
                persentase_yakin = round(keyakinan * 100, 1)
                
                # 5. Tampilkan Hasil
                st.write(f"**Teks setelah dibersihkan (Sastrawi):** *{teks_bersih}*")
                if prediction == 'Positif':
                    st.markdown(f'<div class="success-box"><h3>🟢 Sentimen POSITIF (Tingkat Keyakinan: {persentase_yakin}%)</h3></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="error-box"><h3>🔴 Sentimen NEGATIF (Tingkat Keyakinan: {persentase_yakin}%)</h3></div>', unsafe_allow_html=True)

# ==========================================
# FITUR 2 & 3: ANALISIS MASSAL + DASHBOARD
# ==========================================
with tab2:
    st.subheader("Unggah Data Ulasan Baru")
    st.info("⚠️ **Catatan Sistem:** Sistem hanya menerima file Excel/CSV hasil scraping dari Google Play Store Aplikasi JAKI yang memiliki kolom **'Ulasan'**.")
    
    uploaded_file = st.file_uploader("Unggah File (Drag and Drop)", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            # Membaca data
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
                
            # --- VALIDASI KETAT FILE JAKI ---
            kolom_wajib = ['Ulasan', 'Rating'] # Cek eksistensi kolom asli
            if not all(kolom in df.columns for kolom in kolom_wajib):
                st.error("❌ **AKSES DITOLAK:** Struktur file tidak dikenali! Harap pastikan ini adalah file hasil scraping JAKI yang memuat kolom 'Ulasan' dan 'Rating'.")
            else:
                st.success(f"✅ File tervalidasi! Menemukan {df.shape[0]} baris ulasan.")
                
                if st.button("Mulai Pembersihan & Analisis Massal", type="primary"):
                    with st.spinner('Sedang membersihkan emoji, slang, kata hubung, dan melakukan prediksi. Mohon tunggu...'):
                        
                        # 1. Hapus data kosong
                        df_bersih = df.dropna(subset=['Ulasan']).copy()
                        
                        # 2. Proses Pembersihan (Sastrawi dkk)
                        df_bersih['Ulasan_Bersih'] = df_bersih['Ulasan'].apply(pembersihan_teks)
                        
                        # 3. Buang yang setelah dibersihkan jadi kosong
                        df_bersih = df_bersih[df_bersih['Ulasan_Bersih'].str.strip() != '']
                        
                        # 4. Prediksi SVM
                        vektor_massal = tfidf_vect.transform(df_bersih['Ulasan_Bersih'])
                        df_bersih['Label_Sentimen'] = svm_model.predict(vektor_massal)
                        
                        st.success("🎉 Pemrosesan Selesai!")
                        
                        # Menampilkan Tabel 10 Teratas
                        st.write("### 📄 Sampel 10 Data Teratas (Hasil Pemrosesan)")
                        st.dataframe(df_bersih[['Ulasan', 'Ulasan_Bersih', 'Label_Sentimen']].head(10), use_container_width=True)
                        
                        # Tombol Unduh
                        csv_output = df_bersih.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Unduh Hasil ke Excel (CSV)",
                            data=csv_output,
                            file_name='Hasil_Analisis_JAKI_Terbaru.csv',
                            mime='text/csv',
                        )
                        
                        st.divider()
                        
                        # ==========================================
                        # FITUR 3: DASHBOARD VISUALISASI CEPAT
                        # ==========================================
                        st.subheader("📊 Dashboard Visualisasi Cepat")
                        
                        sentimen_counts = df_bersih['Label_Sentimen'].value_counts().reset_index()
                        sentimen_counts.columns = ['Sentimen', 'Jumlah']
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Pie Chart
                            fig_pie = px.pie(sentimen_counts, values='Jumlah', names='Sentimen', 
                                             title='Persentase Sentimen Ulasan Baru',
                                             color='Sentimen',
                                             color_discrete_map={'Positif':'#10B981', 'Negatif':'#EF4444'})
                            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                            st.plotly_chart(fig_pie, use_container_width=True)
                            
                        with col2:
                            # Bar Chart
                            fig_bar = px.bar(sentimen_counts, x='Sentimen', y='Jumlah', 
                                             title='Perbandingan Jumlah Ulasan',
                                             text='Jumlah',
                                             color='Sentimen',
                                             color_discrete_map={'Positif':'#10B981', 'Negatif':'#EF4444'})
                            fig_bar.update_traces(texttemplate='%{text}', textposition='outside')
                            st.plotly_chart(fig_bar, use_container_width=True)
                            
                        st.divider()
                        
                        # Word Cloud
                        st.write("### ☁️ Word Cloud (Kata Paling Sering Muncul)")
                        semua_kata = " ".join(review for review in df_bersih['Ulasan_Bersih'])
                        
                        if len(semua_kata) > 0:
                            wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis', max_words=100).generate(semua_kata)
                            fig_wc, ax = plt.subplots(figsize=(10, 5))
                            ax.imshow(wordcloud, interpolation='bilinear')
                            ax.axis("off")
                            st.pyplot(fig_wc)
                        else:
                            st.write("Tidak ada kata yang cukup untuk membuat Word Cloud.")
                            
        except Exception as e:
            st.error(f"Terjadi kesalahan teknis: {e}")