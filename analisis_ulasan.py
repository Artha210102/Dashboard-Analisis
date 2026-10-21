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
from sklearn.metrics import confusion_matrix, classification_report

# --- PENGATURAN HALAMAN ---
st.set_page_config(page_title="Sentimen JAKI | Pemprov DKI", page_icon="🏛️", layout="wide")

# --- CUSTOM CSS LANJUTAN ---
st.markdown("""
    <style>
    .main-header {font-size: 2.5rem; color: #1E3A8A; text-align: center; font-weight: bold;}
    .sub-header {text-align: center; color: #6B7280; margin-bottom: 30px;}
    .success-box {padding: 15px; border-radius: 10px; background-color: #D1FAE5; border-left: 5px solid #059669; margin-top: 10px;}
    .error-box {padding: 15px; border-radius: 10px; background-color: #FEE2E2; border-left: 5px solid #DC2626; margin-top: 10px;}
    
    /* Styling Custom KPI Cards */
    .kpi-container {display: flex; justify-content: space-between; gap: 20px; margin-bottom: 25px;}
    .kpi-card {flex: 1; padding: 20px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.05); text-align: center;}
    
    /* MENAMBAHKAN GARIS ABU TERANG DI KIRI CARD TOTAL */
    .kpi-total {background-color: #f8f9fa; border: 1px solid #e0e0e0; border-left: 6px solid #adb5bd;}
    
    .kpi-positif {background-color: #F0F4FA; border-left: 6px solid #4E79A7;}
    .kpi-negatif {background-color: #FDF0F0; border-left: 6px solid #E15759;}
    .kpi-title {font-size: 1.1rem; font-weight: 600; margin: 0; padding-bottom: 5px;}
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

if 'df_hasil' not in st.session_state:
    st.session_state.df_hasil = None

# --- TAB NAVIGASI ---
tab1, tab2, tab3, tab4 = st.tabs(["💬 Analisis Teks Tunggal", "📁 Pemrosesan Massal", "📊 Dashboard Visualisasi", "📈 Evaluasi Model"])

# ==========================================
# TAB 1: ANALISIS TEKS TUNGGAL
# ==========================================
with tab1:
    st.subheader("Cek Ulasan Spesifik")
    user_input = st.text_area("Ketik keluhan atau ulasan secara manual:", placeholder="Contoh: Aplikasi JAKI lemot banget pas mau daftar vaksin, tolong diperbaiki!")
    
    if st.button("Prediksi Sentimen", type="primary"):
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
    st.info("⚠️ **Catatan Sistem:** Unggah file CSV/Excel hasil scraping dengan kolom ulasan ('Ulasan', 'Review', 'Komentar', atau 'Text').")
    
    uploaded_file = st.file_uploader("Unggah File (Drag and Drop)", type=['csv', 'xlsx'])
    
    if uploaded_file is None:
        st.session_state.df_hasil = None
    else:
        try:
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
                    st.success(f"✅ File tervalidasi! Memproses data dari kolom '{kolom_terpilih}' ({df.shape[0]} baris).")
                    
                    if st.button("Mulai Pembersihan & Analisis Massal", type="primary"):
                        with st.status("Memproses Data...", expanded=True) as status:
                            df_bersih = df.dropna(subset=['Ulasan']).copy()
                            df_bersih['Ulasan_Bersih'] = df_bersih['Ulasan'].apply(pembersihan_teks)
                            df_bersih = df_bersih[df_bersih['Ulasan_Bersih'].str.strip() != '']
                            
                            vektor_massal = tfidf_vect.transform(df_bersih['Ulasan_Bersih'])
                            df_bersih['Label_Sentimen'] = svm_model.predict(vektor_massal)
                            
                            st.session_state.df_hasil = df_bersih
                            status.update(label="Pemrosesan Selesai!", state="complete", expanded=False)
                        
                        with st.expander("📄 Klik untuk melihat sampel data yang telah diproses", expanded=True):
                            df_display = df_bersih.head(10).copy()
                            df_display.index = np.arange(1, len(df_display) + 1)
                            st.dataframe(df_display, use_container_width=True)
                        
                        csv_output = df_bersih.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Unduh Hasil Analisis (CSV)", data=csv_output, file_name='Hasil_JAKI.csv', mime='text/csv')
                        st.success("🎉 Berhasil! Buka **Tab 3** untuk Visualisasi atau **Tab 4** untuk Evaluasi Model.")
                        
        except Exception as e:
            st.error(f"Terjadi kesalahan teknis: {e}")
            st.session_state.df_hasil = None

# ==========================================
# TAB 3: DASHBOARD VISUALISASI EKSKLUSIF
# ==========================================
with tab3:
    if st.session_state.df_hasil is None:
        st.warning("⚠️ Silakan unggah dan proses data di **Tab 2** terlebih dahulu untuk mengaktifkan Dashboard.")
    else:
        df_viz = st.session_state.df_hasil
        total_ulasan = len(df_viz)
        jml_positif = len(df_viz[df_viz['Label_Sentimen'] == 'Positif'])
        jml_negatif = len(df_viz[df_viz['Label_Sentimen'] == 'Negatif'])
        
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-card kpi-total">
                <p class="kpi-title" style="color:#6c757d;">Total Ulasan Diproses</p>
                <p class="kpi-value" style="color:#343a40;">{total_ulasan:,}</p>
            </div>
            <div class="kpi-card kpi-positif">
                <p class="kpi-title" style="color:#4E79A7;">Sentimen Positif (Biru)</p>
                <p class="kpi-value" style="color:#4E79A7;">{jml_positif:,} <span style="font-size:1rem;">({round((jml_positif/total_ulasan)*100, 1)}%)</span></p>
            </div>
            <div class="kpi-card kpi-negatif">
                <p class="kpi-title" style="color:#E15759;">Sentimen Negatif (Merah)</p>
                <p class="kpi-value" style="color:#E15759;">{jml_negatif:,} <span style="font-size:1rem;">({round((jml_negatif/total_ulasan)*100, 1)}%)</span></p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("---")
        color_map = {'Positif':'#4E79A7', 'Negatif':'#E15759'}
        
        col1, col2, col3 = st.columns(3)
        sentimen_counts = df_viz['Label_Sentimen'].value_counts().reset_index()
        sentimen_counts.columns = ['Sentimen', 'Jumlah']
        
        with col1:
            fig_pie = px.pie(sentimen_counts, values='Jumlah', names='Sentimen', title='Sentimen Keseluruhan',
                             color='Sentimen', color_discrete_map=color_map, hole=0.4)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(margin=dict(t=40, b=20, l=10, r=10), showlegend=True)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col2:
            fig_bar = px.bar(sentimen_counts, x='Sentimen', y='Jumlah', title='Distribusi Jumlah',
                             text='Jumlah', color='Sentimen', color_discrete_map=color_map)
            fig_bar.update_traces(texttemplate='%{text}', textposition='outside')
            fig_bar.update_layout(margin=dict(t=40, b=20, l=10, r=10), showlegend=True)
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col3:
            if 'Jumlah_Like' in df_viz.columns:
                df_viz['Jumlah_Like'] = pd.to_numeric(df_viz['Jumlah_Like'], errors='coerce').fillna(0)
                df_viz['Ulasan_Wrap'] = df_viz['Ulasan'].apply(lambda x: '<br>'.join(textwrap.wrap(str(x), width=50)))
                
                fig_scatter = px.strip(df_viz, x='Label_Sentimen', y='Jumlah_Like', color='Label_Sentimen',
                                         title="Sebaran Interaksi Ulasan (Berdasarkan Like)",
                                         custom_data=['Ulasan_Wrap'], color_discrete_map=color_map)
                
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
                hari_map = {0:'Senin', 1:'Selasa', 2:'Rabu', 3:'Kamis', 4:'Jumat', 5:'Sabtu', 6:'Minggu'}
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
                
                fig_hm = px.imshow(pivot_data, labels=dict(x="Jam Masuk", y="", color="Jumlah"),
                                   x=pivot_data.columns, y=pivot_data.index, 
                                   color_continuous_scale="Blues", aspect="auto",
                                   title="Kecepatan Ulasan Peta Panas")
                
                fig_hm.update_traces(hovertemplate='<b>Hari:</b> %{y}<br><b>Jam:</b> %{x}:00<br><b>Jumlah Ulasan Masuk:</b> %{z}<extra></extra>')
                fig_hm.update_layout(margin=dict(t=30, b=20, l=10, r=10), coloraxis_showscale=False)
                st.plotly_chart(fig_hm, use_container_width=True)
            else:
                st.info("💡 Kolom 'Tanggal_dan_Jam' tidak ditemukan untuk membuat Heatmap.")

        def buat_wordcloud(teks_gabungan, title, colormap):
            if len(teks_gabungan) > 0:
                wc = WordCloud(width=500, height=350, background_color='white', colormap=colormap, max_words=80).generate(teks_gabungan)
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
                st.pyplot(fig_wc_pos, use_container_width=True)

        with col6:
            teks_neg = " ".join(review for review in df_viz[df_viz['Label_Sentimen'] == 'Negatif']['Ulasan_Bersih'])
            fig_wc_neg = buat_wordcloud(teks_neg, "Kata Kunci Negatif", "Reds")
            if fig_wc_neg: 
                st.pyplot(fig_wc_neg, use_container_width=True)


# ==========================================
# TAB 4: EVALUASI MODEL (CONFUSION MATRIX)
# ==========================================
with tab4:
    if st.session_state.df_hasil is None:
        st.warning("⚠️ Silakan unggah dan proses data di **Tab 2** terlebih dahulu.")
    else:
        st.subheader("Evaluasi Performa Model")
        st.info("💡 **Konsep Evaluasi:** Untuk menghitung Akurasi dan *Confusion Matrix*, sistem butuh membandingkan **Tebakan Sistem (Prediksi Model)** dengan **Kunci Jawaban (Nilai Aktual/Manual)**.")
        
        df_eval = st.session_state.df_hasil.copy()
        
        # Kolom yang tidak masuk akal untuk dijadikan Ground Truth
        kolom_dikecualikan = ['Label_Sentimen', 'Ulasan_Bersih', 'Ulasan']
        opsi_kolom = [col for col in df_eval.columns if col not in kolom_dikecualikan]

        st.markdown("Jika dataset Anda memiliki kolom **Label Manual** atau **Rating Bintang**, silakan pilih di bawah ini untuk melihat evaluasinya:")
        
        # DROPDOWN PILIHAN KUNCI JAWABAN (GROUND TRUTH)
        kolom_ground_truth = st.selectbox(
            "Pilih Kolom Kunci Jawaban (Ground Truth):",
            ["-- Pilih Kolom --"] + opsi_kolom
        )

        if kolom_ground_truth != "-- Pilih Kolom --":
            y_pred = df_eval['Label_Sentimen']
            
            # CEK APAKAH KOLOM YANG DIPILIH ANGKA (RATING) ATAU TEKS
            if pd.api.types.is_numeric_dtype(df_eval[kolom_ground_truth]):
                st.caption(f"ℹ️ *Sistem mendeteksi kolom **'{kolom_ground_truth}'** berisi Angka/Rating. Otomatis dikonversi: Bintang >= 4 (Positif), <= 3 (Negatif).*")
                df_eval['y_true_temp'] = pd.to_numeric(df_eval[kolom_ground_truth], errors='coerce').apply(
                    lambda x: 'Positif' if x >= 4 else 'Negatif'
                )
            else:
                st.caption(f"ℹ️ *Sistem menggunakan kolom teks **'{kolom_ground_truth}'** sebagai acuan jawaban.*")
                df_eval['y_true_temp'] = df_eval[kolom_ground_truth].astype(str).str.strip().str.title()
                # Menyamakan format jika ada yang menulis Positive / Negative dari bahasa Inggris
                df_eval['y_true_temp'] = df_eval['y_true_temp'].replace({'Positive': 'Positif', 'Negative': 'Negatif'})

            y_true = df_eval['y_true_temp']
            labels_urut = ['Positif', 'Negatif']

            # PASTIKAN KOLOM YANG DIPILIH HANYA MENGANDUNG KATA POSITIF / NEGATIF SETELAH DIKONVERSI
            if set(y_true.unique()).issubset(set(labels_urut)):
                col_eval1, col_eval2 = st.columns(2)
                
                with col_eval1:
                    st.markdown("#### Confusion Matrix Heatmap")
                    cm = confusion_matrix(y_true, y_pred, labels=labels_urut)
                    
                    fig_cm = px.imshow(cm, 
                                       labels=dict(x="Prediksi Model (Label_Sentimen)", y=f"Kunci Jawaban Asli ({kolom_ground_truth})", color="Jumlah"),
                                       x=labels_urut, 
                                       y=labels_urut,
                                       text_auto=True,
                                       color_continuous_scale='Blues')
                    fig_cm.update_xaxes(side="bottom")
                    fig_cm.update_layout(margin=dict(t=30, l=10, r=10, b=10))
                    st.plotly_chart(fig_cm, use_container_width=True)
                    
                with col_eval2:
                    st.markdown("#### Classification Report")
                    report = classification_report(y_true, y_pred, output_dict=True, labels=labels_urut, zero_division=0)
                    
                    akurasi = report.pop('accuracy')
                    df_report = pd.DataFrame(report).transpose()
                    df_report[['precision', 'recall', 'f1-score']] = df_report[['precision', 'recall', 'f1-score']].round(3)
                    df_report['support'] = df_report['support'].astype(int)
                    
                    st.dataframe(df_report, use_container_width=True)
                    st.success(f"🎯 **Tingkat Akurasi Model terhadap {kolom_ground_truth}: {akurasi * 100:.2f}%**")
            else:
                st.error(f"❌ Kolom '{kolom_ground_truth}' berisi format yang tidak didukung untuk evaluasi Sentimen (misal nama orang, tanggal, dll). Silakan pilih kolom yang berisi nilai manual sentimen atau rating.")
