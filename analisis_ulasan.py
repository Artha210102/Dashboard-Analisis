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
st.set_page_config(page_title="Sentimen JAKI | Pemprov DKI", page_icon="🏛️", layout="wide")

# --- CIVIC INTELLIGENCE UI: CUSTOM CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --navy: #142B3B;
    --navy-deep: #0D202D;
    --teal: #0F8B8D;
    --teal-light: #E8F6F4;
    --ink: #243746;
    --muted: #71808D;
    --line: #E2E9EC;
    --surface: #FFFFFF;
    --canvas: #F4F7F8;
}

.stApp {
    background: var(--canvas);
    color: var(--ink);
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stHeader"] { background: rgba(244,247,248,0.92); }
.block-container {
    max-width: 1440px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
h1, h2, h3, h4, p, label, [data-testid="stMarkdownContainer"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 { color: var(--navy); letter-spacing: -0.025em; }
h2 { font-weight: 750; }
h3 { font-weight: 700; }

/* Editorial header */
.top-banner {
    position: relative;
    overflow: hidden;
    background: linear-gradient(115deg, #102838 0%, #173F50 70%, #126E72 100%);
    color: #FFFFFF;
    padding: 34px 38px 32px;
    border-radius: 18px;
    margin: 0 0 28px 0;
    box-shadow: 0 12px 30px rgba(17, 45, 60, 0.10);
    border: 1px solid rgba(255,255,255,0.10);
}
.top-banner:after {
    content: "";
    position: absolute;
    width: 250px;
    height: 250px;
    right: -65px;
    top: -115px;
    border: 1px solid rgba(255,255,255,0.16);
    border-radius: 50%;
    box-shadow: 0 0 0 28px rgba(255,255,255,0.035),
                0 0 0 58px rgba(255,255,255,0.025);
    pointer-events: none;
}
.eyebrow {
    display: inline-block;
    color: #A9E4DD;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.main-header {
    font-family: 'Manrope', sans-serif;
    font-size: clamp(1.7rem, 3vw, 2.45rem);
    line-height: 1.2;
    color: #FFFFFF;
    font-weight: 800;
    margin: 0 0 9px 0;
    padding: 0;
}
.sub-header { color: #D0E0E6; margin: 0; font-size: 1rem; }

/* Tabs: clean navigation with teal active indicator */
div[data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid var(--line);
    margin-bottom: 22px;
}
div[data-baseweb="tab"] {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 10px 10px 0 0;
    padding: 12px 18px;
    color: #637582;
    font-weight: 650;
    transition: color .18s ease, background .18s ease;
}
div[data-baseweb="tab"]:hover {
    color: var(--teal);
    background: #EAF3F4;
}
div[data-baseweb="tab"][aria-selected="true"] {
    background: #FFFFFF;
    color: var(--navy) !important;
    border-color: var(--line);
    border-bottom: 3px solid var(--teal);
}
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] { display: none; }

/* Main content panels */
.content-box {
    background: var(--surface);
    padding: 28px 30px;
    border-radius: 16px;
    border: 1px solid var(--line);
    box-shadow: 0 5px 18px rgba(24, 52, 66, 0.035);
    margin-bottom: 20px;
}
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px;
}
div[data-testid="stTextArea"] textarea,
div[data-testid="stTextInput"] input,
div[data-testid="stFileUploader"] section {
    border-radius: 10px;
}
div[data-testid="stTextArea"] textarea:focus,
div[data-testid="stTextInput"] input:focus {
    border-color: var(--teal);
    box-shadow: 0 0 0 1px var(--teal);
}
.stButton > button, .stDownloadButton > button {
    border-radius: 9px;
    font-weight: 700;
    min-height: 2.7rem;
    padding: 0.55rem 1.15rem;
    transition: all .18s ease;
}
.stButton > button[kind="primary"] {
    background: var(--teal);
    border: 1px solid var(--teal);
    color: #FFFFFF;
}
.stButton > button[kind="primary"]:hover {
    background: #0B7375;
    border-color: #0B7375;
    box-shadow: 0 5px 12px rgba(15,139,141,.18);
}
.stDownloadButton > button {
    border: 1px solid #B8D8D6;
    color: #116B6D;
    background: #F1FAF9;
}
div[data-testid="stAlert"] { border-radius: 10px; }

/* Result states */
.success-box {
    padding: 17px 19px;
    border-radius: 11px;
    background: #ECF8F4;
    border: 1px solid #BCE6D8;
    border-left: 5px solid #159A75;
    color: #12684F;
    margin-top: 16px;
}
.success-box h3 { color: #12684F; margin: 0; font-size: 1.1rem; }
.error-box {
    padding: 17px 19px;
    border-radius: 11px;
    background: #FFF2F0;
    border: 1px solid #F3D0CA;
    border-left: 5px solid #D95D4F;
    color: #9D342A;
    margin-top: 16px;
}
.error-box h3 { color: #9D342A; margin: 0; font-size: 1.1rem; }

/* Dashboard KPI cards */
.kpi-container {
    display: flex;
    justify-content: space-between;
    gap: 16px;
    margin: 8px 0 26px;
}
.kpi-card {
    flex: 1;
    min-width: 0;
    padding: 22px 23px;
    border-radius: 14px;
    background: #FFFFFF;
    border: 1px solid var(--line);
    box-shadow: 0 5px 18px rgba(24,52,66,.035);
    text-align: left;
    position: relative;
    overflow: hidden;
}
.kpi-card:before {
    content: "";
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 4px;
    background: #94A3AD;
}
.kpi-total:before { background: #607785; }
.kpi-positif:before { background: #0F8B8D; }
.kpi-negatif:before { background: #D95D4F; }
.kpi-title {
    font-size: .83rem;
    font-weight: 700;
    margin: 0;
    padding-bottom: 10px;
    letter-spacing: .035em;
    text-transform: uppercase;
    color: #71808D;
}
.kpi-value {
    font-family: 'Manrope', sans-serif;
    font-size: clamp(1.8rem, 3vw, 2.5rem);
    line-height: 1.1;
    font-weight: 800;
    margin: 0;
}
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
hr { border-color: var(--line); }
small, .stCaption { color: var(--muted); }

@media (max-width: 760px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    .top-banner { padding: 25px 22px; }
    .content-box { padding: 20px 16px; }
    .kpi-container { flex-direction: column; }
    div[data-baseweb="tab"] { padding: 9px 10px; font-size: .82rem; }
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

# --- HEADER APLIKASI MODERN ---
st.markdown("""
    <div class="top-banner">
        <div class="eyebrow">Civic Intelligence · Analisis Data Publik</div>
        <p class="main-header">Dashboard Analisis Sentimen JAKI</p>
        <p class="sub-header">Pemantauan ulasan aplikasi pelayanan masyarakat · Model Support Vector Machine (SVM)</p>
    </div>
""", unsafe_allow_html=True)

if 'df_hasil' not in st.session_state:
    st.session_state.df_hasil = None

# --- TAB NAVIGASI ---
tab1, tab2, tab3 = st.tabs(["💬 Analisis Teks Tunggal", "📁 Pemrosesan Massal", "📊 Dashboard Visualisasi"])

# ==========================================
# TAB 1: ANALISIS TEKS TUNGGAL
# ==========================================
with tab1:
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
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
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 2: PEMROSESAN MASSAL
# ==========================================
with tab2:
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
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
                        st.success("🎉 Berhasil! Buka **Tab 3** untuk melihat Dashboard Visualisasi Anda.")
                        
        except Exception as e:
            st.error(f"Terjadi kesalahan teknis: {e}")
            st.session_state.df_hasil = None
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 3: DASHBOARD VISUALISASI EKSKLUSIF
# ==========================================
with tab3:
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
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
        
        header_style = '<p style="font-size:16px; font-weight:bold; color:#31333F; margin-bottom:10px; text-align: center;">{}</p>'
        
        # DEFINISIKAN TINGGI YANG SAMA UNTUK KETIGA GRAFIK DI BARIS BAWAH
        TINGGI_GRAFIK = 420 

        col4, col5, col6 = st.columns(3)
        
        with col4:
            st.markdown(header_style.format("Kecepatan Ulasan Peta Panas"), unsafe_allow_html=True)
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
                                   color_continuous_scale="Blues", aspect="auto")
                
                fig_hm.update_traces(hovertemplate='<b>Hari:</b> %{y}<br><b>Jam:</b> %{x}:00<br><b>Jumlah Ulasan Masuk:</b> %{z}<extra></extra>')
                # Set Tinggi Tetap (420px)
                fig_hm.update_layout(height=TINGGI_GRAFIK, margin=dict(t=10, b=30, l=10, r=10), coloraxis_showscale=False)
                st.plotly_chart(fig_hm, use_container_width=True)
            else:
                st.info("💡 Kolom 'Tanggal_dan_Jam' tidak ditemukan untuk membuat Heatmap.")

        # FUNGSI BARU: Hasilkan array gambar WordCloud (Resolusi tinggi, bentuk kotak)
        def buat_wordcloud_array(teks_gabungan, colormap):
            if len(teks_gabungan) > 0:
                wc = WordCloud(width=800, height=800, background_color='white', colormap=colormap, max_words=80).generate(teks_gabungan)
                return wc.to_array() 
            return None

        with col5:
            st.markdown(header_style.format("Kata Kunci Positif"), unsafe_allow_html=True)
            teks_pos = " ".join(review for review in df_viz[df_viz['Label_Sentimen'] == 'Positif']['Ulasan_Bersih'])
            img_pos = buat_wordcloud_array(teks_pos, "Blues")
            if img_pos is not None: 
                # BUNGKUS WORD CLOUD KE DALAM PLOTLY AGAR SIFATNYA SAMA DENGAN HEATMAP
                fig_wc_pos = px.imshow(img_pos)
                fig_wc_pos.update_xaxes(visible=False) # Sembunyikan sumbu X
                fig_wc_pos.update_yaxes(visible=False) # Sembunyikan sumbu Y
                fig_wc_pos.update_layout(height=TINGGI_GRAFIK, margin=dict(t=10, b=30, l=10, r=10), hovermode=False)
                st.plotly_chart(fig_wc_pos, use_container_width=True)

        with col6:
            st.markdown(header_style.format("Kata Kunci Negatif"), unsafe_allow_html=True)
            teks_neg = " ".join(review for review in df_viz[df_viz['Label_Sentimen'] == 'Negatif']['Ulasan_Bersih'])
            img_neg = buat_wordcloud_array(teks_neg, "Reds")
            if img_neg is not None: 
                # BUNGKUS WORD CLOUD KE DALAM PLOTLY AGAR SIFATNYA SAMA DENGAN HEATMAP
                fig_wc_neg = px.imshow(img_neg)
                fig_wc_neg.update_xaxes(visible=False) # Sembunyikan sumbu X
                fig_wc_neg.update_yaxes(visible=False) # Sembunyikan sumbu Y
                fig_wc_neg.update_layout(height=TINGGI_GRAFIK, margin=dict(t=10, b=30, l=10, r=10), hovermode=False)
                st.plotly_chart(fig_wc_neg, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
