# 1. Import Pustaka yang dibutuhkan
# Untuk membuat tampilan Web App
import streamlit as st          
# Pandas untuk membuat tabel data
import pandas as pd             
# Algoritma Naive Bayes (Gaussian)
from sklearn.naive_bayes import GaussianNB 
# Preprocessing untuk mengubah kata menjadi angka
from sklearn import preprocessing
# Untuk operasi fungsi matematika.
import numpy as np
# Visualisasi Grafik
import matplotlib.pyplot as plt

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Tugas Naive Bayes", layout="centered")
st.title("Prediksi Kesuksesan Pemasaran Dengan Naïve Bayes")
st.write("Kelompok x - Pemodelan dan Simulasi")
st.write("---")


# 2. Menyiapkan Data 
data = {
    'Iklan': ['Ya', 'Tidak', 'Ya', 'Tidak'],
    'Harga': ['Rendah', 'Tinggi', 'Sedang', 'Rendah'],
    'Kesuksesan': ['Berhasil', 'Tidak Berhasil', 'Berhasil', 'Tidak Berhasil']
}

# Membuat tabel (DataFrame) agar rapi
df = pd.DataFrame(data)

st.subheader("1. Data Awal")
st.dataframe(df, use_container_width=True)

# 3. Preprocessing (Mengubah Kata menjadi Angka)
# LabelEncoder terpisah untuk setiap kolom agar bisa dibalikkan nanti
le_iklan = preprocessing.LabelEncoder()
le_harga = preprocessing.LabelEncoder()
le_kesuksesan = preprocessing.LabelEncoder() # Penting: Simpan encoder target

# Buat salinan tabel untuk menampung angka
df_encoded = df.copy()

# Lakukan transformasi (Kata -> Angka)
df_encoded['Iklan'] = le_iklan.fit_transform(df['Iklan'])
df_encoded['Harga'] = le_harga.fit_transform(df['Harga'])
df_encoded['Kesuksesan'] = le_kesuksesan.fit_transform(df['Kesuksesan'])

# 4. Memisahkan Data
# Fitur: Data penentu (Iklan, Harga)
features = df_encoded[['Iklan', 'Harga']]
# Target: Hasil yang dicari (Kesuksesan)
target = df_encoded['Kesuksesan']

# 5. Membuat Model Naive Bayes
# Menggunakan GaussianNB sesuai standar klasifikasi Naive Bayes
model = GaussianNB()
model.fit(features, target)

# 6. Interaksi Pengguna
st.subheader("2. Simulasi Prediksi")
st.info("Masukkan strategi pemasaran baru untuk melihat prediksi peluangnya.")

col1, col2 = st.columns(2)

with col1:
    input_iklan = st.selectbox("Apakah Melakukan Iklan?", df['Iklan'].unique())
with col2:
    input_harga = st.selectbox("Pilih Level Harga", df['Harga'].unique())

if st.button("🔍 Hitung Prediksi"):
    st.write("---")
    st.subheader("3. Hasil Analisa Perhitungan")
    
    # Ubah input user jadi angka
    iklan_kode = le_iklan.transform([input_iklan])[0]
    harga_kode = le_harga.transform([input_harga])[0]
    
    # Lakukan prediksi kelas (Hasil Akhir)
    prediksi_angka = model.predict([[iklan_kode, harga_kode]])[0]
    prediksi_teks = le_kesuksesan.inverse_transform([prediksi_angka])[0]
    
    # --- BAGIAN INI MENAMPILKAN PROSES PERHITUNGAN ---
    # Ambil probabilitas untuk setiap kelas (Berhasil vs Tidak Berhasil)
    proba = model.predict_proba([[iklan_kode, harga_kode]])[0]
    kelas = le_kesuksesan.classes_
    
    # Tampilkan detail perhitungan probabilitas
    st.write("Sistem menghitung probabilitas (Posterior Probability) untuk setiap kemungkinan:")
    
    c1, c2 = st.columns(2)
    
    # Menampilkan Probabilitas Kelas Pertama
    with c1:
        st.metric(
            label=f"Peluang {kelas[0]}", 
            value=f"{proba[0]*100:.2f}%"
        )
    
    # Menampilkan Probabilitas Kelas Kedua
    with c2:
        st.metric(
            label=f"Peluang {kelas[1]}", 
            value=f"{proba[1]*100:.2f}%"
        )
    
    # Visualisasi Grafik Batang Sederhana
    fig, ax = plt.subplots(figsize=(6, 2))
    bars = ax.barh(kelas, proba, color=['#4CAF50', '#F44336'])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probabilitas")
    
    # Tambahkan label angka di grafik
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width*100:.1f}%', va='center')
        
    st.pyplot(fig)
    
    # Kesimpulan Akhir
    st.success(f"**Kesimpulan:** Berdasarkan nilai probabilitas tertinggi, sistem memprediksi hasil: **{prediksi_teks}**")