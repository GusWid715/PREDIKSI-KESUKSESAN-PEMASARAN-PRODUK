# 1. Import Pustaka yang dibutuhkan
# Untuk membuat tampilan Web App
import streamlit as st          
# Pandas untuk membuat tabel data
import pandas as pd             
# Algoritma Naive Bayes (Gaussian)
from sklearn.naive_bayes import GaussianNB 
# Preprocessing untuk mengubah kata menjadi angka
from sklearn import preprocessing

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

# 6. Menampilkan Informasi Model
st.subheader("2. Informasi Model")
st.write("Model berhasil dilatih menggunakan algoritma **Gaussian Naïve Bayes**.")
st.write(f"Jumlah data latih: {len(df)} baris.")
st.write("---")

# --- INTERAKSI PENGGUNA (PREDIKSI) ---
st.subheader("3. Coba Prediksi")
st.info("Pilih atribut di bawah ini untuk melihat hasil prediksi sistem.")

col1, col2 = st.columns(2)

with col1:
    input_iklan = st.selectbox("Apakah Melakukan Iklan?", df['Iklan'].unique())
with col2:
    input_harga = st.selectbox("Pilih Level Harga", df['Harga'].unique())

if st.button("🔍 Prediksi Kesuksesan"):
    # Ubah input user jadi angka
    iklan_kode = le_iklan.transform([input_iklan])[0]
    harga_kode = le_harga.transform([input_harga])[0]
    
    # Lakukan prediksi
    prediksi_angka = model.predict([[iklan_kode, harga_kode]])[0]
    
    # Kembalikan angka ke kata (inverse transform)
    prediksi_teks = le_kesuksesan.inverse_transform([prediksi_angka])[0]
    
    # Hitung probabilitas (keyakinan model)
    proba = model.predict_proba([[iklan_kode, harga_kode]])
    confidence = max(proba[0]) * 100
    
    # Tampilkan Hasil
    if prediksi_teks == 'Berhasil':
        st.success(f"Hasil Prediksi: **{prediksi_teks}** (Tingkat Keyakinan: {confidence:.2f}%)")
    else:
        st.error(f"Hasil Prediksi: **{prediksi_teks}** (Tingkat Keyakinan: {confidence:.2f}%)")