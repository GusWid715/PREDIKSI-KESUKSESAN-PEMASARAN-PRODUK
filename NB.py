# 1. Import Pustaka yang dibutuhkan
# Untuk membuat tampilan Web App
import streamlit as st          
# Pandas untuk membuat tabel data
import pandas as pd             
# Algoritma Naive Bayes (Gaussian)
from sklearn.naive_bayes import GaussianNB 
# Preprocessing untuk mengubah kata menjadi angka
from sklearn import preprocessing

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