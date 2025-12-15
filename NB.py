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