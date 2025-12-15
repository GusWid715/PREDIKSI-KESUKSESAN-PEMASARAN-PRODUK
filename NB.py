# 1. Import Pustaka yang dibutuhkan
# Untuk membuat tampilan Web App
import streamlit as st          
# Pandas untuk membuat tabel data
import pandas as pd             
# Tree untuk membuat pohon keputusan
from sklearn import tree        
# Preprocessing untuk mengubah kata menjadi angka
from sklearn import preprocessing 
# Untuk menampilkan aturan teks
from sklearn.tree import export_text 
# Untuk visualisasi grafik pohon
import matplotlib.pyplot as plt 

# Konfigurasi Halaman Streamlit
st.title("Klasifikasi Kategori Pakaian Dengan Menggunakan ID3")
st.write("Kelompok x - Pemodelan dan Simulasi")
st.write("---")

# 2. Menyiapkan Data 
data = {
    'Warna': ['Merah', 'Biru', 'Hijau', 'Kuning', 'Merah', 'Biru', 'Hijau', 'Kuning'],
    'Ukuran': ['S', 'M', 'L', 'S', 'M', 'L', 'S', 'M'],
    'Bahan': ['Katun', 'Sutra', 'Wol', 'Sutra', 'Katun', 'Wol', 'Katun', 'Sutra'],
    'Kategori': ['Casual', 'Formal', 'Casual', 'Casual', 'Casual', 'Formal', 'Casual', 'Formal']
}

# Membuat tabel (DataFrame) agar rapi
df = pd.DataFrame(data)

st.subheader("1. Data Awal")
st.dataframe(df, use_container_width=True)

# 3. Preprocessing (Mengubah Kata menjadi Angka)
# LabelEncoder terpisah untuk setiap kolom agar bisa dibalikkan nanti
le_warna = preprocessing.LabelEncoder()
le_ukuran = preprocessing.LabelEncoder()
le_bahan = preprocessing.LabelEncoder()
le_kategori = preprocessing.LabelEncoder() # Penting: Simpan encoder kategori

# Buat salinan tabel untuk menampung angka
df_encoded = df.copy()

# Lakukan transformasi (Kata -> Angka)
df_encoded['Warna'] = le_warna.fit_transform(df['Warna'])
df_encoded['Ukuran'] = le_ukuran.fit_transform(df['Ukuran'])
df_encoded['Bahan'] = le_bahan.fit_transform(df['Bahan'])
df_encoded['Kategori'] = le_kategori.fit_transform(df['Kategori'])

# 4. Memisahkan Data
# Fitur: Data penentu (Warna, Ukuran, Bahan)
features = df_encoded[['Warna', 'Ukuran', 'Bahan']]
# Target: Hasil yang dicari (Kategori)
target = df_encoded['Kategori']

# 5. Membuat Model Pohon Keputusan (ID3)
# criterion='entropy' digunakan agar mirip dengan rumus ID3 manual
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(features, target)

# 6. Menampilkan Hasil Pohon

# Mengambil nama fitur agar aturan mudah dibaca
r = export_text(clf, feature_names=['Warna', 'Ukuran', 'Bahan'])

# Output agar tidak berupa angka
# Ambil daftar nama kategori asli: ['Casual', 'Formal']
# Karena Casual urutan abjad pertama, dia jadi 0. Formal jadi 1.
daftar_kategori = list(le_kategori.classes_) 

# Ganti teks "class: 0" menjadi "class: Casual", dst.
for i, nama in enumerate(daftar_kategori):
    # Logika: ganti tulisan 'class: 0' dengan 'class: Casual'
    r = r.replace(f"class: {i}", f"class: {nama}")

# --- VISUALISASI GRAFIK POHON ---
st.subheader("2. Visualisasi Grafik Pohon")
fig, ax = plt.subplots(figsize=(12, 6))
tree.plot_tree(clf, 
              feature_names=['Warna', 'Ukuran', 'Bahan'], 
              class_names=le_kategori.classes_,
              filled=True, 
              rounded=True)
st.pyplot(fig)

# --- INTERAKSI PENGGUNA (PREDIKSI) ---
st.subheader("3. Coba Prediksi")
st.info("Pilih atribut di bawah ini untuk melihat hasil prediksi sistem.")

col1, col2, col3 = st.columns(3)

with col1:
    input_warna = st.selectbox("Pilih Warna", df['Warna'].unique())
with col2:
    input_ukuran = st.selectbox("Pilih Ukuran", df['Ukuran'].unique())
with col3:
    input_bahan = st.selectbox("Pilih Bahan", df['Bahan'].unique())

if st.button("🔍 Prediksi Kategori"):
    # Ubah input user jadi angka
    warna_kode = le_warna.transform([input_warna])[0]
    ukuran_kode = le_ukuran.transform([input_ukuran])[0]
    bahan_kode = le_bahan.transform([input_bahan])[0]
    
    # Lakukan prediksi
    prediksi_angka = clf.predict([[warna_kode, ukuran_kode, bahan_kode]])[0]
    
    # Kembalikan angka ke kata (inverse transform)
    prediksi_teks = le_kategori.inverse_transform([prediksi_angka])[0]
    
    st.success(f"Hasil Prediksi: **{prediksi_teks}**")