📈 Prediksi Kesuksesan Pemasaran Produk (Naïve Bayes)

Tugas Kelompok 2 - Mata Kuliah Pemodelan dan Simulasi Program Studi Informatika - Universitas Udayana

Aplikasi ini dibangun untuk memprediksi potensi keberhasilan pemasaran suatu produk (Berhasil atau Tidak Berhasil) berdasarkan strategi Iklan dan Harga menggunakan algoritma Gaussian Naïve Bayes.

📋 Fitur Utama

Input Interaktif: Pengguna dapat memilih strategi pemasaran (Iklan & Harga) melalui antarmuka web.

Prediksi Real-time: Sistem langsung menghitung hasil prediksi menggunakan model Naïve Bayes yang telah dilatih.

Visualisasi Probabilitas: Menampilkan grafik tingkat keyakinan (confidence level) sistem terhadap prediksi yang diberikan.

Model Machine Learning: Menggunakan pustaka Scikit-Learn yang handal dan akurat.

🛠️ Teknologi yang Digunakan

Proyek ini dikembangkan menggunakan bahasa pemrograman Python dengan pustaka berikut:

Streamlit: Framework untuk membuat antarmuka web (UI).

Scikit-Learn: Implementasi algoritma Gaussian Naïve Bayes.

Pandas: Manipulasi data tabular.

Matplotlib: Visualisasi grafik batang probabilitas.

NumPy: Komputasi numerik.

🚀 Cara Menjalankan Aplikasi

Ikuti langkah-langkah berikut untuk menjalankan aplikasi di komputer lokal Anda.

1. Persiapan Environment

Pastikan Anda memiliki Python terinstal. Disarankan menggunakan Virtual Environment agar bersih.

# Buat environment baru (Windows)
python -m venv myenv

# Aktifkan environment
myenv\Scripts\activate


2. Instalasi Pustaka

Instal semua dependensi yang diperlukan menggunakan pip.

pip install -r requirements.txt


Catatan: Jika file requirements.txt belum ada, instal manual dengan perintah:
pip install streamlit pandas scikit-learn matplotlib numpy

3. Jalankan Program

Gunakan perintah streamlit untuk memulai aplikasi.

streamlit run NB.py


Setelah berhasil, browser akan otomatis terbuka di alamat: http://localhost:8501.

📂 Struktur File

NB.py: Kode utama aplikasi (berisi model Naïve Bayes dan UI Streamlit).

requirements.txt: Daftar pustaka Python yang dibutuhkan.

README.md: Dokumentasi proyek ini.

👥 Kredit

Kelompok [X] - Pemodelan dan Simulasi

Anggota 1 (NIM)

Anggota 2 (NIM)

Anggota 3 (NIM)

...
