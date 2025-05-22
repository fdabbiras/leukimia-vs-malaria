🧬 Deteksi Sel Leukemia & Malaria Otomatis
Proyek ini adalah aplikasi klasifikasi citra berbasis deep learning untuk mendeteksi Acute Lymphoblastic Leukemia (ALL) dan Malaria menggunakan gambar mikroskopis sel darah.

📂 Struktur File
app.py — Aplikasi Gradio siap pakai untuk prediksi gabungan ALL & Malaria.
combined.py — Script pelatihan model gabungan untuk 6 kelas (4 ALL + 2 Malaria).
leukimia.py — Script pelatihan model khusus leukemia (4 kelas).
malaria.py — Script pelatihan model khusus malaria (2 kelas).

📊 Kelas yang Didukung
Leukemia (ALL):
all_benign
all_early
all_pre
all_pro

Malaria:
malaria_parasitized
malaria_uninfected

🖥️ Cara Instalasi Visual Studio Code (VS Code)
Untuk pengguna pemula, ikuti langkah berikut untuk menginstal VS Code:
Download VS Code:
Kunjungi https://code.visualstudio.com
Pilih sesuai sistem operasi: Windows, macOS, atau Linux
Klik tombol Download

Instalasi:
Jalankan file .exe atau .dmg yang sudah diunduh
Ikuti proses instalasi dengan opsi default

Buka Project:
Buka VS Code
Pilih menu File > Open Folder...
Arahkan ke folder proyek ini

Install Python Extension (jika belum ada):
Buka Extensions (ikon persegi di sidebar kiri)
Cari: Python oleh Microsoft
Klik Install
