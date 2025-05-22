# 🧬 Deteksi Sel Leukemia & Malaria

Proyek ini adalah aplikasi klasifikasi citra berbasis deep learning untuk mendeteksi **Acute Lymphoblastic Leukemia (ALL)** dan **Malaria** menggunakan gambar mikroskopis sel darah.

---

## 📁 Struktur File

- `app.py` — Aplikasi Gradio siap pakai untuk prediksi gabungan ALL & Malaria.
- `combined.py` — Script pelatihan model gabungan untuk 6 kelas (4 ALL + 2 Malaria).
- `leukimia.py` — Script pelatihan model khusus leukemia (4 kelas).
- `malaria.py` — Script pelatihan model khusus malaria (2 kelas).

---

## 🔬 Kelas yang Didukung

**Leukemia (ALL):**

- `all_benign`
- `all_early`
- `all_pre`
- `all_pro`

**Malaria:**

- `malaria_parasitized`
- `malaria_uninfected`

---


---

## 🖥️ Cara Install Visual Studio Code

Untuk pemula, berikut panduan singkat instalasi **VS Code**:

1. **Download:**
   - Kunjungi [https://code.visualstudio.com](https://code.visualstudio.com)
   - Pilih sistem operasi (Windows, macOS, Linux)
   - Klik tombol **Download**

2. **Instalasi:**
   - Jalankan file `.exe` atau `.dmg` yang sudah diunduh
   - Ikuti proses instalasi dengan pilihan default

3. **Buka Proyek:**
   - Buka **VS Code**
   - Klik `File > Open Folder...`
   - Pilih folder proyek ini

4. **Install Python Extension (jika belum ada):**
   - Klik ikon **Extensions** (persegi kiri samping)
   - Cari: `Python` oleh Microsoft
   - Klik **Install**
