# Wildfire CNN Detection & Scenario Comparison Dashboard

Proyek ini adalah dashboard berbasis web interaktif untuk mendeteksi kebakaran hutan menggunakan Convolutional Neural Network (CNN) dengan framework Keras/TensorFlow. Proyek ini membandingkan 3 skenario model CNN yang dilatih dengan parameter yang berbeda (Learning Rate dan jumlah Epoch).

---

## 🚀 Fitur Dashboard
1. **Scenario Comparison**: Membandingkan hasil prediksi dari 3 skenario model secara berdampingan (*side-by-side*) dengan visualisasi gauge interaktif.
2. **Dataset Explorer**: Menjelajahi 61 citra kebakaran hutan bawaan dari dataset secara instan.
3. **Custom Inference**: Unggah citra kebakaran hutan baru menggunakan fitur drag & drop untuk diuji secara real-time.
4. **Evaluation Chart Generator**: Menjalankan evaluasi penuh untuk menghasilkan diagram performa rata-rata tingkat kepercayaan (*confidence*) dan sensitivitas (*recall*).

---

## 🛠️ Cara Menjalankan Proyek Secara Lokal

### 1. Kloning Repository
Kloning repository ini ke komputer lokal Anda:
```bash
git clone https://github.com/thoriqhdap/wildfire-cnn-detection.git
cd wildfire-cnn-detection
```

### 2. Unduh dan Masukkan File Model (`.keras`)
Karena ukuran model sangat besar, file `.keras` diabaikan dari git. Anda harus mengunduh file model secara terpisah dan meletakkannya di dalam folder `models/`:

* **S1_LR_Besar.keras** -> Letakkan di `models/S1_LR_Besar.keras`
* **S2_LR_Kecil.keras** -> Letakkan di `models/S2_LR_Kecil.keras`
* **S5_Epoch_Tinggi.keras** -> Letakkan di `models/S5_Epoch_Tinggi.keras`

> [!IMPORTANT]
> Silakan unduh file model `.keras` di atas melalui link Google Drive berikut:
> 🔗 **[MASUKKAN LINK DOWNLOAD GOOGLE DRIVE ANDA DISINI]**

---

### 3. Persiapan Virtual Environment & Dependensi
Pastikan Python 3.10 telah terinstal. Jalankan perintah berikut untuk membuat virtual environment dan menginstal library yang diperlukan:

**Di Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

**Di Windows (CMD):**
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

### 4. Menjalankan Kode Evaluasi (Terminal)
Untuk melihat statistik performa model langsung di dalam terminal dan menghasilkan grafik perbandingan baru:
```bash
python evaluate.py
```
Grafik visualisasi akan disimpan di folder `static/evaluation_results.png` dan `static/predictions_grid.png`.

---

### 5. Menjalankan Server Web Dashboard
Untuk mengaktifkan antarmuka web interaktif:
```bash
python app.py
```
Setelah server aktif, buka web browser Anda dan akses alamat berikut:
👉 **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)**

---

## 📊 Hasil Evaluasi Singkat
* **S1_LR_Besar (Learning Rate = 0.01)**: Performa kurang optimal (Recall ~9.84%).
* **S2_LR_Kecil (Learning Rate = 0.0001)**: Performa terbaik dengan tingkat kepekaan tinggi (Recall 100%).
* **S5_Epoch_Tinggi (Learning Rate = 0.001)**: Performa sangat baik dan stabil (Recall ~95.08%).
