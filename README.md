# 🚀 YOLOv8 - Sistem Deteksi Man/Woman

Sistem deteksi objek canggih berbasis **YOLOv8** dengan fitur pemrosesan kamera, penyimpanan hasil, dan optimasi performa.

---

## 📌 Ringkasan
Proyek ini mengimplementasikan **sistem deteksi objek real-time** dengan **YOLOv8**, mendukung berbagai sumber input seperti **kamera live** dan **gambar statis**. Dengan antarmuka yang intuitif, sistem ini memberikan pengalaman deteksi yang efisien dan terorganisir.

---

## ✨ Fitur Utama

✅ **Deteksi objek real-time** dari webcam atau kamera eksternal
✅ **Penghitungan berdasarkan kelas** untuk analisis lebih mendalam
✅ **Pemantauan FPS** untuk menilai kinerja deteksi
✅ **Simpan gambar dengan satu klik** saat deteksi berlangsung
✅ **Rekaman video otomatis** dari sesi deteksi
✅ **Manajemen output terorganisir** untuk gambar dan video
✅ **Ambang batas deteksi yang dapat dikonfigurasi** untuk Confidence dan IoU

---

## 🔧 Teknologi yang Digunakan

- **Python 3.x** → Bahasa pemrograman utama
- **OpenCV** → Pengolahan video & kamera
- **Ultralytics YOLOv8** → Model deteksi objek mutakhir
- **NumPy** → Pemrosesan data numerik
- **PyTorch** → Framework pembelajaran mesin

---

## ⚙️ Instalasi

### 📌 Prasyarat
- **Python** 3.6 atau lebih tinggi
- **pip** (manajer paket Python)
- **Conda** *(opsional, tetapi disarankan untuk lingkungan virtual)*

### 🔹 Langkah Instalasi

1️⃣ **Klon repositori**
```bash
git clone https://github.com/manap01/detections_man_woman.git
cd detections_man_woman
```

2️⃣ **Buat dan aktifkan lingkungan virtual** *(disarankan untuk isolasi dependensi)*
```bash
# Menggunakan venv
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Atau menggunakan conda
conda create -n detections_man_woman python=3.8 -y
conda activate detections_man_woman
```

3️⃣ **Instal dependensi**
```bash
pip install -r requirements.txt
```

4️⃣ **Instal PyTorch** *(jika belum otomatis terinstal)*
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

5️⃣ **Instal dependensi tambahan untuk YOLOv8**
```bash
pip install ultralytics-thop>=2.0.0
```

6️⃣ **Unduh dan tempatkan model YOLOv8**
- Simpan model YOLOv8 (`best.pt`) di dalam folder **`models/`**

---

## 🚀 Cara Menggunakan

### 🔹 Menjalankan Deteksi Kamera
```bash
python src/detect.py
```

### 🔹 Kontrol Keyboard
- **'s'** → Simpan frame deteksi
- **'q'** → Keluar dari aplikasi

### 🔹 Kustomisasi Parameter Deteksi
```python
# Contoh: Mengubah confidence threshold dan ID kamera
detector = ObjectDetector(model_path, conf_threshold=0.4)
detector.detect_from_camera(camera_id=1, save_video=True)
```

### 🔹 Memproses Gambar Statis
```python
from detect import ObjectDetector

detector = ObjectDetector("models/best.pt")
result_image = detector.detect_on_image("path/to/your/image.jpg")
```

---

## 🛠️ Konfigurasi Parameter

| Parameter         | Deskripsi |
|------------------|-----------|
| `model_path`     | Path ke file model YOLOv8 |
| `conf_threshold` | Ambang batas kepercayaan deteksi (default: 0.5) |
| `iou_threshold`  | Ambang batas IoU untuk Non-Maximum Suppression (default: 0.45) |

---

## ⚠️ Troubleshooting

### ❌ **1. ModuleNotFoundError: No module named 'torch'**
✅ **Solusi:** Instal PyTorch:
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### ❌ **2. Dependensi `ultralytics-thop` hilang**
✅ **Solusi:**
```bash
pip install ultralytics-thop>=2.0.0
```

### ❌ **3. Model YOLOv8 tidak ditemukan**
✅ **Solusi:** Pastikan file `best.pt` berada di folder `models/`

### ❌ **4. OpenCV Error saat mengakses kamera**
✅ **Solusi:**
```bash
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```
Jika hasilnya `False`, restart sistem atau perbarui OpenCV:
```bash
pip install --upgrade opencv-python
```

---

## 🎓 Cara Melatih Model Sendiri

Untuk melatih model YOLOv8 sendiri:

1️⃣ Siapkan dataset dalam format YOLOv8.
2️⃣ Jalankan perintah berikut:
```bash
!yolo task=detect mode=train data=path/to/data.yaml model=yolov8s.pt epochs=100 imgsz=640
```
3️⃣ Simpan model hasil pelatihan (`best.pt`) di folder `models/`

---

## 🌐 Sumber Daya Tambahan
- [📖 Dokumentasi YOLOv8](https://docs.ultralytics.com/)
- [🔬 Contoh Google Colab](https://colab.research.google.com/github/ultralytics/yolov8/blob/main/examples/tutorial.ipynb)
- [📊 Persiapan Dataset dengan Roboflow](https://roboflow.com/)

---

## 🤝 Kontribusi
Kami terbuka untuk kontribusi! Silakan ajukan **Pull Request** atau **Issue** untuk perbaikan dan fitur baru.

---

## 📝 Lisensi
Proyek ini dilisensikan di bawah **MIT License** - lihat file `LICENSE` untuk detail lebih lanjut.

---

## 👥 Tim Pengembang

KELOMPOK1

---

## 📩 Kontak  
Untuk pertanyaan atau dukungan, silakan [buka issue](https://github.com/manap01/detections_man_woman/issues/1) di repositori ini.

