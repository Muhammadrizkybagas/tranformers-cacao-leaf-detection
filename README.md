# Cocoa Leaf Disease Detection (CSSVD)

## Deskripsi Project
Project ini bertujuan untuk melakukan **deteksi penyakit pada daun kakao** menggunakan metode **Object Detection berbasis Transformer / Deep Learning**. Model digunakan untuk mengklasifikasikan daun kakao menjadi dua kategori:
- Daun sehat (healthy)
- Daun terinfeksi CSSVD (Cocoa Swollen Shoot Virus Disease)

---

## Dataset

- **Nama Dataset**: CSSVD (Cocoa Swollen Shoot Virus Disease Dataset)
- **Sumber Dataset**: Roboflow Universe  
- **Link Dataset**: https://universe.roboflow.com/cocoa-o3obu/cssvd
- **Jumlah Gambar**: ±9.600 images :contentReference[oaicite:0]{index=0}
- **Jenis Dataset**: Object Detection (Bounding Box)
- **Jumlah Kelas**:
  - `healthy-cocoa-leaf`
  - `cocoa-swollen-shoot-virus-leaf`

- **Lisensi**: CC BY 4.0 :contentReference[oaicite:1]{index=1}

---

## Model yang Digunakan

Model yang digunakan dalam project ini adalah:
- **Transformer-based Model** (contoh: DETR / Vision Transformer / YOLO berbasis transformer jika kamu pakai YOLOv8 bisa tulis itu)

Contoh penulisan:
- Model: YOLOv8 (Ultralytics)
- Arsitektur: Deep Learning (Object Detection)

---

## Tools & Teknologi
- Python
- Roboflow (Dataset Management & Annotation)
- PyTorch / Ultralytics
- OpenCV

---

## Tujuan Project
- Mendeteksi penyakit pada daun kakao secara otomatis
- Membantu proses monitoring kesehatan tanaman kakao
- Mendukung implementasi Smart Agriculture

---

## Output yang Diharapkan
- Bounding box pada daun kakao
- Klasifikasi kondisi daun (sehat / terinfeksi)
- Evaluasi model (mAP, accuracy, dll)

---

## Noted
Dataset ini diambil dari Roboflow Universe dan dapat digunakan untuk keperluan penelitian maupun pembelajaran dengan tetap mengikuti lisensi yang berlaku.

---

## Sitasi Dataset

```bibtex
@misc{
  cssvd_dataset,
  title = { cssvd Dataset },
  author = { cocoa },
  howpublished = { https://universe.roboflow.com/cocoa-o3obu/cssvd },
  year = { 2024 }
}
