
---

# Tugas Tim

---

## 1. Pipeline

Alur berjalan secara linier dengan feedback loop:

**Roboflow Dataset → Data Ingestion → EDA & Augmentasi → Model Training (YOLOv8) → Evaluasi → Deployment API → Frontend/Mobile → Monitoring → (feedback ke training)**

---

## 2. Breakdown Tim 7 Orang

---

### Role 1 — Data Engineer
**Tugas utama:** Menjamin semua data masuk bersih, terstruktur, dan siap diproses tim downstream.

**Detail pekerjaan:**
- Download dataset dari Roboflow via API (format YOLO `.txt`)
- Validasi integritas: cek jumlah pasangan `image ↔ label`, resolusi minimum, format bounding box
- Split dataset: 80/10/10 (train/val/test) secara stratified per kelas
- Bangun script pipeline ingestion yang reproducible

**Output deliverables:**
- `data/raw/` berisi dataset asli
- `data/processed/train|val|test/` berisi split siap pakai
- `reports/data_validation_report.md` (jumlah gambar per kelas, distribusi)

**Tools:** Python, `roboflow` library, `pandas`, `PIL`, `yaml`

```python
# Contoh: download via Roboflow API
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")
project = rf.workspace("workspace").project("cssvd")
dataset = project.version(1).download("yolov8")
```

---

### Role 2 — Data Scientist (EDA & Augmentasi)
**Tugas utama:** Pahami distribusi data secara visual dan rancang strategi augmentasi yang tepat.

**Detail pekerjaan:**
- EDA: distribusi kelas, ukuran bounding box, aspect ratio, kecerahan gambar
- Identifikasi class imbalance (healthy vs diseased)
- Desain augmentasi: flip, rotate, mosaic, HSV shift, blur
- Buat konfigurasi augmentasi dalam file `augmentation_config.yaml`

**Output deliverables:**
- `notebooks goole colab` (visualisasi distribusi)
- `configs/augmentation_config.yaml`
- `reports/eda_summary.md`

**Tools:** `matplotlib`, `seaborn`, `albumentations`, Jupyter

```python
# Contoh: cek distribusi kelas dari label YOLO
import os
from collections import Counter

label_dir = "data/processed/train/labels"
class_counts = Counter()
for f in os.listdir(label_dir):
    with open(os.path.join(label_dir, f)) as lf:
        for line in lf:
            class_counts[int(line.split()[0])] += 1
print(class_counts)  # {0: 4800, 1: 4800}
```

---

### Role 3 — ML Engineer (Model Development)
**Tugas utama:** Bangun dan konfigurasi model YOLOv8 sebagai baseline, eksplorasi model lain jika waktu memungkinkan.

**Detail pekerjaan:**
- Buat `dataset.yaml` sesuai format Ultralytics
- Konfigurasi YOLOv8n/s/m tergantung resource
- Fine-tune dari pretrained weights (`yolov8m.pt`)
- Tuning hyperparameter: `lr0`, `batch`, `imgsz`, `epochs`
- Simpan setiap eksperimen dengan nama yang descriptive

**Output deliverables:**
- `configs/dataset.yaml`
- `configs/train_config.yaml`
- `models/` berisi `.pt` weights per eksperimen

**Tools:** `ultralytics`, PyTorch, `yaml`

```python
from ultralytics import YOLO

model = YOLO("yolov8m.pt")  # pretrained backbone
model.train(
    data="configs/dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name="cssvd_yolov8m_v1",
    project="runs/train"
)
```

---

### Role 4 — MLOps Engineer (Experiment Tracking & CI/CD)
**Tugas utama:** Pastikan setiap eksperimen ter-track dengan baik dan pipeline training bisa direproduksi.

**Detail pekerjaan:**
- Setup MLflow atau Weights & Biases untuk logging metrics
- Buat training script dengan argument parsing (`argparse`)
- Versi model dengan naming convention: `model_v{version}_{date}`
- Setup Git workflow (branching strategy untuk tim)
- Buat `requirements.txt` dan `Dockerfile` (opsional)

**Output deliverables:**
- `scripts/train.py` (script training utama)
- `mlflow/` atau W&B dashboard
- `requirements.txt`, `README.md`
- Git branching guide

**Tools:** MLflow / W&B, Docker, Git, `argparse`

```python
import mlflow
mlflow.set_experiment("cssvd_detection")
with mlflow.start_run(run_name="yolov8m_lr001"):
    mlflow.log_param("model", "yolov8m")
    mlflow.log_param("lr", 0.001)
    mlflow.log_metric("mAP50", 0.87)
    mlflow.log_artifact("runs/train/cssvd_yolov8m_v1/weights/best.pt")
```

---

### Role 5 — Evaluation Engineer
**Tugas utama:** Evaluasi model secara komprehensif dan buat laporan performa yang mudah dibaca stakeholder.

**Detail pekerjaan:**
- Hitung Klasifikasi Precision, Recall, F1 per kelas
- Buat confusion matrix dan visualisasi PR curve
- Analisis false positive dan false negative (contoh gambar gagal deteksi)
- Bandingkan performa antar eksperimen

**Output deliverables:**
- `notebooks/02_evaluation.ipynb`
- `reports/evaluation_report.md` (tabel metrics semua run)
- `reports/figures/` (confusion matrix, PR curve)

**Tools:** `ultralytics`, `sklearn`, `matplotlib`

```python
from ultralytics import YOLO

model = YOLO("models/best.pt")
metrics = model.val(data="configs/dataset.yaml", split="test")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"Precision: {metrics.box.p.mean():.4f}")
print(f"Recall: {metrics.box.r.mean():.4f}")
```

---

### Role 6 — Backend / Deployment Engineer
**Tugas utama:** Bungkus model menjadi REST API yang bisa dikonsumsi frontend atau mobile.

**Detail pekerjaan:**
- Buat endpoint FastAPI: `POST /predict` menerima gambar, return bounding box + label + confidence
- Export model ke ONNX untuk inferensi lebih cepat
- Tambahkan input validation dan error handling
- Uji dengan Postman / `curl`
- Deploy ke server lokal / cloud (Heroku, Railway, atau VPS)

**Output deliverables:**
- `api/main.py` (FastAPI app)
- `api/model_inference.py`
- `models/best.onnx`
- API documentation (Swagger auto-generated)

**Tools:** FastAPI, Uvicorn, ONNX Runtime, Pydantic

```python
# api/main.py
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2, numpy as np

app = FastAPI()
model = YOLO("models/best.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    results = model(img)[0]
    detections = []
    for box in results.boxes:
        detections.append({
            "class": results.names[int(box.cls)],
            "confidence": float(box.conf),
            "bbox": box.xyxy[0].tolist()
        })
    return {"detections": detections}
```

---

### Role 7 — Frontend / Mobile Engineer
**Tugas utama:** Buat antarmuka pengguna untuk menggunakan model deteksi penyakit kakao.

**Detail pekerjaan:**
- Bangun web app sederhana (React / Streamlit) atau mobile (React Native)
- Integrasi ke endpoint API Role 6
- Tampilkan hasil deteksi dengan bounding box di atas gambar
- Untuk mobile: ekspor model ke ONNX lalu jalankan offline dengan ONNX Runtime Mobile

**Output deliverables:**
- `frontend/` (React app atau Streamlit app)
- Demo video / screenshot
- APK prototype (jika mobile, opsional)

**Tools:** Streamlit / React, `axios`, ONNX Runtime Mobile (React Native)

```python
# Opsi cepat: Streamlit app
import streamlit as st
import requests

st.title("CSSVD Leaf Detection")
uploaded = st.file_uploader("Upload gambar daun kakao", type=["jpg","png"])
if uploaded:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": uploaded.getvalue()}
    )
    st.json(response.json())
```

---

## Workflow Antar Tim

```
[Role 1] raw dataset + split
     ↓
[Role 2] EDA report + augmentation config
     ↓
[Role 3] trained model weights (.pt)
     ↓                    ↓
[Role 4]              [Role 5]
MLflow logs        evaluation report
     ↓
[Role 6] REST API (FastAPI)
     ↓
[Role 7] Web / Mobile UI
     ↑_____________feedback & issue report_____________|
```

Setiap Role mengambil output dari role sebelumnya melalui folder `data/`, `models/`, dan `reports/` yang dishare via Git.

---

## Struktur Project Folder (Jika Ingin Lebih Serius)

```
cssvd-detection/
├── data/
│   ├── raw/                   # dataset asli dari Roboflow
│   ├── processed/
│   │   ├── train/images|labels
│   │   ├── val/images|labels
│   │   └── test/images|labels
├── configs/
│   ├── dataset.yaml
│   ├── train_config.yaml
│   └── augmentation_config.yaml
├── models/
│   ├── best.pt                # best checkpoint
│   └── best.onnx              # untuk deployment
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── export_onnx.py
├── api/
│   ├── main.py                # FastAPI app
│   └── model_inference.py
├── frontend/
│   └── app.py                 # Streamlit / React
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_evaluation.ipynb
├── reports/
│   ├── eda_summary.md
│   ├── evaluation_report.md
│   └── figures/
├── mlruns/                    # MLflow artifacts
├── requirements.txt
├── Dockerfile
└── README.md
```

---

**Ringkasan pembagian kerja tim:**

| # | Role | Fokus | Tim |
|---|------|-------|------|
| 1 | Data Engineer | Ingestion, split, validasi | Isnaini |
| 2 | Data Scientist | EDA, augmentasi | Shalfa |
| 3 | ML Engineer | YOLOv8 training | Zain |
| 4 | MLOps Engineer | Tracking, versioning, Git | Ramdan Riyadh |
| 5 | Eval Engineer | mAP, metrics, laporan | Fayed |
| 6 | Backend Engineer | FastAPI, ONNX export | Bagas |
| 7 | Frontend Engineer | Web UI | Bagas |

Semoga membantu!

---

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
```

