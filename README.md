#  Bahasa Isyarat SIBI — Real-Time Sign Language Translator

Sistem penerjemah Bahasa Isyarat Indonesia (SIBI) secara real-time menggunakan deep learning BiLSTM dan MediaPipe, dijalankan melalui antarmuka web.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.11x-009688?logo=fastapi&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-orange)

---

## Tentang Proyek

Proyek ini dikembangkan sebagai bagian dari mata kuliah **Praktik Computer Vision** di Politeknik Negeri Lhokseumawe. Sistem ini mampu mendeteksi dan menerjemahkan gestur tangan SIBI menjadi teks secara langsung melalui webcam browser.

Dataset dikumpulkan **secara mandiri** menggunakan tangan peraga tunggal, mencakup 63 kelas gestur:
- **A–Z** (26 huruf)
- **1–10** (10 angka)
- **5 kata dasar** (Hallo, Maaf, Permisi, Selamat Jalan, Terima Kasih) — 1 tangan
- **22 kata komunikasi** (Saya, Kamu, Dia, Mereka, Nama, Siapa, Apa, Kapan, Di mana, dll.) — 2 tangan

---

## Arsitektur Sistem

```
Webcam (Browser)
    │
    │  JPEG Base64 via HTTP POST (setiap 100ms)
    ▼
FastAPI Backend
    │
    ├── CLAHE Preprocessing (peningkatan kontras)
    ├── MediaPipe Hands (ekstraksi 21 landmark/tangan)
    ├── Normalisasi Landmark
    │     ├── V1: wrist origin (dataset huruf/angka)
    │     └── V2: wrist origin + MCP scaling (dataset kata)
    │
    ├── Buffer Sekuens
    │     ├── V1: 30 frame × 63 fitur (1 tangan)
    │     └── V2: 40 frame × 126 fitur (2 tangan)
    │
    ├── BiLSTM Inference
    │     ├── Model V1 → 41 kelas (huruf + angka + 5 kata)
    │     └── Model V2 → 22 kelas (kata 2 tangan)
    │
    └── Softmax + Confidence Threshold (≥0.70) + Majority Voting (7 frame)
            │
            ▼
       Hasil Teks → Tampil di UI
```

---

## Teknologi

| Komponen | Teknologi |
|---|---|
| Backend | FastAPI, Python 3.10 |
| Deep Learning | PyTorch, BiLSTM |
| Hand Tracking | MediaPipe Hands |
| Image Processing | OpenCV, CLAHE |
| Frontend | HTML, CSS, JavaScript (Vanilla) |
| Dataset | Custom-collected (self-recorded) |

---

## Struktur Proyek

```
sibi-project/
├── backend/
│   ├── main.py                  # FastAPI server + inferensi
│   └── mediapipe_processor.py   # Ekstraksi & normalisasi landmark
├── frontend/
│   ├── index.html               # UI utama
│   ├── style.css                # Styling
│   └── app.js                   # Logika frontend
├── models/
│   ├── best_model_v1.pth        # Model huruf + angka + 5 kata
│   ├── label_map_v1.npy         # Label map V1 (41 kelas)
│   ├── best_model_v2.pth        # Model kata 2 tangan
│   └── label_map_v2.npy         # Label map V2 (22 kelas)
├── dataset_v1/                  # Dataset huruf, angka, kata 1 tangan
├── dataset_v2/                  # Dataset kata 2 tangan
├── train_v1.py                  # Script training model V1
├── train_v2.py                  # Script training model V2
└── README.md
```

---

## Cara Menjalankan

### 1. Install dependensi

```bash
pip install fastapi uvicorn torch mediapipe opencv-python numpy
```

### 2. Jalankan backend

```bash
cd backend
uvicorn main:app --workers 1
```

### 3. Jalankan frontend

```bash
cd frontend
python -m http.server 3000
```

Buka browser di `http://localhost:3000`

---

## Cara Penggunaan

1. Buka web di browser
2. Izinkan akses kamera
3. Klik **Mulai Deteksi**
4. Pilih mode:
   - **Otomatis** — sistem mendeteksi otomatis berdasarkan jumlah tangan
   - **Huruf & Angka** — paksa ke model V1
   - **Kata** — paksa ke model V2
5. Tunjukkan gestur SIBI ke kamera
6. Hasil muncul di panel **Hasil Deteksi** dan terakumulasi di **Kalimat Terbentuk**

---

## Detail Model

### Model V1 — Huruf, Angka, dan Kata 1 Tangan
- **Input:** sekuens 30 frame × 63 fitur
- **Arsitektur:** BiLSTM (hidden=128, layers=2, bidirectional) → FC(256→128→41)
- **Kelas:** 41 (A-Z, 1-10, Hallo, Maaf, Permisi, Selamat_Jalan, Terima_kasih)
- **Dataset:** 100 sekuens/kelas, sliding window step 10

### Model V2 — Kata 2 Tangan
- **Input:** sekuens 40 frame × 126 fitur
- **Arsitektur:** BiLSTM (hidden=128, layers=2, bidirectional) → FC(256→128→64→22)
- **Kelas:** 22 kata komunikasi dasar
- **Dataset:** 100 sekuens/kelas, toleransi 8 frame hilang

---

## Pengembang

**Yuli Mulia Dinah**  
Teknik Informatika — Politeknik Negeri Lhokseumawe  

---

## Lisensi

Proyek ini dibuat untuk keperluan akademik. Dataset dikumpulkan secara mandiri dan bukan merupakan dataset publik.
