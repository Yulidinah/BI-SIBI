import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from collections import Counter

# ── Config ──────────────────────────────────────────
DATA_PATH       = "datasets/dataset_v1"
SEQUENCE_LENGTH = 30
FEATURE_SIZE    = 63
EPOCHS          = 60
BATCH_SIZE      = 32
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Gunakan GPU jika tersedia

# ── Load Data ────────────────────────────────────────
X, y = [], []

for label in os.listdir(DATA_PATH): # Ambil seluruh folder label
    label_path = os.path.join(DATA_PATH, label) # Gabungkan path folder
    if not os.path.isdir(label_path): continue # Lewati jika bukan folder
    for file in os.listdir(label_path): # Baca seluruh file pada folder
        if not file.endswith(".npy"): continue # Hanya file .npy
        data = np.load(os.path.join(label_path, file)) # Muat data sequence
        if data.shape != (SEQUENCE_LENGTH, FEATURE_SIZE): continue # Validasi ukuran data
        X.append(data) # Simpan fitur
        y.append(label) # Simpan label

X = np.array(X, dtype=np.float32) # Ubah menjadi array NumPy
y = np.array(y)

print(f"Dataset: {X.shape} | Kelas: {len(set(y))}")
print(f"Distribusi: {Counter(y)}") # Tampilkan jumlah data tiap kelas

# ── Label Encoder ────────────────────────────────────
"""
Karena model neural network hanya dapat memproses data numerik,
maka label gesture yang masih berupa teks diubah menjadi bilangan bulat
menggunakan LabelEncoder
"""
le = LabelEncoder()
y_encoded = le.fit_transform(y) # Encode label menjadi angka
np.save("models/label_map_v1.npy", le.classes_) # Simpan mapping label
print(f"Kelas: {le.classes_}")

# ── Split ────────────────────────────────────────────
X, y_encoded = shuffle(X, y_encoded, random_state=42) # Acak urutan dataset
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42 # Bagi data train dan validasi
)

X_train = torch.tensor(X_train, dtype=torch.float32) # Ubah ke tensor
X_val   = torch.tensor(X_val,   dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_val   = torch.tensor(y_val,   dtype=torch.long)

# ── Model ────────────────────────────────────────────
class GestureBiLSTM(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, # Jumlah fitur input
            hidden_size=128, # Jumlah neuron LSTM
            num_layers=2, # Dua layer LSTM
            batch_first=True, # Format input (batch, seq, fitur)
            bidirectional=True, # LSTM dua arah
            dropout=0.3 # Dropout antar layer LSTM
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4), # Fully Connected + ReLU + Dropout
            nn.Linear(128, 64),  nn.ReLU(), nn.Dropout(0.3), # Fully Connected + ReLU + Dropout
            nn.Linear(64, num_classes) # Output sesuai jumlah kelas
        )

    def forward(self, x):
        out, _ = self.lstm(x) # Forward melalui BiLSTM
        return self.classifier(out[:, -1, :]) # Gunakan output frame terakhir

model = GestureBiLSTM(
    input_size=FEATURE_SIZE,
    num_classes=len(le.classes_)
).to(DEVICE) # Pindahkan model ke CPU/GPU

criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # Fungsi loss klasifikasi
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4) # Optimizer AdamW
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=5, verbose=True # Turunkan learning rate jika stagnan
)

# ── Training ─────────────────────────────────────────
best_val_acc = 0.0
os.makedirs("models", exist_ok=True) # Pastikan folder model tersedia

for epoch in range(EPOCHS): # Ulangi proses training sebanyak epoch
    model.train() # Aktifkan mode training
    perm = torch.randperm(X_train.size(0)) # Acak urutan data
    total_loss, correct, total = 0, 0, 0

    for i in range(0, X_train.size(0), BATCH_SIZE): # Training per mini-batch
        idx = perm[i:i+BATCH_SIZE] # Ambil indeks batch
        xb  = X_train[idx].to(DEVICE) # Pindahkan fitur ke device
        yb  = y_train[idx].to(DEVICE) # Pindahkan label ke device

        optimizer.zero_grad() # Reset gradien
        out  = model(xb) # Forward propagation
        loss = criterion(out, yb) # Hitung loss
        loss.backward() # Backpropagation
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Batasi gradien
        optimizer.step() # Update bobot

        total_loss += loss.item() # Akumulasi loss
        correct    += (out.argmax(1) == yb).sum().item() # Hitung prediksi benar
        total      += len(yb) # Total data

    train_acc = correct / total # Hitung akurasi training

    model.eval() # Mode evaluasi
    with torch.no_grad(): # Nonaktifkan perhitungan gradien
        out_val  = model(X_val.to(DEVICE)) # Prediksi data validasi
        val_acc  = (out_val.argmax(1) == y_val.to(DEVICE)).float().mean().item() # Hitung akurasi validasi

    scheduler.step(val_acc) # Update learning rate
    print(f"Epoch {epoch+1:03d} | Train {train_acc:.4f} | Val {val_acc:.4f}")

    if val_acc > best_val_acc: # Simpan jika performa meningkat
        best_val_acc = val_acc
        torch.save(model.state_dict(), "models/best_model_v1.pth") # Simpan bobot model
        print(f"  ✅ Saved (val={val_acc:.4f})")

print(f"\nBest Val Acc: {best_val_acc:.4f}")

# ── Classification Report ────────────────────────────
from sklearn.metrics import classification_report
model.load_state_dict(torch.load("models/best_model_v1.pth", map_location=DEVICE)) # Muat model terbaik
model.eval() # Mode evaluasi
with torch.no_grad(): # Tanpa menghitung gradien
    preds = model(X_val.to(DEVICE)).argmax(1).cpu().numpy() # Prediksi data validasi

print("\n" + classification_report(
    y_val.numpy(), preds,
    target_names=le.classes_ # Tampilkan precision, recall, f1-score
))