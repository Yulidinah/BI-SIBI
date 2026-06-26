import os
import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import shuffle

# ==================================
# CONFIG
# ==================================
DATA_PATH = "dataset_v2"

SEQUENCE_LENGTH = 40
FEATURE_SIZE = 126

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
) # Gunakan GPU jika tersedia

# ==================================
# LOAD DATA
# ==================================
X = []
y = []

for label in os.listdir(DATA_PATH): # Ambil seluruh folder label

    label_path = os.path.join(DATA_PATH, label) # Gabungkan path folder

    if not os.path.isdir(label_path): # Lewati jika bukan folder
        continue

    for file in os.listdir(label_path): # Baca seluruh file

        if not file.endswith(".npy"): # Hanya file .npy
            continue

        file_path = os.path.join(label_path, file)

        try:
            data = np.load(file_path) # Muat data sequence

            if data.shape != (SEQUENCE_LENGTH, FEATURE_SIZE): # Validasi ukuran data
                continue

            X.append(data) # Simpan fitur
            y.append(label) # Simpan label

        except:
            continue # Lewati file yang rusak

X = np.array(X) # Ubah ke array NumPy
y = np.array(y)

print("\nDataset Loaded")
print("X:", X.shape)
print("y:", y.shape)

# ==================================
# LABEL ENCODER
# ==================================
le = LabelEncoder()

y_encoded = le.fit_transform(y) # Ubah label menjadi angka

np.save(
    "label_map_v2.npy",
    le.classes_ # Simpan mapping label
)

# ==================================
# SHUFFLE
# ==================================
X, y_encoded = shuffle(
    X,
    y_encoded,
    random_state=42 # Acak urutan dataset
)

# ==================================
# SPLIT
# ==================================
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    stratify=y_encoded, # Menjaga proporsi tiap kelas
    random_state=42
)

# ==================================
# TENSOR
# ==================================
X_train = torch.tensor(
    X_train,
    dtype=torch.float32 # Ubah fitur menjadi tensor
)

X_val = torch.tensor(
    X_val,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.long # Ubah label menjadi tensor integer
)

y_val = torch.tensor(
    y_val,
    dtype=torch.long
)

# ==================================
# MODEL
# ==================================
class GestureBiLSTM(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=126, # Jumlah fitur input
            hidden_size=128, # Jumlah neuron LSTM
            num_layers=2, # Dua layer LSTM
            batch_first=True, # Format input (batch, seq, fitur)
            bidirectional=True, # LSTM dua arah
            dropout=0.3 # Dropout antar layer
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 128), # Fully Connected
            nn.ReLU(), # Aktivasi ReLU
            nn.Dropout(0.4), # Mengurangi overfitting

            nn.Linear(128, 64), # Fully Connected
            nn.ReLU(), # Aktivasi ReLU
            nn.Dropout(0.3), # Mengurangi overfitting

            nn.Linear(64, num_classes) # Layer output
        )

    def forward(self, x):

        out, _ = self.lstm(x) # Forward melalui BiLSTM

        out = out[:, -1, :] # Ambil output frame terakhir

        return self.classifier(out) # Klasifikasi gesture

# ==================================
# MODEL INIT
# ==================================
model = GestureBiLSTM(
    num_classes=len(le.classes_)
).to(DEVICE) # Pindahkan model ke CPU/GPU

# ==================================
# LOSS
# ==================================
criterion = nn.CrossEntropyLoss(
    label_smoothing=0.1 # Cross Entropy dengan label smoothing
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0005,
    weight_decay=1e-4 # Regularisasi bobot
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=5 # Turunkan learning rate jika stagnan
)

# ==================================
# TRAIN
# ==================================
EPOCHS = 50
BATCH_SIZE = 32

best_val_acc = 0

for epoch in range(EPOCHS): # Ulangi proses training

    model.train() # Aktifkan mode training

    permutation = torch.randperm(
        X_train.size(0) # Acak urutan data
    )

    train_correct = 0
    train_total = 0
    train_loss = 0

    for i in range(
        0,
        X_train.size(0),
        BATCH_SIZE # Training per mini-batch
    ):

        idx = permutation[i:i+BATCH_SIZE] # Ambil indeks batch

        xb = X_train[idx].to(DEVICE) # Pindahkan fitur ke device
        yb = y_train[idx].to(DEVICE) # Pindahkan label ke device

        optimizer.zero_grad() # Reset gradien

        outputs = model(xb) # Forward propagation

        loss = criterion(
            outputs,
            yb # Hitung loss
        )

        loss.backward() # Backpropagation

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0 # Batasi gradien maksimum
        )

        optimizer.step() # Update bobot

        train_loss += loss.item() # Akumulasi loss

        pred = outputs.argmax(1) # Ambil kelas dengan skor tertinggi

        train_correct += (
            pred == yb
        ).sum().item() # Hitung prediksi benar

        train_total += len(yb) # Total data

    train_acc = (
        train_correct /
        train_total # Hitung akurasi training
    )

    # =====================
    # VALIDATION
    # =====================
    model.eval() # Aktifkan mode evaluasi

    with torch.no_grad(): # Nonaktifkan gradien

        outputs = model(
            X_val.to(DEVICE) # Prediksi data validasi
        )

        pred = outputs.argmax(1) # Ambil hasil prediksi

        val_acc = (
            (pred ==
            y_val.to(DEVICE))
            .sum()
            .item()
            / len(y_val) # Hitung akurasi validasi
        )

    scheduler.step(val_acc) # Update learning rate

    print(
        f"Epoch {epoch+1:03d}"
        f" | Train {train_acc:.4f}"
        f" | Val {val_acc:.4f}"
    )

    if val_acc > best_val_acc: # Simpan model terbaik

        best_val_acc = val_acc

        torch.save(
            model.state_dict(),
            "best_model_v2.pth" # Simpan bobot model
        )

        print(
            f"Saved Best Model "
            f"({val_acc:.4f})"
        )

print("\nTraining Finished")
print("Best Val Acc:", best_val_acc)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# =====================
# LOAD BEST MODEL
# =====================
model.load_state_dict(
    torch.load(
        "best_model_v2.pth",
        map_location=DEVICE # Muat model terbaik
    )
)

model.eval() # Mode evaluasi

with torch.no_grad(): # Nonaktifkan gradien

    outputs = model(
        X_val.to(DEVICE) # Prediksi data validasi
    )

    preds = outputs.argmax(1).cpu().numpy() # Ambil hasil prediksi

y_true = y_val.cpu().numpy()

# =====================
# CLASSIFICATION REPORT
# =====================
print("\n")
print("=" * 80)
print("CLASSIFICATION REPORT")
print("=" * 80)

print(
    classification_report(
        y_true,
        preds,
        target_names=le.classes_,
        digits=4 # Tampilkan precision, recall, dan F1-score
    )
)

# =====================
# CONFUSION MATRIX
# =====================
cm = confusion_matrix(
    y_true,
    preds # Hitung confusion matrix
)

print("\n")
print("=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

print(cm)

# =====================
# PLOT
# =====================
plt.figure(figsize=(16, 14)) # Buat ukuran gambar

sns.heatmap(
    cm,
    annot=True, # Tampilkan nilai pada sel
    fmt="d", # Format bilangan bulat
    cmap="Blues", # Gunakan warna biru
    xticklabels=le.classes_,
    yticklabels=le.classes_
)

plt.title("Confusion Matrix") # Judul grafik
plt.xlabel("Predicted") # Label sumbu X
plt.ylabel("Actual") # Label sumbu Y

plt.xticks(rotation=45, ha="right") # Putar label sumbu X
plt.yticks(rotation=0) # Label sumbu Y tetap horizontal

plt.tight_layout() # Atur tata letak otomatis
plt.show() # Tampilkan grafik