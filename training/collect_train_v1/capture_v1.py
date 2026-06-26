import cv2
import mediapipe as mp
import numpy as np
import os
import time

# =========================
# CONFIG
# =========================
DATA_PATH = "dataset_v1"
sequence_length = 30
target_sequences = 100  # jumlah sequence yang dikumpulkan per label

labels = ['1', 'A']

# =========================
# MEDIAPIPE
# =========================
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=1,                 # maksimum tangan yang dideteksi
    min_detection_confidence=0.7,    # ambang deteksi awal
    min_tracking_confidence=0.7      # ambang pelacakan setelah terdeteksi
)

mp_draw = mp.solutions.drawing_utils  # utilitas menggambar landmark

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)  # membuka kamera default

# =========================
# CREATE FOLDER
# =========================
for label in labels:
    os.makedirs(
        os.path.join(DATA_PATH, label),  # membuat path folder dataset
        exist_ok=True                    # tidak error jika folder sudah ada
    )

# =========================
# VARIABLES
# =========================
current_idx = 0
current_label = labels[current_idx]
collecting = False
sequence = []
sequence_count = 0

print("\nKONTROL:")
print("N → next label")
print("P → previous label")
print("S → start")
print("Q → stop")
print("ESC → keluar")

# =========================
# MAIN LOOP
# =========================
while True:

    ret, frame = cap.read()            # mengambil satu frame dari kamera
    frame = cv2.flip(frame, 1)         # membalik frame secara horizontal (mirror)
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB              # konversi BGR ke RGB untuk MediaPipe
    )

    result = hands.process(rgb)        # menjalankan deteksi landmark tangan
    landmarks = []

    # =========================
    # DETECTION
    # =========================
    if result.multi_hand_landmarks:    # jika tangan berhasil dideteksi
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS   # menggambar landmark dan koneksinya
            )

            for lm in hand_landmarks.landmark:
                # menyimpan koordinat x,y,z setiap landmark
                landmarks.extend([
                    lm.x,
                    lm.y,
                    lm.z
                ])

        # =========================
        # VALIDATION + NORMALIZATION
        # =========================
        if len(landmarks) == 63:       # memastikan jumlah fitur sesuai
            landmarks = np.array(
                landmarks              # mengubah list menjadi array NumPy
            )
            # normalisasi terhadap landmark pertama (wrist)
            landmarks = landmarks - landmarks[0]

            # =========================
            # COLLECT
            # =========================
            if collecting:
                sequence.append(
                    landmarks          # menambahkan satu frame ke sequence
                )

    else:
        # jika tangan hilang, sequence dihapus
        sequence = []

    # =========================
    # SAVE SEQUENCE
    # =========================
    if len(sequence) == sequence_length:
        np.save(
            os.path.join(
                DATA_PATH,
                current_label,
                f"{time.time()}"       # nama file berdasarkan timestamp
            ),
            sequence                   # menyimpan sequence ke file .npy
        )

        sequence = []
        sequence_count += 1
        time.sleep(0.3)                # jeda agar data tidak terlalu mirip

    # =========================
    # DISPLAY
    # =========================
    cv2.putText(
        frame,
        f"Label: {current_label}",
        (10,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )                                  # menampilkan label gesture

    cv2.putText(
        frame,
        f"Collected: {sequence_count}/{target_sequences}",
        (10,70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,0,0),
        2
    )                                  # menampilkan jumlah data yang telah dikumpulkan

    status = "COLLECTING" if collecting else "IDLE"

    cv2.putText(
        frame,
        status,
        (10,110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,255),
        2
    )                                  # menampilkan status pengambilan data

    cv2.imshow(
        "Capture",
        frame
    )                                  # menampilkan hasil kamera
    key = cv2.waitKey(10) & 0xFF        # membaca input keyboard setiap 10 ms

    # =========================
    # NEXT LABEL
    # =========================
    if key == ord('n'):                # berpindah ke label berikutnya
        current_idx = (current_idx + 1) % len(labels)
        current_label = labels[current_idx]
        sequence_count = 0
        sequence = []
        print(f"Pilih: {current_label}")

    # =========================
    # PREVIOUS LABEL
    # =========================
    if key == ord('p'):                # kembali ke label sebelumnya
        current_idx = (current_idx - 1) % len(labels)
        current_label = labels[current_idx]
        sequence_count = 0
        sequence = []
        print(f"Pilih: {current_label}")

    # =========================
    # START
    # =========================
    if key == ord('s'):
        collecting = True              # mulai proses pengumpulan data
        sequence = []
        print("START")

    # =========================
    # STOP
    # =========================
    if key == ord('q'):
        collecting = False             # menghentikan proses pengumpulan
        sequence = []
        print("STOP")

    # =========================
    # AUTO STOP
    # =========================
    if sequence_count >= target_sequences:
        collecting = False             # berhenti otomatis jika target tercapai
        print(f"Selesai untuk {current_label}")

    # =========================
    # EXIT
    # =========================
    if key == 27:                      # tombol ESC
        break

# ========================
# RELEASE RESOURCE
# =========================
cap.release()              # melepaskan kamera

cv2.destroyAllWindows()    # menutup seluruh jendela OpenCV