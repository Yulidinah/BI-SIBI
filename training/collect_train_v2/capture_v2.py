import cv2
import mediapipe as mp
import numpy as np
import os
import time

# =========================
# CONFIG
# =========================
DATA_PATH = "dataset_v2"

sequence_length = 40
target_sequences = 100

# Toleransi frame tangan hilang (untuk gerakan tepuk, dll)
MAX_MISSING_FRAMES = 8

# Sliding window step (supaya gerakan berkelanjutan lebih banyak ditangkap)
STEP_SIZE = 10

labels = [
    "Tes",
]

# =========================
# CREATE DATASET FOLDER
# =========================
for label in labels:
    os.makedirs(os.path.join(DATA_PATH, label), exist_ok=True)

# =========================
# CLAHE untuk preprocessing (bantu deteksi punggung tangan & cahaya redup)
# =========================
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def preprocess_frame(frame):
    """Tingkatkan kontras frame sebelum dikirim ke MediaPipe."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    return enhanced

# =========================
# NORMALISASI LANDMARK
# =========================
def normalize_hand(hand_data):
    """
    Normalisasi berbasis wrist + skala jarak ke MCP jari tengah.
    Lebih stabil dibanding hanya geser ke origin.
    """
    # Geser ke origin (wrist = titik 0)
    hand_data = hand_data - hand_data[0:3].repeat(21)

    # Scale: jarak wrist (idx 0) ke MCP jari tengah (idx 9)
    wrist  = hand_data[0:3]
    mid_mcp = hand_data[9*3 : 9*3+3]
    scale  = np.linalg.norm(mid_mcp - wrist)

    if scale > 1e-6:
        hand_data = hand_data / scale

    return hand_data

# =========================
# MEDIAPIPE
# Threshold diturunkan supaya:
# - Punggung tangan lebih mudah terdeteksi
# - Tangan kiri lebih responsif
# - Landmark lebih cepat muncul
# =========================
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    max_num_hands=2,
    min_detection_confidence=0.3,   # turun dari 0.5
    min_tracking_confidence=0.3     # turun dari 0.5
)

mp_draw = mp.solutions.drawing_utils

# Gaya landmark kustom (lebih jelas di layar)
landmark_style = mp_draw.DrawingSpec(color=(0, 255, 128), thickness=2, circle_radius=3)
connection_style = mp_draw.DrawingSpec(color=(255, 255, 0), thickness=2)

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)

# =========================
# STATE
# =========================
current_idx      = 0
current_label    = labels[current_idx]
collecting       = False
sequence         = []
sequence_count   = 0
missing_frame_count = 0
last_landmarks   = np.zeros(126)

# =========================
# INFO
# =========================
print("\n==========================")
print("SIGN LANGUAGE COLLECTOR v3")
print("==========================")
print("S -> START")
print("Q -> STOP")
print("N -> NEXT LABEL")
print("P -> PREVIOUS LABEL")
print("ESC -> EXIT")
print("==========================\n")

# =========================
# MAIN LOOP
# =========================
while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # --- Preprocessing ---
    frame_processed = preprocess_frame(frame)
    rgb = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # =========================
    # DEFAULT FEATURES
    # =========================
    left_hand  = np.zeros(63)
    right_hand = np.zeros(63)
    hand_detected = False

    # =========================
    # DETECT HANDS
    # =========================
    if result.multi_hand_landmarks and result.multi_handedness:

        hand_detected = True

        for hand_landmarks, handedness in zip(
            result.multi_hand_landmarks,
            result.multi_handedness
        ):
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_style,
                connection_style
            )

            # Ekstrak landmark (x, y, z) × 21 titik = 63 nilai
            hand_data = np.array([
                val
                for lm in hand_landmarks.landmark
                for val in (lm.x, lm.y, lm.z)
            ])

            hand_data = normalize_hand(hand_data)

            hand_label = handedness.classification[0].label
            print("HAND:", hand_label)
            if hand_label == "Left":
                left_hand = hand_data
            else:
                right_hand = hand_data
                
            
    # =========================
    # GABUNGKAN KEDUA TANGAN
    # =========================
    landmarks = np.concatenate([left_hand, right_hand])

    # =========================
    # COLLECT DENGAN TOLERANSI HILANG
    # =========================
    if collecting:

        if hand_detected:
            sequence.append(landmarks)
            last_landmarks = landmarks.copy()
            missing_frame_count = 0

        else:
            # Tangan hilang sementara → interpolasi pakai frame terakhir
            if missing_frame_count < MAX_MISSING_FRAMES:
                sequence.append(last_landmarks)
                missing_frame_count += 1
            else:
                # Hilang terlalu lama → reset sequence
                if len(sequence) > 0:
                    print(f"  [RESET] Tangan hilang >  {MAX_MISSING_FRAMES} frame")
                sequence = []
                missing_frame_count = 0

    # =========================
    # SAVE DENGAN SLIDING WINDOW
    # Satu gerakan bisa menghasilkan beberapa sample sekaligus
    # =========================
    if len(sequence) >= sequence_length:

        save_data = np.array(sequence[-sequence_length:])  # ambil 40 frame terakhir

        filename = str(time.time())
        np.save(
            os.path.join(DATA_PATH, current_label, filename),
            save_data
        )

        sequence_count += 1
        print(f"[{current_label}] {sequence_count}/{target_sequences}")

        # Geser window, tidak reset penuh
        sequence = sequence[STEP_SIZE:]

        time.sleep(0.1)  # delay lebih pendek

    # =========================
    # DISPLAY UI
    # =========================
    status      = "COLLECTING" if collecting else "IDLE"
    buf_len     = len(sequence)
    missing_str = f" (missing: {missing_frame_count})" if missing_frame_count > 0 else ""

    cv2.rectangle(frame, (0, 0), (680, 200), (0, 0, 0), -1)

    cv2.putText(frame, f"Label    : {current_label}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Collected: {sequence_count}/{target_sequences}",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    cv2.putText(frame, f"Status   : {status}  Buffer: {buf_len}/{sequence_length}{missing_str}",
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

    cv2.putText(frame, f"Hand     : {'DETECTED' if hand_detected else 'NOT FOUND'}",
                (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0) if hand_detected else (0, 0, 255), 2)

    cv2.putText(frame, "S=Start  Q=Stop  N/P=Change  ESC=Exit",
                (10, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    cv2.imshow("Sign Language Collector v3", frame)

    key = cv2.waitKey(10) & 0xFF

    if key == ord('s'):
        collecting = True
        sequence = []
        missing_frame_count = 0
        print(f"\nSTART collecting [{current_label}]")

    elif key == ord('q'):
        collecting = False
        sequence = []
        missing_frame_count = 0
        print("\nSTOP collecting")

    elif key == ord('n'):
        current_idx   = (current_idx + 1) % len(labels)
        current_label = labels[current_idx]
        sequence = []
        sequence_count = 0
        missing_frame_count = 0
        print(f"\nLabel → {current_label}")

    elif key == ord('p'):
        current_idx   = (current_idx - 1) % len(labels)
        current_label = labels[current_idx]
        sequence = []
        sequence_count = 0
        missing_frame_count = 0
        print(f"\nLabel → {current_label}")

    if sequence_count >= target_sequences:
        collecting = False
        print(f"\nDONE collecting [{current_label}]")

    if key == 27:
        break

# =========================
# RELEASE
# =========================
cap.release()
cv2.destroyAllWindows()