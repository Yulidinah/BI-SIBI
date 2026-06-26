import cv2
import mediapipe as mp
import numpy as np
import threading

mp_hands = mp.solutions.hands

# CLAHE untuk meningkatkan kontras — membantu deteksi di cahaya redup
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Instance MediaPipe permanen + lock untuk thread safety
_hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    max_num_hands=2,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)
_lock = threading.Lock()


def preprocess_frame(frame):
    """Tingkatkan kontras frame menggunakan CLAHE sebelum dikirim ke MediaPipe."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def normalize_v1(hand_data):
    """
    Normalisasi sesuai webcam Python:
    1. Reshape ke (21, 3)
    2. Kurangi wrist (titik 0) sebagai origin
    3. Bagi dengan max absolute value untuk scaling
    """
    data = hand_data.reshape(21, 3)
    wrist = data[0]
    data = data - wrist
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val
    return data.flatten()

def normalize_v2(hand_data):
    """
    Normalisasi untuk model V2 (kata 2 tangan).
    Sesuai dengan pipeline saat collect dataset_v2:
    - Geser ke origin (wrist = titik 0)
    - Scaling berdasarkan jarak wrist ke MCP jari tengah
    """
    hand_data = hand_data - hand_data[0:3].repeat(21)
    wrist   = hand_data[0:3]
    mid_mcp = hand_data[9*3 : 9*3+3]
    scale   = np.linalg.norm(mid_mcp - wrist)
    if scale > 1e-6:
        hand_data = hand_data / scale
    return hand_data


def extract_landmarks(image):
    """
    Ekstrak landmark tangan dari frame gambar.
    Mengembalikan dua set landmark:
    - landmarks_v1: (63,) untuk model V1 — hanya tangan kanan, normalisasi Opsi A
    - landmarks_v2: (126,) untuk model V2 — dua tangan, normalisasi Opsi B
    """
    result_data = {
        "detected":       False,
        "left_active":    False,
        "right_active":   False,
        "landmarks_v1":   [0.0] * 63,   # 1 tangan × 21 × 3
        "landmarks_v2":   [0.0] * 126,  # 2 tangan × 21 × 3
    }

    left_raw  = np.zeros(63)
    right_raw = np.zeros(63)

    # Preprocessing CLAHE
    image_enhanced = preprocess_frame(image)
    image_rgb = cv2.cvtColor(image_enhanced, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False

    with _lock:
        results = _hands.process(image_rgb)

    detected_count = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
    print(f"[MP] hands={detected_count}")
    
    image_rgb.flags.writeable = True

    if results.multi_hand_landmarks and results.multi_handedness:
        result_data["detected"] = True

        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            hand_type = handedness.classification[0].label

            # Ekstrak koordinat raw (x, y, z) × 21 titik = 63 nilai
            coords_raw = np.array([
                val
                for lm in hand_landmarks.landmark
                for val in (lm.x, lm.y, lm.z)
            ])

            if hand_type == "Left":
                left_raw = coords_raw
                result_data["left_active"] = True
            elif hand_type == "Right":
                right_raw = coords_raw
                result_data["right_active"] = True

        # ── Landmarks V1 ──────────────────────────────────────────────
        # Hanya tangan kanan, normalisasi Opsi A (sesuai auto_capture)
        if result_data["right_active"]:
            norm = normalize_v1(right_raw)
            result_data["landmarks_v1"] = normalize_v1(right_raw).tolist()

        # ── Landmarks V2 ──────────────────────────────────────────────
        # Dua tangan, normalisasi Opsi B (sesuai capture_v3)
        left_v2  = normalize_v2(left_raw)  if result_data["left_active"]  else np.zeros(63)
        right_v2 = normalize_v2(right_raw) if result_data["right_active"] else np.zeros(63)
        result_data["landmarks_v2"] = np.concatenate([left_v2, right_v2]).tolist()

    return result_data