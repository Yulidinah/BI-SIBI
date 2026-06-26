import os
import base64
import cv2
import numpy as np
import torch
import torch.nn as nn
import time
from collections import deque, Counter
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from mediapipe_processor import extract_landmarks

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════
# ARSITEKTUR MODEL V1 — huruf + angka + 5 kata 1 tangan
# Input: (batch, 30, 63)
# ══════════════════════════════════════════════════════
class GestureBiLSTM_V1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=63, hidden_size=128, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.3
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = torch.mean(out, dim=1)  # mean pooling
        return self.fc(out)


# ══════════════════════════════════════════════════════
# ARSITEKTUR MODEL V2 — kata 2 tangan
# Input: (batch, 40, 126)
# ══════════════════════════════════════════════════════
class GestureBiLSTM_V2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=126, hidden_size=128, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.3
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])


# ══════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(BASE_DIR), "models")
device     = torch.device("cpu")

try:
    raw_v1 = np.load(os.path.join(MODELS_DIR, "label_map_v1.npy"), allow_pickle=True)
    raw_v2 = np.load(os.path.join(MODELS_DIR, "label_map_v2.npy"), allow_pickle=True)
    label_map_v1 = {i: str(l) for i, l in enumerate(raw_v1)}
    label_map_v2 = {i: str(l) for i, l in enumerate(raw_v2)}

    model_v1 = GestureBiLSTM_V1(num_classes=len(label_map_v1)).to(device)
    model_v1.load_state_dict(torch.load(
        os.path.join(MODELS_DIR, "best_model_v1.pth"), map_location=device
    ))
    model_v1.eval()

    model_v2 = GestureBiLSTM_V2(num_classes=len(label_map_v2)).to(device)
    model_v2.load_state_dict(torch.load(
        os.path.join(MODELS_DIR, "best_model_v2.pth"), map_location=device
    ))
    model_v2.eval()

    print("\n" + "="*55)
    print(f"✅ V1 — {len(label_map_v1)} kelas: {list(label_map_v1.values())}")
    print(f"✅ V2 — {len(label_map_v2)} kelas: {list(label_map_v2.values())}")
    print("="*55 + "\n")

except Exception as e:
    print(f"\n❌ ERROR LOAD MODEL: {e}\n")


# ══════════════════════════════════════════════════════
# GLOBAL STATE
# ══════════════════════════════════════════════════════
buffer_v1         = []
buffer_v2         = []
cooldown_frames   = 0
missing_frames    = 0
was_detected      = False
last_landmarks_v1 = [0.0] * 63
last_landmarks_v2 = [0.0] * 126
last_left_active  = False
last_right_active = False

MAX_MISSING          = 10
CONFIDENCE_THRESHOLD = 0.70
SEQ_LEN_V1           = 30
SEQ_LEN_V2           = 40

pred_history_v1 = deque(maxlen=7)
pred_history_v2 = deque(maxlen=7)


class FrameRequest(BaseModel):
    image: str
    mode:  str = "auto"


# ══════════════════════════════════════════════════════
# PREDICT ENDPOINT
# ══════════════════════════════════════════════════════
@app.post("/predict")
def predict(req: FrameRequest):
    global buffer_v1, buffer_v2, cooldown_frames, missing_frames
    global was_detected, last_landmarks_v1, last_landmarks_v2
    global last_left_active, last_right_active

    start = time.time()

    try:
        # ── Decode frame ─────────────────────────────────────────────
        image_bytes = base64.b64decode(req.image.split(",")[-1])
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return {"detected": False, "prediction": "-"}

        # ── Ekstrak landmark ─────────────────────────────────────────
        result = extract_landmarks(image)

        # ── Cooldown ─────────────────────────────────────────────────
        if cooldown_frames > 0:
            cooldown_frames -= 1
            buffer_v1      = []
            buffer_v2      = []
            missing_frames = 0
            return {"detected": True, "prediction": "Menunggu Transisi Gerakan..."}

        # ── Update buffer ────────────────────────────────────────────
        if result["detected"]:
            if not was_detected:
                buffer_v1       = []
                buffer_v2       = []
                missing_frames  = 0
                pred_history_v1.clear()
                pred_history_v2.clear()

            was_detected      = True
            missing_frames    = 0
            last_landmarks_v1 = result["landmarks_v1"]
            last_landmarks_v2 = result["landmarks_v2"]
            last_left_active  = result["left_active"]
            last_right_active = result["right_active"]

            # Isi buffer sesuai mode
            if req.mode == "kata":
                buffer_v2.append(result["landmarks_v2"])
            elif req.mode == "huruf":
                buffer_v1.append(result["landmarks_v1"])
            else:
                if last_left_active and last_right_active:
                    buffer_v2.append(result["landmarks_v2"])
                else:
                    buffer_v1.append(result["landmarks_v1"])

        else:
            # Toleransi tangan hilang sesaat
            if missing_frames < MAX_MISSING and (len(buffer_v1) > 0 or len(buffer_v2) > 0):
                missing_frames += 1
                result["left_active"]  = last_left_active
                result["right_active"] = last_right_active
                result["detected"]     = True

                if req.mode == "kata":
                    buffer_v2.append(last_landmarks_v2)
                elif req.mode == "huruf":
                    buffer_v1.append(last_landmarks_v1)
                else:
                    if last_left_active and last_right_active:
                        buffer_v2.append(last_landmarks_v2)
                    else:
                        buffer_v1.append(last_landmarks_v1)
            else:
                buffer_v1         = []
                buffer_v2         = []
                missing_frames    = 0
                was_detected      = False
                last_left_active  = False
                last_right_active = False
                pred_history_v1.clear()
                pred_history_v2.clear()
                return {"detected": False, "prediction": "-"}

        # Batasi ukuran buffer
        if len(buffer_v1) > SEQ_LEN_V1:
            buffer_v1 = buffer_v1[-SEQ_LEN_V1:]
        if len(buffer_v2) > SEQ_LEN_V2:
            buffer_v2 = buffer_v2[-SEQ_LEN_V2:]

        # ── Routing ──────────────────────────────────────────────────
        if req.mode == "kata":
            use_v2 = True
        elif req.mode == "huruf":
            use_v2 = False
        else:
            use_v2 = last_left_active and last_right_active

        # ── Inferensi V2 ─────────────────────────────────────────────
        if use_v2:
            if len(buffer_v2) < SEQ_LEN_V2:
                return {"detected": True, "prediction": f"Mengumpulkan Kata ({len(buffer_v2)}/{SEQ_LEN_V2})"}

            input_tensor = torch.FloatTensor(
                np.array(buffer_v2[-SEQ_LEN_V2:])
            ).unsqueeze(0).to(device)

            with torch.no_grad():
                probs      = torch.softmax(model_v2(input_tensor), dim=1)[0]
                confidence = float(probs.max())
                pred_idx   = int(probs.argmax())

            pred_history_v2.append(pred_idx)
            stable_idx = Counter(pred_history_v2).most_common(1)[0][0]

            if confidence >= CONFIDENCE_THRESHOLD:
                prediction = label_map_v2[stable_idx]
                cooldown_frames = 15
                buffer_v1       = []
                buffer_v2       = []
                missing_frames  = 0
                pred_history_v2.clear()
                return {"detected": True, "prediction": str(prediction)}
            else:
                if len(buffer_v2) >= SEQ_LEN_V2:
                    buffer_v2 = buffer_v2[-20:]
                return {"detected": True, "prediction": f"Mengumpulkan Kata ({len(buffer_v2)}/{SEQ_LEN_V2})"}

        # ── Inferensi V1 ─────────────────────────────────────────────
        else:
            if len(buffer_v1) < SEQ_LEN_V1:
                return {"detected": True, "prediction": f"Mengumpulkan Huruf ({len(buffer_v1)}/{SEQ_LEN_V1})"}

            input_tensor = torch.FloatTensor(
                np.array(buffer_v1[-SEQ_LEN_V1:])
            ).unsqueeze(0).to(device)

            with torch.no_grad():
                probs      = torch.softmax(model_v1(input_tensor), dim=1)[0]
                confidence = float(probs.max())
                pred_idx   = int(probs.argmax())

            pred_history_v1.append(pred_idx)
            stable_idx = Counter(pred_history_v1).most_common(1)[0][0]
            if confidence >= CONFIDENCE_THRESHOLD:
                prediction = label_map_v1[stable_idx]
                cooldown_frames = 12
                buffer_v1       = []
                buffer_v2       = []
                missing_frames  = 0
                pred_history_v1.clear()
                return {"detected": True, "prediction": str(prediction)}
            else:
                if len(buffer_v1) >= SEQ_LEN_V1:
                    buffer_v1 = buffer_v1[-15:]
                return {"detected": True, "prediction": f"Mengumpulkan Huruf ({len(buffer_v1)}/{SEQ_LEN_V1})"}

    except Exception as e:
        print(f"[ERROR] {e} | time={time.time()-start:.3f}s")
        return {"error": str(e)}


# ══════════════════════════════════════════════════════
# RESET ENDPOINT
# ══════════════════════════════════════════════════════
@app.post("/reset_buffer")
def reset_buffer():
    global buffer_v1, buffer_v2, cooldown_frames, missing_frames
    global was_detected, last_landmarks_v1, last_landmarks_v2
    global last_left_active, last_right_active

    buffer_v1         = []
    buffer_v2         = []
    cooldown_frames   = 0
    missing_frames    = 0
    was_detected      = False
    last_landmarks_v1 = [0.0] * 63
    last_landmarks_v2 = [0.0] * 126
    last_left_active  = False
    last_right_active = False
    pred_history_v1.clear()
    pred_history_v2.clear()
    return {"status": "ok"}