import cv2
import torch
import numpy as np
import mediapipe as mp

from collections import deque, Counter

# ==========================================
# CONFIG
# ==========================================
SEQUENCE_LENGTH = 40

CONFIDENCE_THRESHOLD = 0.70

PRED_HISTORY = 7

MAX_MISSING_FRAMES = 8

MODEL_PATH = "best_model_v3.pth"
LABEL_PATH = "label_map_v3.npy"

# ==========================================
# NORMALIZATION
# ==========================================
def normalize_hand(hand_data):

    hand_data = hand_data - hand_data[0:3].repeat(21)

    wrist = hand_data[0:3]

    mid_mcp = hand_data[9*3 : 9*3+3]

    scale = np.linalg.norm(mid_mcp - wrist)

    if scale > 1e-6:
        hand_data = hand_data / scale

    return hand_data

# ==========================================
# MODEL
# ==========================================
class GestureBiLSTM(torch.nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.lstm = torch.nn.LSTM(
            input_size=126,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),

            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(64, num_classes)
        )

    def forward(self, x):

        out, _ = self.lstm(x)

        out = out[:, -1, :]

        return self.classifier(out)

# ==========================================
# LOAD MODEL
# ==========================================
device = torch.device("cpu")

labels = np.load(
    LABEL_PATH,
    allow_pickle=True
)

model = GestureBiLSTM(
    num_classes=len(labels)
).to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("Model Loaded")
print("Classes:", len(labels))

# ==========================================
# MEDIAPIPE
# ==========================================
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    max_num_hands=2,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

mp_draw = mp.solutions.drawing_utils

landmark_style = mp_draw.DrawingSpec(
    color=(0,255,128),
    thickness=2,
    circle_radius=3
)

connection_style = mp_draw.DrawingSpec(
    color=(255,255,0),
    thickness=2
)

# ==========================================
# CAMERA
# ==========================================
cap = cv2.VideoCapture(0)

# ==========================================
# STATE
# ==========================================
sequence = deque(maxlen=SEQUENCE_LENGTH)

prediction_history = deque(maxlen=PRED_HISTORY)

confidence = 0.0

final_output = "..."

missing_frames = 0

last_landmarks = np.zeros(126)

# ==========================================
# LOOP
# ==========================================

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    result = hands.process(rgb)

    left_hand = np.zeros(63)
    right_hand = np.zeros(63)

    hand_detected = False

    # =====================================
    # HAND DETECTION
    # =====================================
    if (
        result.multi_hand_landmarks
        and
        result.multi_handedness
    ):

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

            hand_data = np.array([
                val
                for lm in hand_landmarks.landmark
                for val in (lm.x, lm.y, lm.z)
            ])

            hand_data = normalize_hand(
                hand_data
            )

            hand_label = (
                handedness
                .classification[0]
                .label
            )
            
            if hand_label == "Left":
                left_hand = hand_data
            else:
                right_hand = hand_data

    # =====================================
    # COMBINE HANDS
    # =====================================
    landmarks = np.concatenate(
        [left_hand, right_hand]
    )

    # =====================================
    # HANDLE MISSING FRAMES
    # =====================================
    if hand_detected:
        sequence.append(
            landmarks
        )

        last_landmarks = landmarks.copy()

        missing_frames = 0

    else:

        if missing_frames < MAX_MISSING_FRAMES:

            sequence.append(
                last_landmarks
            )

            missing_frames += 1

    # =====================================
    # PREDICT
    # =====================================
    if len(sequence) == SEQUENCE_LENGTH:

        input_data = np.array(
            sequence,
            dtype=np.float32
        )
        
        input_tensor = torch.tensor(
            input_data
        ).unsqueeze(0).to(device)

        with torch.no_grad():

            output = model(
                input_tensor
            )

            probs = torch.softmax(
                output,
                dim=1
            )[0]

            confidence = float(
                probs.max()
            )

            pred_idx = int(
                probs.argmax()
            )

        prediction_history.append(
            pred_idx
        )

        stable_idx = Counter(
            prediction_history
        ).most_common(1)[0][0]

        if confidence > CONFIDENCE_THRESHOLD:

            final_output = str(
                labels[stable_idx]
            )

        else:

            final_output = "..."

    # =====================================
    # UI
    # =====================================
    h, w = frame.shape[:2]

    # Status
    status = (
        "ACTIVE"
        if hand_detected
        else
        "READY"
    )

    status_color = (
        (0, 255, 0)
        if hand_detected
        else
        (0, 165, 255)
    )

    # Top Panel
    cv2.rectangle(
        frame,
        (0, 0),
        (w, 140),
        (20, 20, 20),
        -1
    )

    # Title
    cv2.putText(
        frame,
        "SIBI Sign Language Recognition",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (200, 200, 200),
        2
    )

    # Gesture
    cv2.putText(
        frame,
        f"Gesture : {final_output}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )

    # Confidence
    cv2.putText(
        frame,
        f"Confidence : {confidence:.1%}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # Status
    cv2.putText(
        frame,
        f"Status : {status}",
        (700, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        status_color,
        2
    )

    # Model Info
    cv2.putText(
        frame,
        "BiLSTM Model",
        (700, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # Footer
    cv2.rectangle(
        frame,
        (0, h - 35),
        (w, h),
        (20, 20, 20),
        -1
    )

    cv2.putText(
        frame,
        "42 Gesture Classes | 4,200 Custom-Collected Sequences",
        (15, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 180, 180),
        1
    )   
    
    # Show Window
    cv2.imshow(
        "SIBI Sign Language Recognition",
        frame
    )

    # Exit with ESC
    key = cv2.waitKey(1)

    if key == 27:
        break

# ==========================================
# RELEASE
# ==========================================
cap.release()
cv2.destroyAllWindows()