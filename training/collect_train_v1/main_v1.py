import cv2
import torch
import numpy as np
import mediapipe as mp
from collections import deque, Counter

# ======================
# MODEL
# ======================
class GestureModel(torch.nn.Module):
    def __init__(self, input_size=63, hidden=128, num_classes=10):
        super().__init__()

        self.lstm = torch.nn.LSTM(
            input_size,
            hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden * 2, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = torch.mean(out, dim=1)
        return self.fc(out)

# ======================
# NORMALIZATION (WAJIB SAMA)
# ======================
def normalize_per_sample(data):
    data = data.reshape(21, 3)

    wrist = data[0]
    data = data - wrist

    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val

    return data.flatten()

# ======================
# LOAD
# ======================
labels = np.load("label_map.npy", allow_pickle=True)

device = torch.device("cpu")

model = GestureModel(num_classes=len(labels)).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# ======================
# CONFIG
# ======================
sequence_length = 30
confidence_threshold = 0.5

sequence = deque(maxlen=sequence_length)
pred_history = deque(maxlen=5)

confidence = 0.0
final_output = "..."

# ======================
# MEDIAPIPE
# ======================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

# ======================
# LOOP
# ======================
while True: # loop utama untuk capture video dan prediksi
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks: 
        landmarks = [] 

        for hand_landmarks in result.multi_hand_landmarks: #
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS) 

            for lm in hand_landmarks.landmark: 
                landmarks.extend([lm.x, lm.y, lm.z]) 

        if len(landmarks) == 63: # validasi jumlah landmark
            landmarks = np.array(landmarks, dtype=np.float32)
            landmarks = normalize_per_sample(landmarks)

            sequence.append(landmarks)

    else:
        if len(sequence) > 0:
            sequence.popleft()

    # ======================
    # PREDICT
    # ======================
    if len(sequence) == sequence_length: # hanya prediksi jika sudah ada 30 frame
        input_data = torch.tensor(np.array(sequence), dtype=torch.float32).unsqueeze(0)

        with torch.no_grad(): # nonaktifkan gradient untuk efisiensi
            output = model(input_data)
            probs = torch.softmax(output, dim=1).numpy()[0]

        idx = np.argmax(probs)
        confidence = float(probs[idx]) # confidence untuk kelas yang diprediksi

        pred_history.append(idx) # simpan prediksi terakhir untuk voting
        most_common = Counter(pred_history).most_common(1)[0][0] # voting untuk stabilkan prediksi

        if confidence > confidence_threshold: # hanya tampilkan prediksi jika confidence > threshold
            final_output = str(labels[most_common])
        else:
            final_output = "..."

    # ======================
    # UI
    # ======================
    cv2.rectangle(frame, (0, 0), (640, 100), (30, 30, 30), -1)

    cv2.putText(frame, f"Gesture: {final_output}", (20, 60),
                font, 1, (255, 255, 255), 2)

    cv2.putText(frame, f"Conf: {confidence:.2f}", (400, 60),
                font, 0.6, (200, 200, 200), 2)

    cv2.imshow("Gesture FINAL FIX", frame)

    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()