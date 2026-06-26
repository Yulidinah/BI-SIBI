import torch
import torch.nn as nn
import numpy as np

class GestureBiLSTM_V1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=63, hidden_size=128, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.3
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# Load
labels = np.load(r"models\\label_map_v1.npy", allow_pickle=True)
model = GestureBiLSTM_V1(num_classes=len(labels))
model.load_state_dict(torch.load(r"models\\best_model_v1.pth", map_location="cpu"))
model.eval()

# Test dengan data asli dari dataset
sample = np.load(r"datasets\\dataset_v1\\A\\1777373724.5347443.npy")
print(f"Sample shape: {sample.shape}")

tensor = torch.FloatTensor(sample).unsqueeze(0)
with torch.no_grad():
    out   = model(tensor)
    probs = torch.softmax(out, dim=1)[0]
    conf  = float(probs.max())
    pred  = int(probs.argmax())

print(f"Prediksi: {labels[pred]} | Confidence: {conf:.4f}")
print(f"Top 5:")
top5 = probs.topk(5)
for i, (v, idx) in enumerate(zip(top5.values, top5.indices)):
    print(f"  {i+1}. {labels[idx]}: {v:.4f}")