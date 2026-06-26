import os
import time
import gdown
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import defaultdict, deque

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# CONFIG
# =========================
MODEL_PATH = "models/engagement/engagement.pth"
FILE_ID = "1vyaQTQfc12nSjlVwiakfDcmBE-AW79Og"

SEQUENCE_LENGTH = 16
BUFFER_TIMEOUT = 30

# =========================
# DOWNLOAD MODEL
# =========================
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

if not os.path.exists(MODEL_PATH):
    gdown.download(
        f"https://drive.google.com/uc?id={FILE_ID}",
        MODEL_PATH,
        quiet=False
    )

# =========================
# MODEL
# =========================
class CNN_BiLSTM(nn.Module):

    def __init__(self):
        super().__init__()

        self.cnn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.cnn.fc = nn.Identity()

        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)
        x = self.cnn(x)
        x = x.view(B, T, 2048)

        _, (h_n, _) = self.lstm(x)

        x = torch.cat([h_n[-2], h_n[-1]], dim=1)
        x = self.dropout(x)

        return self.fc(x)


# =========================
# LOAD
# =========================
model = CNN_BiLSTM()

checkpoint = torch.load(MODEL_PATH, map_location=device)

if isinstance(checkpoint, dict):
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()

# =========================
# LABELS
# =========================
idx_to_class = {
    0: "Disengaged",
    1: "Engaged"
}

# =========================
# STATE
# =========================
student_buffers = defaultdict(
    lambda: {
        "frames": deque(maxlen=SEQUENCE_LENGTH),
        "last_seen": None
    }
)

# =========================
# PREDICTION
# =========================
def predict_engagement(frames):

    if len(frames) != SEQUENCE_LENGTH:
        return None

    x = torch.stack(frames).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)

        conf, pred = torch.max(probs, dim=1)

    return {
        "engagement": idx_to_class[int(pred.item())],
        "confidence": float(conf.item())
    }

# =========================
# PIPELINE
# =========================
def process_student_frame(student_id, face_tensor):

    if face_tensor is None:
        return None

    now = time.time()
    student = student_buffers[student_id]

    # =========================
    # TIMEOUT RESET
    # =========================
    if student["last_seen"] is not None:
        if now - student["last_seen"] > BUFFER_TIMEOUT:
            student["frames"].clear()

    student["last_seen"] = now

    # =========================
    # ADD FRAME
    # =========================
    frame = face_tensor.detach().cpu().clone()
    student["frames"].append(frame)

    # =========================
    # NOT READY
    # =========================
    if len(student["frames"]) < SEQUENCE_LENGTH:
        return {
            "student_id": student_id,
            "status": "collecting",
            "frames": len(student["frames"])
        }

    # =========================
    # PREDICT
    # =========================
    frames = list(student["frames"])

    result = predict_engagement(frames)

    # =========================
    # RESET (STRICT NO OVERLAP)
    # =========================
    student["frames"].clear()

    if result is None:
        return {
            "student_id": student_id,
            "status": "error"
        }

    return {
        "student_id": student_id,
        "engagement": result["engagement"],
        "confidence": result["confidence"],
        "status": "done"
    }

# =========================
# UTILITIES
# =========================
def clear_all():
    student_buffers.clear()


def remove_student(student_id):
    student_buffers.pop(student_id, None)