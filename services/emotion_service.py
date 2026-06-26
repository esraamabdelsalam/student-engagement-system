import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import gdown
from torchvision.models import resnet50, ResNet50_Weights

# ======================
# DEVICE
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# MODEL DOWNLOAD
# ======================
MODEL_PATH = "emotion.pth"
FILE_ID = "1r-a6mRzm2GrjjYPuVXE5UMOxuTIvXkk7"

if not os.path.exists(MODEL_PATH):
    gdown.download(
        f"https://drive.google.com/uc?id={FILE_ID}",
        MODEL_PATH,
        quiet=False
    )

# ======================
# MODEL
# ======================
def build_model():
    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    model.fc = nn.Sequential(
        nn.Linear(2048, 1000),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1000, 5)
    )

    return model


model = build_model()

state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)

model.to(device)
model.eval()

# ======================
# LABELS
# ======================
idx_to_class = {
    0: "Angry",
    1: "Fear",
    2: "Happy",
    3: "Sad",
    4: "Surprise"
}

# ======================
# CORE FUNCTION (SAFE + FAST)
# ======================
def predict_emotion_faces(faces):

    if not faces:
        return []

    # ensure valid tensors only
    faces = [f for f in faces if f is not None]

    if len(faces) == 0:
        return []

    batch = torch.stack(faces).to(device)

    with torch.inference_mode():
        logits = model(batch)
        probs = F.softmax(logits, dim=1)

        conf, pred = torch.max(probs, dim=1)

    return [
        {
            "emotion": idx_to_class[int(p)],
            "emotion_confidence": float(c)
        }
        for p, c in zip(pred, conf)
    ]


# alias
predict_emotion = predict_emotion_faces