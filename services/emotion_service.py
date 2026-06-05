import torch
import torch.nn.functional as F
from PIL import Image
import gdown
import os
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from utils.face_utils import extract_face

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Model download from Drive
# =========================
MODEL_PATH = "emotion.pth"

DRIVE_URL = "https://drive.google.com/uc?id=1r-a6mRzm2GrjjYPuVXE5UMOxuTIvXkk7"

if not os.path.exists(MODEL_PATH):
    gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)

# =========================
# Model architecture (NO models folder needed)
# =========================
def get_model():
    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 1000),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1000, 5)
    )

    return model

# =========================
# Load model + weights
# =========================
model_50 = get_model()

state_dict = torch.load(MODEL_PATH, map_location=device)
model_50.load_state_dict(state_dict)

model_50.to(device)
model_50.eval()

# =========================
# Labels
# =========================
idx_to_class = {
    0: "Angry",
    1: "Fear",
    2: "Happy",
    3: "Sad",
    4: "Surprise"
}

# =========================
# Prediction function
# =========================
def predict_emotion(image: Image.Image):

    try:
        # 1. detect face
        face = extract_face(image)

        if face is None:
            return {"error": "No face detected"}

        # 2. prepare tensor
        face = face.unsqueeze(0).to(device)

        # 3. inference
        with torch.no_grad():
            outputs = model_50(face)
            probs = F.softmax(outputs, dim=1)

            conf, pred = torch.max(probs, 1)

        # 4. return result
        return {
            "emotion": idx_to_class[int(pred.item())],
            "confidence": float(conf.item())
        }

    except Exception as e:
        return {"error": str(e)}