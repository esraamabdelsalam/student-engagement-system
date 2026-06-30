import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# MODEL
# =========================
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

# ⚠️ لو عندك weights مدربة فعليًا فعلّي السطر ده:
# model.load_state_dict(torch.load("emotion.pth", map_location=device))

model.to(device)
model.eval()


# =========================
# LABELS
# =========================
idx_to_class = {
    0: "Angry",
    1: "Fear",
    2: "Happy",
    3: "Sad",
    4: "Surprise"
}


# =========================
# TRANSFORM
# =========================
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# SAFE CONVERSION
# =========================
def safe_convert(img):
    """
    Converts ANY input (Tensor / PIL / ndarray) → PIL Image
    """
    if img is None:
        return None

    # Tensor → PIL
    if torch.is_tensor(img):
        return to_pil_image(img)

    # numpy → PIL
    if isinstance(img, np.ndarray):
        return T.ToPILImage()(img)

    # already PIL
    return img


# =========================
# INFERENCE
# =========================
def predict_emotion_faces(faces):
    if not faces:
        return []

    processed = []

    for f in faces:

        f = safe_convert(f)

        if f is None:
            continue

        processed.append(transform(f))

    if len(processed) == 0:
        return []

    batch = torch.stack(processed).to(device)

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