import torch
import torch.nn.functional as F
from PIL import Image

from models.emotion.model import model_50
from utils.face_utils import extract_face

# --------------------
# labels
# --------------------
idx_to_class = {
    0: "Angry",
    1: "Fear",
    2: "Happy",
    3: "Sad",
    4: "Surprise"
}

# --------------------
# device
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_50.to(device)
model_50.eval()


def predict_emotion(image: Image.Image):

    face = extract_face(image)

    if face is None:
        return {"error": "No face detected"}

    face = face.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model_50(face)
        probs = F.softmax(outputs, dim=1)

        conf, pred = torch.max(probs, 1)

    pred = int(pred.item())

    return {
        "emotion": idx_to_class[pred],
        "confidence": float(conf.item())
    }