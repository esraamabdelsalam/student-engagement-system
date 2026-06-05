import torch
import numpy as np
import joblib
import gdown
import os
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Download SVM model (Drive fallback)
# =========================
MODEL_PATH = "models/recognition/svm_face_recognition.pkl"

DRIVE_URL = "https://drive.google.com/uc?id=1vthpA88ww3gTY0La9pIGDaXVAX3S6trI"

if not os.path.exists(MODEL_PATH):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)

# =========================
# Models
# =========================
mtcnn = MTCNN(image_size=160, margin=20, keep_all=False)

face_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
svm_model = joblib.load(MODEL_PATH)

# =========================
# Labels
# =========================
idx_to_class = {
    0: "Akshay Kumar", 1: "Alexandra Daddario", 2: "Alia Bhatt",
    3: "Amitabh Bachchan", 4: "Andy Samberg", 5: "Anushka Sharma",
    6: "Billie Eilish", 7: "Brad Pitt", 8: "Camila Cabello",
    9: "Charlize Theron", 10: "Claire Holt", 11: "Courtney Cox",
    12: "Dwayne Johnson", 13: "Elizabeth Olsen", 14: "Ellen Degeneres",
    15: "Esraa Mostafa", 16: "Henry Cavill", 17: "Hrithik Roshan",
    18: "Hugh Jackman", 19: "Jessica Alba", 20: "Kashyap",
    21: "Lisa Kudrow", 22: "Margot Robbie", 23: "Marmik",
    24: "Natalie Portman", 25: "Priyanka Chopra", 26: "Robert Downey Jr",
    27: "Roger Federer", 28: "Tom Cruise", 29: "Vijay Deverakonda",
    30: "Virat Kohli", 31: "Zac Efron"
}

# =========================
# Main Function
# =========================
def recognize_face(image: Image.Image):

    image = image.convert("RGB")

    # 1. Detect face
    face = mtcnn(image)

    if face is None:
        return {"error": "No face detected"}

    # 2. Add batch dim
    face = face.unsqueeze(0).to(device)

    # 3. FaceNet embedding
    with torch.no_grad():
        embedding = face_model(face)

    # IMPORTANT FIX: normalize embedding (VERY IMPORTANT for SVM)
    embedding = embedding / torch.norm(embedding, p=2, dim=1, keepdim=True)

    embedding = embedding.cpu().numpy()

    # 4. Prediction
    pred = svm_model.predict(embedding)[0]

    # 5. Confidence
    confidence = None
    if hasattr(svm_model, "predict_proba"):
        confidence = float(np.max(svm_model.predict_proba(embedding)[0]))

    # 6. Output
    return {
        "student_id": int(pred),
        "student_name": idx_to_class.get(int(pred), "Unknown"),
        "confidence": confidence
    }