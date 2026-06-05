import torch
import numpy as np
import joblib
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN

# --------------------
# Models
# --------------------
mtcnn = MTCNN(image_size=160, margin=20, keep_all=False)

face_model = InceptionResnetV1(pretrained='vggface2').eval()
svm_model = joblib.load("models/recognition/svm_face_recognition.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
face_model.to(device)

# --------------------
# Labels
# --------------------
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


# --------------------
# Recognition function
# --------------------
def recognize_face(image: Image.Image):

    image = image.convert("RGB")

    # 1. FACE DETECTION (VERY IMPORTANT)
    face = mtcnn(image)

    if face is None:
        return {"error": "No face detected"}

    face = face.unsqueeze(0).to(device)

    # 2. FACE EMBEDDING (ENCODING)
    with torch.no_grad():
        embedding = face_model(face)

    embedding = embedding.cpu().numpy().reshape(1, -1)

    # 3. SVM PREDICTION
    pred = svm_model.predict(embedding)[0]

    # 4. CONFIDENCE
    confidence = None
    if hasattr(svm_model, "predict_proba"):
        confidence = float(np.max(svm_model.predict_proba(embedding)[0]))

    return {
        "student_id": int(pred),
        "student_name": idx_to_class.get(int(pred), "Unknown"),
        "confidence": confidence
    }