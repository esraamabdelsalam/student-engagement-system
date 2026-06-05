import joblib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1


# =========================
# Load FaceNet model
# =========================
face_model = InceptionResnetV1(pretrained='vggface2').eval()


# =========================
# Load SVM model
# =========================
svm_model = joblib.load("models/recognition/svm_face_recognition.pkl")


# =========================
# Label Mapping
# =========================
idx_to_class = {
    0: "Akshay Kumar",
    1: "Alexandra Daddario",
    2: "Alia Bhatt",
    3: "Amitabh Bachchan",
    4: "Andy Samberg",
    5: "Anushka Sharma",
    6: "Billie Eilish",
    7: "Brad Pitt",
    8: "Camila Cabello",
    9: "Charlize Theron",
    10: "Claire Holt",
    11: "Courtney Cox",
    12: "Dwayne Johnson",
    13: "Elizabeth Olsen",
    14: "Ellen Degeneres",
    15: "Esraa Mostafa",
    16: "Henry Cavill",
    17: "Hrithik Roshan",
    18: "Hugh Jackman",
    19: "Jessica Alba",
    20: "Kashyap",
    21: "Lisa Kudrow",
    22: "Margot Robbie",
    23: "Marmik",
    24: "Natalie Portman",
    25: "Priyanka Chopra",
    26: "Robert Downey Jr",
    27: "Roger Federer",
    28: "Tom Cruise",
    29: "Vijay Deverakonda",
    30: "Virat Kohli",
    31: "Zac Efron"
}


# =========================
# Preprocessing (used internally)
# =========================
preprocess = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])


# =========================
# Convert PIL image → tensor
# =========================
def prepare_image(image: Image.Image):

    image = image.convert("RGB")
    tensor = preprocess(image)
    return tensor.unsqueeze(0)  # [1, 3, 160, 160]


# =========================
# Extract embedding
# =========================
def extract_embedding(img_tensor):

    with torch.no_grad():
        embedding = face_model(img_tensor)

    return embedding.cpu().numpy()


# =========================
# Main recognition function
# =========================
def recognize_face(image: Image.Image):

    # 1. convert image → tensor
    img_tensor = prepare_image(image)

    # 2. embedding
    embedding = extract_embedding(img_tensor)
    embedding = np.array(embedding).reshape(1, -1)

    # 3. prediction
    pred = svm_model.predict(embedding)[0]

    student_name = idx_to_class.get(int(pred), "Unknown")

    # 4. confidence (if available)
    confidence = None
    if hasattr(svm_model, "predict_proba"):
        probs = svm_model.predict_proba(embedding)[0]
        confidence = float(np.max(probs))

    # 5. result
    return {
        "student_id": int(pred),
        "student_name": student_name,
        "confidence": confidence
    }