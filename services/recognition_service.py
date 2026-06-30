import os
import numpy as np
import torch
import joblib
import gdown

from facenet_pytorch import InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

# ======================
# DEVICE
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# PATHS
# ======================
SVM_PATH = "models/svm_face.pkl"
NORM_PATH = "models/normalizer.pkl"
CENTROIDS_PATH = "models/centroids.pkl"

SVM_ID = "1EeApYLi2P6kgRY4QObaI_kdob_cFcIV-"
NORM_ID = "1qLZHaZV8sJtyYNN7tYbWvY9BCpwMUjK1"
CENTROIDS_ID = "1L5wpCcILVW7wnKPKkQiJlmZJVOr4GE52"

# ======================
# MODELS
# ======================
svm_model = None
normalizer = None
centroids = None
face_model = None
loaded = False


def download_if_missing(path, file_id):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        gdown.download(
            f"https://drive.google.com/uc?id={file_id}",
            path,
            quiet=False
        )


def load_models():
    global svm_model, normalizer, centroids, face_model, loaded

    if loaded:
        return

    download_if_missing(SVM_PATH, SVM_ID)
    download_if_missing(NORM_PATH, NORM_ID)
    download_if_missing(CENTROIDS_PATH, CENTROIDS_ID)

    svm_model = joblib.load(SVM_PATH)
    normalizer = joblib.load(NORM_PATH)
    centroids = joblib.load(CENTROIDS_PATH)

    face_model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    loaded = True


# ======================
# LABELS
# ======================
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


# ======================
# CORE
# ======================
def recognize_faces(face_images, cosine_threshold=0.50):

    if not face_images:
        return []

    load_models()

    results = []

    with torch.inference_mode():

        for face in face_images:

            if face is None:
                continue

            # ======================
            # EMBEDDING
            # ======================
            face = face.unsqueeze(0).to(device)

            embedding = face_model(face)
            embedding = embedding.squeeze(0).cpu().numpy()

            # normalize ONCE
            embedding_norm = normalizer.transform([embedding])[0]

            # ======================
            # SVM
            # ======================
            probs = svm_model.predict_proba([embedding_norm])[0]

            pred = int(np.argmax(probs))
            svm_conf = float(np.max(probs))

            # ======================
            # COSINE (SAFE + FAST)
            # ======================
            centroid = centroids.get(pred)

            if centroid is not None:
                centroid = np.array(centroid).reshape(1, -1)
                cosine = float(
                    cosine_similarity([embedding_norm], centroid)[0][0]
                )
            else:
                cosine = 0.0

            # ======================
            # FINAL DECISION
            # ======================
            if cosine >= cosine_threshold:
                name = idx_to_class.get(pred, f"Person_{pred}")
            else:
                name = "Unknown"

            results.append({
                "student_id": pred,
                "student_name": name,

                # مهم: خفيف و usable فقط
                "embedding": embedding_norm.tolist(),

                "svc_confidence": round(svm_conf, 4),
                "cosine_similarity": round(cosine, 4),
                "recognition_confidence": round(cosine, 4)
            })

    return results


recognize_face = recognize_faces