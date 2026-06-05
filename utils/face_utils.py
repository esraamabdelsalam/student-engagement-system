from facenet_pytorch import MTCNN
from PIL import Image

mtcnn = MTCNN(image_size=160, margin=20, keep_all=False)

def extract_face(image: Image.Image):
    image = image.convert("RGB")
    face = mtcnn(image)

    return face