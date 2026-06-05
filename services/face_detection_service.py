from facenet_pytorch import MTCNN
from PIL import Image

# Detector
mtcnn = MTCNN(keep_all=True)


def detect_faces(image: Image.Image):
    """
    Input: Full classroom image
    Output: list of face crops (PIL Images)
    """

    image = image.convert("RGB")

    # detect faces
    boxes, probs = mtcnn.detect(image)

    faces = []

    if boxes is None:
        return faces

    for box in boxes:

        # crop face
        face = image.crop(box)

        faces.append(face)

    return faces