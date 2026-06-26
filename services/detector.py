import numpy as np
import torch
from retinaface import RetinaFace
import cv2

def detect_faces(image):
    img = np.array(image)

    faces = RetinaFace.detect_faces(img)

    if not isinstance(faces, dict):
        return [], []

    crops = []
    boxes = []

    for k in faces:
        x1, y1, x2, y2 = faces[k]["facial_area"]

        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        crop = cv2.resize(crop, (160, 160))

        crop = torch.from_numpy(crop).float().permute(2, 0, 1) / 255.0

        crops.append(crop)
        boxes.append([x1, y1, x2, y2])

    return crops, boxes