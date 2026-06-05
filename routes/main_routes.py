from fastapi import APIRouter, UploadFile, File
from PIL import Image
import io

from services.face_detection_service import detect_faces
from services.recognition_service import recognize_face
from services.emotion_service import predict_emotion

router = APIRouter()


@router.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    faces = detect_faces(image)

    results = []

    for face in faces:

        face_result = recognize_face(face)
        emotion_result = predict_emotion(face)

        results.append({
            "face": face_result,
            "emotion": emotion_result
        })

    return {
        "success": True,
        "num_faces": len(results),
        "results": results
    }