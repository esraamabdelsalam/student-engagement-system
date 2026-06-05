from fastapi import APIRouter, UploadFile, File
from PIL import Image
import io

from services.recognition_service import recognize_face
from services.face_detection_service import detect_faces  # لازم يكون موجود

router = APIRouter(prefix="/recognition", tags=["Recognition"])


@router.post("/predict")
async def predict(file: UploadFile = File(...)):

    # 1. قراءة الصورة
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # 2. detect faces
    faces = detect_faces(image)

    results = []

    # 3. recognition لكل face
    for face in faces:
        result = recognize_face(face)
        results.append(result)

    # 4. return response
    return {
        "success": True,
        "num_faces": len(results),
        "results": results
    }