from fastapi import APIRouter, UploadFile, File
from PIL import Image
import io

from services.recognition_service import recognize_face

router = APIRouter(prefix="/recognition", tags=["Recognition"])


@router.post("/predict")
async def predict(file: UploadFile = File(...)):

    # 1. قراءة الصورة
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # 2. inference مباشرة (بدون preprocess هنا)
    result = recognize_face(image)

    return {
        "success": True,
        "result": result
    }