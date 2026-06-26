from fastapi import APIRouter, UploadFile, File
from PIL import Image

from services.recognition_service import recognize_face
from services.emotion_service import predict_emotion

router = APIRouter(prefix="/analyze", tags=["Analyze"])

@router.post("/")
async def analyze(file: UploadFile = File(...)):

    image = Image.open(file.file).convert("RGB")

    return {
        "recognition": recognize_face(image),
        "emotion": predict_emotion(image)
    }