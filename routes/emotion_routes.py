from fastapi import APIRouter, UploadFile, File
from PIL import Image
import io

from services.emotion_service import predict_emotion

router = APIRouter(prefix="/emotion", tags=["Emotion"])

@router.post("/predict")
async def predict(file: UploadFile = File(...)):

    image = Image.open(io.BytesIO(await file.read()))

    result = predict_emotion(image)

    return {"success": True, "result": result}