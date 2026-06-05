from fastapi import APIRouter, UploadFile, File
from PIL import Image
import io

from services.recognition_service import recognize_face

router = APIRouter(prefix="/recognition", tags=["Recognition"])

@router.post("/predict")
async def predict(file: UploadFile = File(...)):

    image = Image.open(io.BytesIO(await file.read()))

    result = recognize_face(image)

    return {"success": True, "result": result}