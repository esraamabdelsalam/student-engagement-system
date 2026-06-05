from fastapi import FastAPI

from routes.recognition_routes import router as recognition_router
from routes.emotion_routes import router as emotion_router

app = FastAPI(title="Student Engagement System")

app.include_router(recognition_router)
app.include_router(emotion_router)