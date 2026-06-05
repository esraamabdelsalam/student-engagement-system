from fastapi import FastAPI

from routes.recognition_routes import router as recognition_router

app = FastAPI(
    title="Face Recognition API",
    version="1.0"
)

# ربط routes
app.include_router(recognition_router)


# اختبار سريع
@app.get("/")
def home():
    return {
        "message": "Face Recognition API is running 🚀"
    }