from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import asyncio

from services.face_utils import detect_faces
from services.recognition_service import recognize_faces
from services.emotion_service import predict_emotion_faces
from services.engagement_service import predict_engagement_sequence
from services.student_manager import student_manager

router = APIRouter(prefix="/analyze", tags=["Analyze"])


@router.post("/")
async def analyze(file: UploadFile = File(...)):

    # =========================
    # LOAD IMAGE
    # =========================
    try:
        image = Image.open(file.file).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    # =========================
    # FACE DETECTION
    # =========================
    faces, _ = detect_faces(image)

    if not faces:
        return {"faces_detected": 0, "results": []}

    # =========================
    # PARALLEL MODELS
    # =========================
    recognition_task = asyncio.to_thread(recognize_faces, faces)
    emotion_task = asyncio.to_thread(predict_emotion_faces, faces)

    recognition_result, emotion_result = await asyncio.gather(
        recognition_task,
        emotion_task
    )

    results = []

    # =========================
    # PROCESS EACH FACE
    # =========================
    for i, face in enumerate(faces):

        rec = recognition_result[i]
        emo = emotion_result[i]

        # =====================================================
        # FIX #1: STABLE TRACKING KEY (VERY IMPORTANT)
        # =====================================================
        student_id = rec.get("student_id")

        if student_id is None:
            student_id = f"unknown_{i}"

        student = student_manager.get(student_id)
        student.update_last_seen()

        # =========================
        # RECOGNITION
        # =========================
        student.student_id = rec.get("student_id")
        student.student_name = rec.get("student_name")
        student.recognition_confidence = rec.get("recognition_confidence", 0.0)

        # =========================
        # EMOTION STREAM
        # =========================
        live_emotion = emo.get("emotion", "Unknown")
        student.add_emotion(live_emotion)

        emotion_final = student.get_emotion_result()

        # =========================
        # ENGAGEMENT STREAM
        # =========================
        if face is not None:
            student.add_frame(face)

        seq = student.get_engagement_sequence()

        engagement_result = None
        status = "Collecting"

        if seq is not None:

            engagement_result = await asyncio.to_thread(
                predict_engagement_sequence,
                seq
            )

            status = "Completed"

        # =========================
        # RESPONSE
        # =========================
        results.append({

            "face_id": i,

            # identity
            "student_name": student.student_name,
            "recognition_confidence": student.recognition_confidence,

            # LIVE emotion
            "live_emotion": live_emotion,
            "emotion_buffer_size": len(student.emotion_buffer),

            # FINAL emotion (mode)
            "emotion": emotion_final if emotion_final else "Collecting",

            # ENGAGEMENT
            "engagement": (
                engagement_result["engagement"]
                if engagement_result
                else "Collecting"
            ),

            "engagement_confidence": (
                engagement_result["confidence"]
                if engagement_result
                else 0.0
            ),

            "engagement_buffer_size": len(student.engagement_buffer),

            "status": status
        })

    # =========================
    # CLEANUP EXPIRED SESSIONS
    # =========================
    student_manager.cleanup()

    return {
        "faces_detected": len(results),
        "results": results
    }