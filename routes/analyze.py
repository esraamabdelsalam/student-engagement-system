from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import asyncio

from services.face_utils import detect_faces
from services.recognition_service import recognize_faces
from services.emotion_service import predict_emotion_faces
from services.engagement_service import process_student_frame

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
    # DETECT FACES
    # =========================
    faces, _ = detect_faces(image)

    if not faces:
        return {
            "faces_detected": 0,
            "results": []
        }

    # =========================
    # PARALLEL INFERENCE (stateless)
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

        student_id = rec.get("student_id", None)

        # =========================
        # GET STATE (TRACKED STUDENT)
        # =========================
        student = student_manager.get(student_id if student_id is not None else i)

        # =========================
        # UPDATE RECOGNITION STATE
        # =========================
        student.student_id = rec.get("student_id")
        student.student_name = rec.get("student_name")
        student.embedding = rec.get("embedding")

        student.svc_confidence = rec.get("svc_confidence", 0.0)
        student.cosine_similarity = rec.get("cosine_similarity", 0.0)
        student.recognition_confidence = rec.get("recognition_confidence", 0.0)

        # =========================
        # UPDATE EMOTION STREAM
        # =========================
        student.add_emotion(emo.get("emotion", "Unknown"))

        # =========================
        # UPDATE ENGAGEMENT STREAM
        # =========================
        student.add_frame(face)

        engagement_result = None

        if student.is_engagement_ready():
            seq = student.get_engagement_sequence()

            engagement_result = await asyncio.to_thread(
                process_student_frame,
                student.track_id,
                seq
            )

            student.reset_engagement_cycle()

        # =========================
        # BUILD RESPONSE
        # =========================
        results.append({
            "face_id": i,

            # Recognition
            "student_name": student.student_name,
            "svc_confidence": student.svc_confidence,
            "cosine_similarity": student.cosine_similarity,
            "recognition_confidence": student.recognition_confidence,

            # Emotion (stream value)
            "emotion": student.emotion_buffer[-1] if student.emotion_buffer else "Unknown",
            "emotion_buffer_size": len(student.emotion_buffer),

            # Engagement (cycle-based)
            "engagement": (
                engagement_result.get("engagement")
                if engagement_result else "Not Ready"
            ),

            "engagement_confidence": (
                engagement_result.get("confidence", 0.0)
                if engagement_result else 0.0
            )
        })

    # =========================
    # CLEANUP OLD TRACKS
    # =========================
    student_manager.cleanup()

    return {
        "faces_detected": len(results),
        "results": results
    }