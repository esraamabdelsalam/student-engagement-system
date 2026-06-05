from PIL import Image
from services.face_detection_service import detect_faces

# افتحي الصورة
image = Image.open("test_img.jpg").convert("RGB")

# detection
faces = detect_faces(image)

# النتيجة
print("Number of faces detected:", len(faces))

# احفظي الوجوه
for i, face in enumerate(faces):
    face.save(f"face_{i}.jpg")

print("Done")