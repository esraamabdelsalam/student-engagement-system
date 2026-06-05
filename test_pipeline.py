from PIL import Image
from services.face_detection_service import detect_faces
from services.recognition_service import recognize_face

# load image
image = Image.open("new_img.jpg").convert("RGB")

# detect faces
faces = detect_faces(image)

print("Faces detected:", len(faces))

results = []

for i, face in enumerate(faces):

    result = recognize_face(face)

    results.append({
        "face_id": i,
        "result": result
    })

# print final output
for r in results:
    print(r)