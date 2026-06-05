from PIL import Image
from services.emotion_service import predict_emotion

# load image
image = Image.open("new_img.jpeg").convert("RGB")

# predict
result = predict_emotion(image)

print(result)