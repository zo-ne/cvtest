from deepface import DeepFace

source_img = "src/face/find_test1.jpg"
vector = DeepFace.represent(
    source_img, 
    model_name="Facenet512",
    detector_backend="retinaface"
)[0]["embedding"]
print('維度：', len(vector))
