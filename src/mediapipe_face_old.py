import cv2
import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# 初始化 Mediapipe Face Landmarker（新版 Tasks API）
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="model/face_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
)
face_landmarker = FaceLandmarker.create_from_options(options)

# 讀取圖檔
image_path = "src/face/face.jpg"  # 替換為你的圖檔路徑
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# 將 BGR 圖片轉換為 RGB
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

# 偵測人臉地標
results = face_landmarker.detect(mp_image)

# 繪製地標
if results.face_landmarks:
    for face_landmarks in results.face_landmarks:
        points = []
        for idx, landmark in enumerate(face_landmarks):
            h, w, _ = image.shape
            x, y = int(landmark.x * w), int(landmark.y * h)
            points.append((x, y))
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
        
        # 繪製最外圍的白色線條
        hull = cv2.convexHull(np.array(points))
        cv2.polylines(image, [hull], isClosed=True, color=(255, 255, 255), thickness=2)

# 顯示結果
cv2.imshow("Face Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
