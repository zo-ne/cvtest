import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 初始化 MediaPipe Face Landmarker
base_options = python.BaseOptions(model_asset_path='src/../model/face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options)

# 建立 Face Landmarker
with vision.FaceLandmarker.create_from_options(options) as landmarker:
    # 讀取圖片
    image = mp.Image.create_from_file('src/face/face.jpg')
    
    # 進行偵測
    detection_result = landmarker.detect(image)
    
    # 讀取原始圖片用於繪製
    img = cv2.imread('src/face/face.jpg')
    
    # 繪製所有地標點
    if detection_result.face_landmarks:
        for face_landmarks in detection_result.face_landmarks:
            points = []
            for landmark in face_landmarks:
                x = int(landmark.x * img.shape[1])
                y = int(landmark.y * img.shape[0])
                points.append([x, y])
                cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
            # 使用 convexHull 畫出最外圍白色線條
            points_np = np.array(points, dtype=np.int32)
            hull = cv2.convexHull(points_np)
            cv2.polylines(img, [hull], True, (255, 255, 255), 2)
    
    # 顯示結果
    cv2.imshow('Face Landmarks', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
