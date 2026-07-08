import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import queue
 
result_queue = queue.Queue(1)
VisionRunningMode = mp.tasks.vision.RunningMode

def print_result(result, output_image, timestamp_ms):
    frame = output_image.numpy_view().copy()
    # 繪製所有地標點
    if result.face_landmarks:
        for face_landmarks in result.face_landmarks:
            points = []
            for landmark in face_landmarks:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                points.append([x, y])
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            # 使用 convexHull 畫出最外圍白色線條
            points_np = np.array(points, dtype=np.int32)
            hull = cv2.convexHull(points_np)
            cv2.polylines(frame, [hull], True, (255, 255, 255), 2)
    result_queue.put(frame)

# 初始化 MediaPipe Face Landmarker
base_options = python.BaseOptions(model_asset_path='src/../model/face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)

# 建立 Face Landmarker
with vision.FaceLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 將 BGR 轉成 RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 轉成 mediapipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        # 進行偵測
        landmarker.detect_async(mp_image, int(time.time() * 1000))
        frame = result_queue.get()  # 可加上 timeout 避免阻塞過久

        # 顯示前先轉回 BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Face Landmarks', frame)
        if cv2.waitKey(1) == 27:  # 按下 ESC 離開
            break
    cap.release()
    cv2.destroyAllWindows()
