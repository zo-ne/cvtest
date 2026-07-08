from deepface import DeepFace
import cv2
import os

# 讀取圖片
script_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(script_dir, 'src/face/face_voyager.jpg')
img = cv2.imread(img_path)

# 加載眼睛和嘴巴檢測器
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# 使用 deepface 偵測人臉
detections = DeepFace.extract_faces(
    img_path=img_path, detector_backend='retinaface')

# 畫出 bounding box 和眼睛、嘴巴
for face in detections:
    area = face['facial_area']
    x, y, w, h = area['x'], area['y'], area['w'], area['h']
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # 在人臉區域內偵測眼睛
    roi_gray = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
    for (ex, ey, ew, eh) in eyes[:2]:  # 只取前2個眼睛
        cv2.circle(img, (x + ex + ew//2, y + ey + eh//2), 3, (255, 0, 0), -1)
    
    # 在人臉區域內偵測嘴巴
    roi_mouth = roi_gray[h//2:, :]
    mouths = mouth_cascade.detectMultiScale(roi_mouth, 1.3, 5)
    for (mx, my, mw, mh) in mouths[:1]:  # 只取第1個嘴巴
        cv2.circle(img, (x + mx + mw//2, y + h//2 + my + mh//2), 3, (0, 0, 255), -1)

# 顯示結果到螢幕上
cv2.imshow("Face Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
