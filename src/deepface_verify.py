from deepface import DeepFace
from operator import itemgetter
import cv2
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
img1_path = os.path.join(script_dir, 'face', 'face1.jpg')
img2_path = os.path.join(script_dir, 'face', 'face3.jpg')

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)
# 使用 DeepFace 進行人臉比對
result = DeepFace.verify(
    img1_path=img1_path, 
    img2_path=img2_path, 
    model_name='Facenet', 
    detector_backend='retinaface'
)
print(result)
color = (0, 255, 0) if result['verified'] else (0, 0, 255)  # 綠色或紅色 

# 使用 opencv 畫出 bounding box
x1, y1, w1, h1 = itemgetter("x", "y", "w", "h")(result['facial_areas']['img1'])
x2, y2, w2, h2 = itemgetter("x", "y", "w", "h")(result['facial_areas']['img2'])
cv2.rectangle(img1, (x1, y1), (x1 + w1, y1 + h1), color, 2)
cv2.rectangle(img2, (x2, y2), (x2 + w2, y2 + h2), color, 2)
cv2.imshow("image1", img1)
cv2.imshow("image2", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()