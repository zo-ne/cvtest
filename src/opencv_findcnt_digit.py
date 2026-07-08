import cv2
import os

# 讀取圖片（灰階模式）
script_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(script_dir, 'data/lantern.png')
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# 反相二值化（閾值200）
_, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

# 尋找輪廓
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 將灰階圖轉為BGR以便畫彩色線
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# 畫出所有輪廓及其 bounding box
for i, cnt in enumerate(contours):
    # 畫輪廓
    cv2.drawContours(img_color, [cnt], -1, (0, 255, 0), 2)
    # 計算並畫出 bounding box
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # 裁切出每個輪廓範圍內的圖片
    roi = img[y:y+h, x:x+w]
    cv2.imshow(f'ROI {i}', roi)

# 顯示結果
cv2.imshow('Contours and Bounding Boxes', img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
