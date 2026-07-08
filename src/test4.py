import cv2
import os

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(script_dir, 'data/digit.png')

# 读取图片
img = cv2.imread(img_path)
if img is None:
    print(f"无法读取图片: {img_path}")
    exit()

# 转为灰度
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 二值化
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# 找轮廓
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 按x坐标排序轮廓（从左到右）
contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

print(f"找到 {len(contours)} 个数字")

# 为每个轮廓裁切出独立图片并显示
for i, cnt in enumerate(contours):
    x, y, w, h = cv2.boundingRect(cnt)
    
    # 裁切出每个数字
    digit_img = img[y:y+h, x:x+w]
    
    # 显示
    cv2.imshow(f'Digit {i}', digit_img)
    print(f"数字 {i}: 位置({x}, {y}), 大小({w}x{h})")

cv2.waitKey(0)
cv2.destroyAllWindows()
