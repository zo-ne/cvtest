from colorfilters.colorfilters import HSVFilter
import cv2 as cv
import os

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 读取图片（尝试多个位置）
img_paths = [
    os.path.join(script_dir, 'src/data/beemo.png'),
]

img = None
for path in img_paths:
    if os.path.exists(path):
        img = cv.imread(path)
        if img is not None:
            print(f"已加载图片: {path}")
            break

if img is None:
    print("找不到图片文件")
    exit()

height, width, channel = img.shape
ratio = width / height
width = 300
height = int(width / ratio)
img = cv.resize(img, (width, height))

window = HSVFilter(img)
window.show()

print(window.lowerb)
print(window.upperb)
