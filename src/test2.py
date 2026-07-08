import cv2
import os

# Read the image
image_path = os.path.join(os.path.dirname(__file__), 'data/nailong.jpg')
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Unable to read image from {image_path}")
else:
    # Display the image
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_mirror = cv2.flip(img, 1)
    img_rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    cv2.putText(
    img_rotated,                     # 1. 要畫在哪張圖片上
    "我是奶龍",       # 2. 要顯示的文字內容 (目前僅限英文/數字)
    (30, 100),               # 3. 文字左下角的座標位置 (X, Y)
    cv2.FONT_HERSHEY_SIMPLEX,# 4. 字型種類 (OpenCV 內建字型)
    1,                       # 5. 字體大小倍率 (1 代表原始大小)
    (0, 0, 255),             # 6. 顏色！依照 (B, G, R) 順序。所以 (0, 0, 255) 是純紅色！
    2                        # 7. 字體線條粗細 (單位是像素)
)
    
    cv2.rectangle(gc, (50, 50), (200, 200), (255, 0, 0), 3)  # Draw a blue rectangle with thickness of 3 pixels
    # cv2.imshow('nailong_gray', img_gray)
    # cv2.imshow('nailong_mirror', img_mirror)
    cv2.imshow('nailong_rotated', img_rotated)
    print("Image displayed. Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

