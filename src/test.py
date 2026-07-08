import cv2
import os

# 1. 自動取得目前 test.py 檔案所在的完整資料夾路徑 (即 src 資料夾)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 自動組合出圖片的絕對路徑，確保萬無一失
image_path = os.path.join(current_dir, 'data', 'nailong.jpg')

print(f"正在嘗試讀取圖片，絕對路徑為: {image_path}")

# 3. 讀取圖片
img = cv2.imread(image_path)

if img is None:
    print("❌ 錯誤：在這個路徑找不到圖片！請確認 nailong.jpg 有放在 src 資料夾內。")
else:
    print("✅ 成功讀取圖片！即將顯示視窗...")
    cv2.imshow('nailong', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()