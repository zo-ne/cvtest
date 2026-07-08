import cv2
import os
import numpy as np

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
    print("✅ 成功讀取圖片！")
    
    # 轉換為灰度圖
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 二值化 (使用簡單閾值, 200 為閾值)
    ret, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # 1. 高斯模糊
    blurred = cv2.GaussianBlur(binary, (5, 5), 0)
    
    # 2. 銳化
    kernel_sharpen = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])
    sharpened = cv2.filter2D(blurred, -1, kernel_sharpen)
    
    # 3. 侵蝕與膨脹處理 (迭代20次)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # 先侵蝕
    eroded = cv2.erode(sharpened, kernel, iterations=20)
    
    # 再膨脹
    dilated = cv2.dilate(eroded, kernel, iterations=20)
    
    # 4. 轉換為彩色以便合併 (因為原圖是彩色的)
    binary_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    dilated_color = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
    
    # 5. 水平合併
    combined = np.hstack([binary_color, dilated_color])
    
    # 顯示合併後的圖片
    cv2.imshow('Binary vs Processed (Gaussian Blur + Sharpen + Erode/Dilate x20)', combined)
    
    print("📊 圖片處理完成！")
    print(f"   - 二值化閾值: 200")
    print(f"   - 高斯模糊: (5, 5)")
    print(f"   - 銳化: 已應用")
    print(f"   - 侵蝕與膨脹: 各迭代20次")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()