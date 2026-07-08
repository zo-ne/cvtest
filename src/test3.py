import cv2
import os

# 讀取圖片
image_path = os.path.join(os.path.dirname(__file__), 'data/nailong.jpg')
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Unable to read image from {image_path}")
else:
    # 1. 取得圖片的高度和寬度
    h, w = img.shape[:2]
    
    # 2. 【填空】取出圖片的左半邊
    # 提示：在 NumPy 中，[高 , 寬]。我們要保留所有的高度，但寬度只要從 0 到正中間
    left_half = img[:, :w//2]
    
    # 3. 將左半邊轉為灰階
    left_gray = cv2.cvtColor(left_half, cv2.COLOR_BGR2GRAY)
    
    # 4. 【大魔王填空】為什麼灰階要轉回 3 通道 (BGR)？
    # 因為原本的 img 是彩色的（有 B、G、R 三個顏色通道），
    # 灰階圖片只有 1 個通道。你不能把 1 通道的黑白貼回 3 通道的彩色陣列裡，形狀會對不起來！
    # 所以我們要用 cv2.cvtColor 把灰階圖「偽裝」成 3 通道的黑白圖：
    left_gray_bgr = cv2.cvtColor(left_gray, cv2.COLOR_GRAY2BGR)
    
    # 5. 【填空】把偽裝好的 3 通道黑白圖，塞回原圖的左半邊
    img[:, :w//2] = left_gray_bgr
    
    # 6. 顯示處理後的結果
    cv2.imshow('Half Gray Nailong', img)
    
    print("陰陽奶龍已顯示！按任意鍵關閉。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()