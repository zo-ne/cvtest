import cv2
import os

# 1. 自動取得目前這支 .py 檔案所在的完整資料夾路徑 (即 src 資料夾)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 自動組合出 nailong.jpg 的絕對路徑 (在 src/data/ 裡面)
image_path = os.path.join(current_dir, 'data', 'nailong.jpg')

print(f"正在嘗試讀取奶龍圖片，絕對路徑為: {image_path}")

# 3. 讀取圖片
image = cv2.imread(image_path)

# 4. 防呆安全檢查，避免空圖片導致後面 cvtColor 崩潰
if image is None:
    print("❌ 錯誤：找不到奶龍圖片！")
    print(f"請確認 nailong.jpg 有放在此路徑下：{image_path}")
    exit()

# 5. 順利讀取後轉為灰階
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# 6. 滑桿回呼函數 (當拉動滑桿時會自動執行這裡)
def update_features(n_features):
    n_features = max(1, n_features)  # 避免為0
    
    # 建立 ORB 特徵偵測器，指定特徵點數量
    feature = cv2.ORB_create(n_features)
    
    # 偵測關鍵點 (Keypoints)
    kp = feature.detect(gray)
    
    # 將灰階圖轉回 BGR 彩色底圖，以便畫上彩色的特徵點圈圈
    img_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # 繪製特徵點（使用 DRAW_RICH_KEYPOINTS 會畫出帶有方向與大小的圈圈）
    feature_image = cv2.drawKeypoints(
        img_bgr, kp, None, 
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    # 顯示結果視窗
    cv2.imshow('ORB Feature Detection', feature_image)


# 7. 建立視窗與滑桿
cv2.namedWindow('ORB Feature Detection')

# 建立滑桿，範圍從 1 到 500 (原本 100 有點少，奶龍特徵點較大可以調多一點來試試)
cv2.createTrackbar('n_features', 'ORB Feature Detection', 10, 500, update_features)

# 初始化顯示（預設先偵測 10 個特徵點）
update_features(10)  

print("✅ 成功跑起來囉！拖動滑桿可以調整特徵點數量。")
print("👉 提示：請在圖片視窗上按下鍵盤『任意鍵』即可關閉程式。")

cv2.waitKey(0)
cv2.destroyAllWindows()