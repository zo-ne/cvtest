import cv2
import os
import numpy as np

# 取得當前目錄
current_dir = os.path.dirname(os.path.abspath(__file__))

# 讀取底圖 (kuilong.jpg)
base_path = os.path.join(current_dir, 'data', 'kuilong.jpg')
overlay_path = os.path.join(current_dir, 'data', 'nailong2.jpg')

print(f"底圖路徑: {base_path}")
print(f"疊圖路徑: {overlay_path}")

# 讀取圖片
base_img = cv2.imread(base_path)
overlay_img = cv2.imread(overlay_path)

if base_img is None or overlay_img is None:
    print("❌ 錯誤：無法讀取圖片！請確認檔案存在。")
else:
    print("✅ 成功讀取兩張圖片")
    
    # 獲取底圖的尺寸
    height, width = base_img.shape[:2]
    
    # 調整疊圖大小為與底圖相同
    overlay_img = cv2.resize(overlay_img, (width, height))
    
    print(f"📐 圖片大小: {width}x{height}")
    
    # 圖片疊加 (使用加權混合)
    result = cv2.addWeighted(base_img, 0.5, overlay_img, 0.5, 0)
    
    # 顯示最終結果
    cv2.imshow('Blended Result', result)
    
    print("🎨 圖片疊加完成！")
    print(f"   - 混合比例：50% / 50%")
    print(f"   按任意鍵關閉視窗...")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
