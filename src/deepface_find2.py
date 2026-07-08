from deepface import DeepFace
import cv2
import os

# 設定來源圖片與資料集資料夾
script_dir = os.path.dirname(os.path.abspath(__file__))
source_img = os.path.join(script_dir, "face", "find_test1.jpg")
db_path = os.path.join(script_dir, "face", "database")

# 執行人臉辨識
results = DeepFace.find(
    img_path=source_img, 
    db_path=db_path,
    detector_backend="retinaface"
)

# 根據結果輸出 identity 或 ❌
if len(results) > 0 and not results[0].empty:
    identity = results[0].iloc[0]['identity']
    label = os.path.basename(os.path.dirname(identity))  # 取得父目錄名稱（人名），相容 Windows/Linux 路徑
    img = cv2.imread(source_img)
    cv2.putText(img, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(identity)
else:
    print("❌")
