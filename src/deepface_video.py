import cv2
import os
import threading
from deepface import DeepFace

# 設定人臉資料庫路徑
script_dir = os.path.dirname(os.path.abspath(__file__))
database_path = os.path.join(script_dir, "face", "database")

# 辨識結果共享變數
result_label = "Analyzing..."
result_confidence = None
is_recognizing = False

def recognize(frame_copy):
    """背景執行緒：執行 DeepFace 辨識，結果寫回全域變數"""
    global result_label, result_confidence, is_recognizing
    try:
        result = DeepFace.find(
            img_path=frame_copy,
            db_path=database_path,
            enforce_detection=False,
            silent=True,
            anti_spoofing=True,
            model_name="Facenet",
            detector_backend="retinaface"
        )
        if len(result) > 0 and len(result[0]) > 0:
            best_match = result[0].iloc[0]
            identity = best_match['identity']
            result_label = os.path.basename(os.path.dirname(identity))

            distance = float(best_match.get('distance', 0.0))
            threshold = best_match.get('threshold', None)
            if threshold is not None and float(threshold) > 0:
                result_confidence = max(0.0, min(1.0, (float(threshold) - distance) / float(threshold)))
            else:
                result_confidence = max(0.0, min(1.0, 1.0 - distance))
        else:
            result_label = "Unknown"
            result_confidence = None
    except Exception:
        result_label = "Error"
        result_confidence = None
    finally:
        is_recognizing = False

# 開啟攝影機
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)  # 水平翻轉畫面

    # 每當上一次辨識結束後，才啟動新的背景辨識
    if not is_recognizing:
        is_recognizing = True
        threading.Thread(target=recognize, args=(frame.copy(),), daemon=True).start()

    # 主迴圈直接疊加上一次辨識結果，不等待
    color = (0, 255, 0) if result_label not in ("Unknown", "Error", "Analyzing...") else (0, 0, 255)
    cv2.putText(frame, f"Match: {result_label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    if result_confidence is not None:
        cv2.putText(frame, f"Confidence: {result_confidence * 100:.1f}%", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("DeepFace Video Recognition", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
