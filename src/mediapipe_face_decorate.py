import cv2
import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def overlay(original_image, overlay_image):
    overlay_image_rgb = overlay_image[:,:,:3]
    overlay_image_alpha = overlay_image[:,:,3:]

    overlay_mask = cv2.threshold(overlay_image_alpha, 0, 255, cv2.THRESH_BINARY)[1]
    original_mask = cv2.bitwise_not(overlay_mask)

    fig1 = cv2.bitwise_and(original_image, original_image, mask=original_mask)
    fig2 = cv2.bitwise_and(overlay_image_rgb, overlay_image_rgb, mask=overlay_mask)
    return cv2.add(fig1, fig2)

# 初始化 Mediapipe Face Landmarker（新版 Tasks API）
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="model/face_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
)
face_landmarker = FaceLandmarker.create_from_options(options)

# 讀取圖檔
image_path = "src/face/face5.jpg"  # 替換為你的圖檔路徑
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# 讀取圖檔
image_path = "src/face/crown.png"  # 替換為你的圖檔路徑
decorate_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
if decorate_image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# 將 BGR 圖片轉換為 RGB
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

# 偵測人臉地標
results = face_landmarker.detect(mp_image)
print(len(results.face_landmarks))

# 繪製地標
if results.face_landmarks:
    for face_landmarks in results.face_landmarks:
        points = []
        for idx, landmark in enumerate(face_landmarks):
            h, w, _ = image.shape
            x, y = int(landmark.x * w), int(landmark.y * h)
            points.append((x, y))
            # cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

            if idx == 54:
                x_54 = x
                y_54 = y
            if idx == 284:
                x_284 = x
                y_284 = y
            if idx == 10:
                x_10 = x
                y_10 = y

        # 繪製最外圍的白色線條
        hull = cv2.convexHull(np.array(points))
        cv2.polylines(image, [hull], isClosed=True, color=(255, 255, 255), thickness=2)

        # 計算 x_54, y_54 到 x_284, y_284 的斜率與角度
        dx = x_284 - x_54
        dy = y_284 - y_54
        angle = np.degrees(np.arctan2(dy, dx))

        # 繪製裝飾圖片（透明 PNG 合成）
        ratio = decorate_image.shape[0] / decorate_image.shape[1]
        decorate_image_resized = cv2.resize(
            decorate_image, 
            (int(x_284 - x_54), int((x_284 - x_54) * ratio)), 
            interpolation=cv2.INTER_AREA
        )

        # 旋轉裝飾圖（改為反方向，錨點設為下緣中心點）
        (h_dec, w_dec) = decorate_image_resized.shape[:2]
        center = (w_dec // 2, h_dec)  # 下緣中心點
        rot_mat = cv2.getRotationMatrix2D(center, -angle, 1.0)
        decorate_image_resized = cv2.warpAffine(
            decorate_image_resized, rot_mat, (w_dec, h_dec), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0)
        )

        # 以 x_10 為中心對齊裝飾圖
        x_center = x_10
        x1 = x_center - w_dec // 2
        x2 = x_center + (w_dec - w_dec // 2)
        y2 = y_10
        y1 = y2 - h_dec  # bottom 對齊 y_10

        # 若 top 超出貼圖區域，僅貼可見部分
        roi_y1 = max(0, y1)
        roi_y2 = y2
        crop_y1 = 0 if y1 >= 0 else -y1
        crop_y2 = decorate_image_resized.shape[0]

        # 計算貼圖區域
        roi = image[roi_y1:roi_y2, x1:x2]

        # 調整裝飾圖尺寸以符合 ROI
        h_roi = roi_y2 - roi_y1
        w_roi = x2 - x1
        decorate_cropped = decorate_image_resized[crop_y1:crop_y2, :]

        if decorate_cropped.shape[0] != h_roi or decorate_cropped.shape[1] != w_roi:
            decorate_cropped = cv2.resize(decorate_cropped, (w_roi, h_roi), interpolation=cv2.INTER_AREA)

        # 若裝飾圖有 alpha 通道，進行 overlay
        if decorate_cropped.shape[2] == 4 and roi.shape[0] > 0 and roi.shape[1] > 0:
            merged = overlay(roi, decorate_cropped)
            image[roi_y1:roi_y2, x1:x2] = merged
        elif roi.shape[0] > 0 and roi.shape[1] > 0:
            image[roi_y1:roi_y2, x1:x2] = decorate_cropped

# 顯示結果
cv2.imshow("Face Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
