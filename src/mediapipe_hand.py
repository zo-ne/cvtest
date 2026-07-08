import cv2
import mediapipe as mp
import time
import numpy as np # numpy is used by mp.Image


def random_vivid_bgr():
    # 用 HSV 產生高飽和高亮度色彩，做出更有科幻感的線條
    hsv_pixel = np.uint8([[[np.random.randint(0, 180), np.random.randint(220, 256), np.random.randint(220, 256)]]])
    bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2])


def smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def lerp_color(color_a, color_b, t):
    return tuple(int(color_a[i] + (color_b[i] - color_a[i]) * t) for i in range(3))


def scale_color(color, scale):
    return tuple(min(255, max(0, int(channel * scale))) for channel in color)


PLASMA_BLUE = (255, 200, 30)


def spawn_endpoint_sparks(particles, origin, target, count=4):
    ox, oy = origin
    tx, ty = target
    dx = float(tx - ox)
    dy = float(ty - oy)
    length = float(np.hypot(dx, dy))
    if length < 1e-6:
        return

    # 連線方向單位向量
    ux = dx / length
    uy = dy / length
    # 垂直向量，提供微量側向擾動，避免粒子過於死板
    px = -uy
    py = ux

    for _ in range(count):
        speed = np.random.uniform(1.2, 3.8)
        lateral = np.random.uniform(-0.8, 0.8)
        vx = ux * speed + px * lateral
        vy = uy * speed + py * lateral
        particles.append({
            "x": float(ox),
            "y": float(oy),
            "vx": float(vx),
            "vy": float(vy),
            "life": float(np.random.uniform(10, 22)),
            "max_life": 22.0,
            "size": float(np.random.uniform(1.2, 3.0)),
        })


def update_and_draw_sparks(particles, core_overlay, glow_overlay, frame_w, frame_h):
    alive_particles = []
    for p in particles:
        p["x"] += p["vx"]
        p["y"] += p["vy"]
        p["vx"] *= 0.96
        p["vy"] = p["vy"] * 0.96 + 0.03
        p["life"] -= 1.0

        if p["life"] <= 0:
            continue

        px = int(p["x"])
        py = int(p["y"])
        if px < 0 or py < 0 or px >= frame_w or py >= frame_h:
            continue

        alpha = max(0.0, p["life"] / p["max_life"])
        brightness = int(255 * alpha)
        # 電漿藍粒子 (BGR): 藍通道最強、綠次之、紅最低
        color = (brightness, int(brightness * 0.78), int(brightness * 0.08))
        radius = max(1, int(1 + p["size"] * alpha))

        cv2.circle(core_overlay, (px, py), radius, color, -1, cv2.LINE_AA)
        cv2.circle(glow_overlay, (px, py), radius * 3, color, -1, cv2.LINE_AA)
        alive_particles.append(p)

    # 限制粒子數量，避免長時間執行造成效能下降
    return alive_particles[-900:]

# 從 mediapipe.tasks.python 匯入必要的模組
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult # 主要用於類型提示，此處未使用回呼
VisionRunningMode = mp.tasks.vision.RunningMode

# 為了取得 HAND_CONNECTIONS
from mediapipe.tasks.python.vision import HandLandmarksConnections

# 手部地標模型檔案的路徑
# 請從 https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task 下載
MODEL_PATH = 'src/../model/hand_landmarker.task' 

def main():
    try:
        # 建立 HandLandmarkerOptions
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO, # 使用 VIDEO 模式進行即時影像流處理
            num_hands=2,                         # 最多偵測 2 隻手
            min_hand_detection_confidence=0.4,   # 手部偵測的最小信心度
            min_hand_presence_confidence=0.4,    # 手部存在的最小信心度 (用於追蹤)
            min_tracking_confidence=0.4)         # 手部追蹤的最小信心度

        # 使用 'with' 陳述式建立 HandLandmarker，確保資源被正確釋放
        with HandLandmarker.create_from_options(options) as landmarker:
            # 開啟攝影機
            cap = cv2.VideoCapture(0)
            # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            if not cap.isOpened():
                print("無法開啟攝影機")
                return

            spark_particles = []
            fingertip_indices = [4, 8, 12, 16, 20]
            line_color_states = {}
            for tip_idx in fingertip_indices:
                base_color = random_vivid_bgr()
                line_color_states[tip_idx] = {
                    "current": base_color,
                    "target": random_vivid_bgr(),
                    "progress": np.random.uniform(0.0, 1.0),
                    "speed": np.random.uniform(0.02, 0.06),
                }

            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("忽略空的攝影機影像幀。")
                    continue

                # 1. 影像讀取後先左右鏡像處理
                image = cv2.flip(image, 1)

                # 2. 將 BGR 影像轉換為 RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # 3. 建立 MediaPipe Image 物件
                # MediaPipe Image 需要 NumPy 陣列
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
                
                # 4. 獲取當前時間戳 (毫秒)
                frame_timestamp_ms = int(time.time() * 1000)

                # 5. 進行手部地標偵測 (同步呼叫)
                hand_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

                # 6. 處理偵測結果並在影像上繪製 (在原始 BGR 影像上操作)
                # 背景壓暗，凸顯能量光效
                annotated_image = cv2.convertScaleAbs(image, alpha=0.45, beta=0)
                spark_core_overlay = np.zeros_like(annotated_image)
                spark_glow_overlay = np.zeros_like(annotated_image)

                if hand_landmarker_result.hand_landmarks:
                    # 五個指尖的地標索引: thumb, index, middle, ring, pinky
                    left_fingertips = {}
                    right_fingertips = {}
                    hand_line_overlay = np.zeros_like(annotated_image)
                    hand_glow_overlay = np.zeros_like(annotated_image)
                    tip_core_overlay = np.zeros_like(annotated_image)
                    tip_glow_overlay = np.zeros_like(annotated_image)

                    for i in range(len(hand_landmarker_result.hand_landmarks)):
                        hand_landmarks = hand_landmarker_result.hand_landmarks[i]
                        handedness = hand_landmarker_result.handedness[i][0].category_name

                        # 根據左右手設定顏色 (OpenCV 使用 BGR 格式)
                        if handedness == "Left":
                            hand_color = (0, 255, 0)  # 綠色
                        else:  # Right
                            hand_color = (0, 0, 255)  # 紅色
                        
                        joint_color = (0, 255, 255) # 黃色 (用於關節點)

                        # 繪製手部連接線
                        if HandLandmarksConnections.HAND_CONNECTIONS:
                            for connection in HandLandmarksConnections.HAND_CONNECTIONS:
                                start_idx = connection.start
                                end_idx = connection.end

                                if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                                    start_landmark = hand_landmarks[start_idx]
                                    end_landmark = hand_landmarks[end_idx]

                                    # 將正規化座標轉換為像素座標
                                    start_point = (int(start_landmark.x * annotated_image.shape[1]),
                                                   int(start_landmark.y * annotated_image.shape[0]))
                                    end_point = (int(end_landmark.x * annotated_image.shape[1]),
                                                 int(end_landmark.y * annotated_image.shape[0]))
                                    
                                    # 手掌骨架改成半透明圖層，並加入低調光暈
                                    cv2.line(hand_line_overlay, start_point, end_point, hand_color, 2, cv2.LINE_AA)
                                    cv2.line(hand_glow_overlay, start_point, end_point, hand_color, 10, cv2.LINE_AA)

                        # 繪製關節點 (黃色實心圓) 並標上編號
                        for idx, landmark in enumerate(hand_landmarks):
                            x = int(landmark.x * annotated_image.shape[1])
                            y = int(landmark.y * annotated_image.shape[0])
                            
                            # 繪製黃色實心圓
                            cv2.circle(annotated_image, (x, y), 5, joint_color, -1) 
                            
                            # 標上編號 (白色文字)
                            cv2.putText(annotated_image, str(idx), (x + 8, y - 8), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

                        # 收集左右手五個指尖的像素座標
                        for tip_idx in fingertip_indices:
                            if tip_idx < len(hand_landmarks):
                                tip_landmark = hand_landmarks[tip_idx]
                                tip_point = (int(tip_landmark.x * annotated_image.shape[1]),
                                             int(tip_landmark.y * annotated_image.shape[0]))

                                # 指尖電漿藍核心與外圈光暈
                                cv2.circle(tip_core_overlay, tip_point, 4, PLASMA_BLUE, -1, cv2.LINE_AA)
                                cv2.circle(tip_glow_overlay, tip_point, 14, PLASMA_BLUE, -1, cv2.LINE_AA)

                                if handedness == "Left":
                                    left_fingertips[tip_idx] = tip_point
                                elif handedness == "Right":
                                    right_fingertips[tip_idx] = tip_point

                    # 左手五個指尖分別與右手對應指尖連線 (半透明 + 光暈附魔效果)
                    line_overlay = np.zeros_like(annotated_image)
                    glow_overlay = np.zeros_like(annotated_image)
                    now_s = time.time()
                    pulse = 0.55 + 0.45 * np.sin(now_s * 5.0)

                    for tip_idx in fingertip_indices:
                        if tip_idx in left_fingertips and tip_idx in right_fingertips:
                            # 隨機色平滑過渡，避免跳色
                            color_state = line_color_states[tip_idx]
                            color_state["progress"] += color_state["speed"]
                            if color_state["progress"] >= 1.0:
                                color_state["current"] = color_state["target"]
                                color_state["target"] = random_vivid_bgr()
                                color_state["progress"] = 0.0
                                color_state["speed"] = np.random.uniform(0.02, 0.06)

                            color_t = smoothstep(color_state["progress"])
                            enchant_color = lerp_color(color_state["current"], color_state["target"], color_t)

                            # 淡入淡出：亮度週期性脈動
                            fade = 0.55 + 0.45 * (0.5 - 0.5 * np.cos(2.0 * np.pi * (now_s * 1.2 + tip_idx * 0.17)))
                            line_color = scale_color(enchant_color, 0.7 + 0.45 * fade)
                            glow_color = scale_color(enchant_color, 0.95 + 0.55 * fade)

                            start_pt = left_fingertips[tip_idx]
                            end_pt = right_fingertips[tip_idx]

                            # 內層亮線加粗
                            cv2.line(line_overlay, start_pt, end_pt, line_color, 3, cv2.LINE_AA)

                            # 外層光暈基底加厚，讓發光更明顯
                            cv2.line(glow_overlay, start_pt, end_pt, glow_color, 24, cv2.LINE_AA)

                            # 單向施法感：火花由右手流向左手
                            spawn_endpoint_sparks(spark_particles, end_pt, start_pt, count=6)

                            # 沿連線加入流動能量光點，方向由右手到左手
                            flow_phase = (now_s * 1.8 + tip_idx * 0.17) % 1.0
                            for k in range(4):
                                t = (flow_phase + k * 0.25) % 1.0
                                fx = int(end_pt[0] + (start_pt[0] - end_pt[0]) * t)
                                fy = int(end_pt[1] + (start_pt[1] - end_pt[1]) * t)
                                cv2.circle(glow_overlay, (fx, fy), 12, PLASMA_BLUE, -1, cv2.LINE_AA)
                                cv2.circle(line_overlay, (fx, fy), 3, (255, 255, 255), -1, cv2.LINE_AA)

                    # 對光暈層做高斯模糊，形成發光擴散
                    hand_glow_overlay = cv2.GaussianBlur(hand_glow_overlay, (0, 0), 4)
                    hand_glow_overlay = cv2.GaussianBlur(hand_glow_overlay, (0, 0), 6)
                    glow_overlay = cv2.GaussianBlur(glow_overlay, (0, 0), 7)
                    glow_overlay = cv2.GaussianBlur(glow_overlay, (0, 0), 13)
                    tip_glow_overlay = cv2.GaussianBlur(tip_glow_overlay, (0, 0), 6)
                    tip_glow_overlay = cv2.GaussianBlur(tip_glow_overlay, (0, 0), 12)

                    # 先混合手掌半透明骨架，再疊加指尖白光與指尖連線特效
                    annotated_image = cv2.addWeighted(annotated_image, 1.0, hand_glow_overlay, 0.26, 0)
                    annotated_image = cv2.addWeighted(annotated_image, 1.0, hand_line_overlay, 0.42, 0)
                    annotated_image = cv2.addWeighted(annotated_image, 1.0, tip_glow_overlay, 0.52 + 0.16 * pulse, 0)
                    annotated_image = cv2.addWeighted(annotated_image, 1.0, tip_core_overlay, 0.62, 0)

                    # 疊加光暈，再以半透明混合主線，做出附魔感
                    annotated_image = cv2.addWeighted(annotated_image, 1.0, glow_overlay, 0.68 + 0.28 * pulse, 0)
                    annotated_image = cv2.addWeighted(annotated_image, 1.0, line_overlay, 0.54, 0)

                spark_particles = update_and_draw_sparks(
                    spark_particles,
                    spark_core_overlay,
                    spark_glow_overlay,
                    annotated_image.shape[1],
                    annotated_image.shape[0],
                )
                spark_glow_overlay = cv2.GaussianBlur(spark_glow_overlay, (0, 0), 4)
                spark_glow_overlay = cv2.GaussianBlur(spark_glow_overlay, (0, 0), 8)
                annotated_image = cv2.addWeighted(annotated_image, 1.0, spark_glow_overlay, 0.68, 0)
                annotated_image = cv2.addWeighted(annotated_image, 1.0, spark_core_overlay, 0.5, 0)

                # 顯示處理後的影像
                cv2.imshow('MediaPipe 手部地標偵測', annotated_image)

                # 按下 'q' 鍵退出
                if cv2.waitKey(1) == 27:
                    break
            
            cap.release()
            cv2.destroyAllWindows()

    except Exception as e:
        print(f"發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
