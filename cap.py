import cv2
import numpy as np
import time

# ROI 工具變數
roi_start = None
roi_end = None
selecting_roi = False
prev_roi_region = None
threshold = 10  # 異動閾值 (%)
red_start_time = None  # 記錄變紅的時間
red_duration = 3  # 維持紅色3秒

def mouse_callback(event, x, y, flags, param):
    global roi_start, roi_end, selecting_roi
    
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_start = (x, y)
        selecting_roi = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if selecting_roi and roi_start:
            roi_end = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        roi_end = (x, y)
        selecting_roi = False

def calculate_change_percentage(prev_roi, curr_roi):
    """計算ROI區域的異動百分比"""
    if prev_roi is None or curr_roi is None:
        return 0
    
    if prev_roi.shape != curr_roi.shape:
        return 0
    
    # 計算兩幀差異
    diff = cv2.absdiff(prev_roi, curr_roi)
    
    # 計算平均異動
    mean_diff = np.mean(diff)
    
    # 轉換為百分比 (0-255 映射到 0-100%)
    change_percentage = (mean_diff / 255.0) * 100
    
    return change_percentage

# 開啟鏡頭 (加上我們剛剛學會的 DirectShow 驅動)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 檢查硬體有沒有連接成功
if not cap.isOpened():
    print("❌ 錯誤：找不到或打不開你的攝影機！")
else:
    print("✅ 攝影機運作中，準備顯示鏡像畫面...")
    print("📍 使用滑鼠拖曳來選擇ROI區域，按Esc鍵離開")
    
    cv2.namedWindow("Webcam Mirror")
    cv2.namedWindow("Control")
    cv2.setMouseCallback("Webcam Mirror", mouse_callback)
    
    # 創建trackbar調整閾值
    def on_threshold_change(val):
        global threshold
        threshold = val
    
    cv2.createTrackbar("異動閾值(%)", "Control", threshold, 50, on_threshold_change)

    while True:
        # 1. 讀取鏡頭的原始畫面
        ret, frame = cap.read()
        
        # 檢查有沒有成功拿到畫面數據
        if not ret:
            print("無法收到鏡頭畫面")
            break
            
        # 2. 【在這裡加工你的畫面】
        frame_mirror = cv2.flip(frame, 1)
        
        # 3. 繪製 ROI 矩形
        display_frame = frame_mirror.copy()
        if roi_start and roi_end:
            x1, y1 = roi_start
            x2, y2 = roi_end
            min_x, max_x = min(x1, x2), max(x1, x2)
            min_y, max_y = min(y1, y2), max(y1, y2)
            
            # 提取 ROI 區域
            roi_region = frame_mirror[min_y:max_y, min_x:max_x]
            
            if roi_region.size > 0:
                # 計算異動百分比
                change_pct = calculate_change_percentage(prev_roi_region, roi_region)
                
                # 獲取當前時間
                current_time = time.time()
                
                # 判斷是否應該變紅或保持紅色
                should_be_red = False
                
                if change_pct > threshold:
                    # 異動超過閾值，變紅
                    should_be_red = True
                    red_start_time = current_time
                elif red_start_time is not None and (current_time - red_start_time) < red_duration:
                    # 異動恢復，但未超過3秒，保持紅色
                    should_be_red = True
                else:
                    # 3秒已過，或未曾變紅，變綠
                    should_be_red = False
                    red_start_time = None
                
                # 根據狀態決定顏色
                if should_be_red:
                    color = (0, 0, 255)  # 紅框
                    status = "異動"
                else:
                    color = (0, 255, 0)  # 綠框
                    status = "正常"
                
                # 繪製矩形
                cv2.rectangle(display_frame, (min_x, min_y), (max_x, max_y), color, 2)
                
                # 顯示異動信息
                text = f"{status}: {change_pct:.1f}% (閾值: {threshold}%)"
                cv2.putText(display_frame, text, (min_x, min_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # 如果正在保持紅色，顯示倒計時
                if should_be_red and red_start_time is not None:
                    remaining_time = red_duration - (current_time - red_start_time)
                    if remaining_time > 0:
                        countdown_text = f"紅色維持: {remaining_time:.1f}s"
                        cv2.putText(display_frame, countdown_text, (min_x, max_y + 25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # 保存當前ROI以供下一幀比較
                prev_roi_region = roi_region.copy()
                
                # 顯示 ROI 區域
                cv2.imshow("ROI Region", roi_region)
        
        # 4. 【在這裡顯示加工後的畫面】
        cv2.imshow("Webcam Mirror", display_frame)
        
        # 按 Esc 鍵 (27) 離開
        if cv2.waitKey(1) == 27:
            break

# 釋放資源
cap.release()
cv2.destroyAllWindows()