import cv2
import os

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 创建KNN背景去除器
bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)

# 打开摄像头（使用DirectShow驱动，Windows下速度更快）
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 设置摄像头分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 225)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

print("摄像头已启动，按 ESC 键退出")

while True:
    # 读取视频帧
    ret, frame = cap.read()
    
    if not ret:
        print("无法读取摄像头")
        break
    
    # 调整帧大小
    frame = cv2.resize(frame, (400, 225))
    
    # 水平翻转（镜像效果）
    frame = cv2.flip(frame, 1)
    
    # 应用KNN背景去除
    gray = bs.apply(frame)
    
    # 二值化
    mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]
    
    # 腐蚀操作（去除噪声）
    mask = cv2.erode(mask, None, iterations=2)
    
    # 膨胀操作（填充空洞）
    mask = cv2.dilate(mask, None, iterations=2)
    
    # 转换为BGR彩色图像以便显示
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # 水平拼接：左边原始图像，右边处理后的图像
    frame = cv2.hconcat([frame, mask])
    
    # 显示
    cv2.imshow('KNN 背景去除 - 左: 原始 | 右: 处理结果', frame)
    
    # 按ESC键退出（27是ESC的ASCII码）
    if cv2.waitKey(10) == 27:
        print("已退出")
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
