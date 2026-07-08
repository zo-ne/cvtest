import cv2
import numpy as np

class HSVFilter:
    def __init__(self, img):
        """
        初始化HSV颜色过滤器
        :param img: BGR图像
        """
        self.img = img
        self.hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 默认HSV范围
        self.lowerb = np.array([0, 0, 0])
        self.upperb = np.array([179, 255, 255])
        
        # 创建窗口
        self.window_name = 'HSV Filter'
        cv2.namedWindow(self.window_name)
        
        # 创建滑块
        cv2.createTrackbar('H_min', self.window_name, 0, 179, self.on_trackbar)
        cv2.createTrackbar('S_min', self.window_name, 0, 255, self.on_trackbar)
        cv2.createTrackbar('V_min', self.window_name, 0, 255, self.on_trackbar)
        cv2.createTrackbar('H_max', self.window_name, 179, 179, self.on_trackbar)
        cv2.createTrackbar('S_max', self.window_name, 255, 255, self.on_trackbar)
        cv2.createTrackbar('V_max', self.window_name, 255, 255, self.on_trackbar)
        
        self.show()
    
    def on_trackbar(self, val):
        """滑块回调函数"""
        pass
    
    def show(self):
        """显示HSV过滤窗口"""
        while True:
            # 获取滑块值
            h_min = cv2.getTrackbarPos('H_min', self.window_name)
            s_min = cv2.getTrackbarPos('S_min', self.window_name)
            v_min = cv2.getTrackbarPos('V_min', self.window_name)
            h_max = cv2.getTrackbarPos('H_max', self.window_name)
            s_max = cv2.getTrackbarPos('S_max', self.window_name)
            v_max = cv2.getTrackbarPos('V_max', self.window_name)
            
            # 更新范围
            self.lowerb = np.array([h_min, s_min, v_min])
            self.upperb = np.array([h_max, s_max, v_max])
            
            # 创建掩模
            mask = cv2.inRange(self.hsv, self.lowerb, self.upperb)
            
            # 应用掩模到原始图像
            result = cv2.bitwise_and(self.img, self.img, mask=mask)
            
            # 显示结果和原始图像
            display = np.hstack([self.img, result])
            cv2.imshow(self.window_name, display)
            
            # 按ESC或Q键退出
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
        
        cv2.destroyWindow(self.window_name)
