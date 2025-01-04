import cv2
import threading
import numpy as np
from datetime import datetime

class myDetect(threading.Thread):
    def __init__(self, name, queue):
        threading.Thread.__init__(self, name=name)
        self.mailbox = queue
        self.frame = None
        self.running = True
        self.status = "未开始"
        self.angle = 0
        self.calibrating = False
        
    def stop(self):
        self.running = False
        
    def run(self):
        # 打开摄像头
        cap = cv2.VideoCapture(0)
        
        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FPS, 30)
        print("The fps is:", cap.get(cv2.CAP_PROP_FPS))
        
        # 校准阶段
        print("正在校准，请正对并平视镜头……( •̀ ω •́ )✧")
        self.calibrating = True
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
                
            # 在这里添加姿态检测的代码
            # 暂时只显示原始视频
            self.frame = frame
            
            # 更新状态
            self.status = "正常"
            self.angle = 0  # 这里应该是实际检测到的角度
            
            # 显示帧
            cv2.imshow('Pose Detection', frame)
            
            # 按ESC退出
            if cv2.waitKey(1) & 0xFF == 27:
                break
                
        cap.release()
        cv2.destroyAllWindows()