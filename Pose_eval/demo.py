import cv2
import numpy as np
import torch
import math
import threading
import time
from datetime import datetime
import winsound
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from modules.keypoints import BODY_PARTS_KPT_IDS
from val import normalize, pad_width
import os

def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad

class myDetect(threading.Thread):
    def __init__(self, name, queue):
        threading.Thread.__init__(self, name=name)
        self.mailbox = queue
        self.frame = None
        self.running = True
        self.status = "未开始"
        self.angle = 0
        self.calibrating = True
        self._calibration_frames = 0
        self.nose_neck_len = 0.0
        self.round = 30
        self.remind_interval = 30
        self.sound_enabled = True
        self.last_remind_time = time.time()
        self.net = PoseEstimationWithMobileNet()
        checkpoint = torch.load('models/checkpoint_iter_370000.pth', 
                              map_location='cpu',
                              weights_only=True)
        load_state(self.net, checkpoint)
        self.net = self.net.eval()
        print("模型加载成功!")

    def calc_angle(self, p1, p2, cross):
        vec1 = np.array(p1 - cross)
        vec2 = np.array(p2 - cross)
        l1 = np.sqrt(vec1.dot(vec1))
        l2 = np.sqrt(vec2.dot(vec2))
        angle = np.arccos(vec1.dot(vec2) / (l1 * l2))
        return math.degrees(angle)
        
    def check_pose(self, nose, neck):
        if self.calibrating:
            if self._calibration_frames == 0:
                print("正在校准，请正对并平视镜头……( •̀ ω •́ )✧")
                self._calibration_frames += 1
                return
                
            if self._calibration_frames <= self.round:
                vec = nose - neck
                length = np.sqrt(vec.dot(vec))
                self.nose_neck_len += length
                self._calibration_frames += 1
                
                if self._calibration_frames == self.round + 1:
                    self.nose_neck_len /= self.round
                    self.calibrating = False
                    print("校准完成!")
                return
                
        vec = nose - neck
        length = np.sqrt(vec.dot(vec))
        ratio = length / self.nose_neck_len
        
        self.angle = ratio * 90
        if ratio > 1.2:
            self.status = "抬头"
        elif ratio < 0.8:
            self.status = "低头"
        else:
            self.status = "正常"
            
        self.check_and_remind()
            
    def check_and_remind(self):
        if not self.sound_enabled:
            return
            
        current_time = time.time()
        if current_time - self.last_remind_time >= self.remind_interval:
            if abs(self.angle) > 30:
                winsound.Beep(1000, 500)
                self.last_remind_time = current_time

    def set_remind_interval(self, seconds):
        self.remind_interval = seconds
        
    def toggle_sound(self, enabled):
        self.sound_enabled = enabled
        
    def stop(self):
        self.running = False
        cv2.destroyAllWindows()
        
    def run(self):
        height_size = 256
        stride = 8
        upsample_ratio = 4
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
                
            self.frame = frame
            
            heatmaps, pafs, scale, pad = infer_fast(self.net, frame, height_size, stride, upsample_ratio, True)
            
            total_keypoints_num = 0
            all_keypoints_by_type = []
            for kpt_idx in range(18):
                total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

            pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
            
            for kpt_id in range(all_keypoints.shape[0]):
                all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
                all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
            
            if len(pose_entries) > 0:
                pose_keypoints = np.ones((18, 2), dtype=np.int32) * -1
                for kpt_id in range(18):
                    if pose_entries[0][kpt_id] != -1.0:
                        pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[0][kpt_id]), 0])
                        pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[0][kpt_id]), 1])
                
                nose = pose_keypoints[0]
                neck = pose_keypoints[1]
                if nose[0] != -1 and neck[0] != -1:
                    self.check_pose(nose, neck)
                    cv2.circle(frame, (nose[0], nose[1]), 3, (0, 255, 0), -1)
                    cv2.circle(frame, (neck[0], neck[1]), 3, (0, 255, 0), -1)
                    cv2.line(frame, (nose[0], nose[1]), (neck[0], neck[1]), (255, 0, 0), 2)
            
            cv2.putText(frame, f"状态: {self.status}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"角度: {self.angle:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Pose Detection', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
                
        cap.release()
        cv2.destroyAllWindows()