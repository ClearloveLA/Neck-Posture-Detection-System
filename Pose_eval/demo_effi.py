import argparse

import cv2
import numpy as np
import torch
import math
import time
from models.with_effi import PoseEstimationWithEffi
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from modules.voice import voice_init, track_state
from val import normalize, pad_width

import threading
from queue import Queue
import share_var

# up, down, left, right
usrMotion = [0,0,0,0]

noimage = 0



# the detect task send the list to the game thread
class myDetect(threading.Thread):
    def __init__(self, name, queue):
        threading.Thread.__init__(self, name=name)
        self.mailbox = queue
    def run(self):
        parser = argparse.ArgumentParser(
        description='''Lightweight human pose estimation python demo.
                       This is just for quick results preview.
                       Please, consider c++ demo for the best performance.''')
        # parser.add_argument('--checkpoint-path', type=str,
        #                     default="./models/checkpoint_iter_370000.pth ", help='path to the checkpoint')
        parser.add_argument('--checkpoint-path', type=str,
                            default="./models/checkpoint_iter_1000.pth ", help='path to the checkpoint')
        parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')
        # parser.add_argument('--video', type=str, default='0', help='path to video file or camera id')
        parser.add_argument('--video', type=str, default='video.mp4', help='path to video file or camera id')
        parser.add_argument('--remind', default=False, help='whether to enable voice reminder')
        # parser.add_argument('--video', type=str, default='', help='path to video file or camera id')
        parser.add_argument('--images', nargs='+', default='', help='path to input image(s)')
        parser.add_argument('--cpu', default=True, action='store_true', help='run network inference on cpu')
        parser.add_argument('--track', type=int, default=False, help='track pose id in video')
        parser.add_argument('--smooth', type=int, default=1, help='smooth pose keypoints')
        args = parser.parse_args()

        if args.video == '' and args.images == '':
            raise ValueError('Either --video or --image has to be provided')

        net = PoseEstimationWithEffi()
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        load_state(net, checkpoint)
        # model_fp32 = PoseEstimationWithEffi()
        # model_fp32.qconfig = torch.quantization.get_default_qconfig('qnnpack')
        # model_fp32_prepared = torch.quantization.prepare(model_fp32)
        # net = torch.quantization.convert(model_fp32_prepared)
        # net.load_state_dict(torch.load(args.checkpoint_path, map_location='cpu'))


        frame_provider = ImageReader(args.images)
        if args.video != '':
            frame_provider = VideoReader(args.video)
        else:
            args.track = 0

        # 初始化
        voice_init(frame_provider.fps)
        self.run_demo(net, frame_provider, args.height_size, args.cpu, args.track, args.smooth, args.remind)

        # self.mailbox.put(usrMotion)
        share_var.setMotionDate(usrMotion)

        print("detection task exited")

    def run_demo(self, net, image_provider, height_size, cpu, track, smooth, remind):
        net = net.eval()
        if not cpu:
            net = net.cuda()

        stride = 8
        upsample_ratio = 4
        num_keypoints = Pose.num_kpts
        previous_poses = []
        delay = 1
        for img in image_provider:
            orig_img = img.copy()
            heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

            total_keypoints_num = 0
            all_keypoints_by_type = []
            for kpt_idx in range(num_keypoints):  # 19th for bg
                total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

            pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
            for kpt_id in range(all_keypoints.shape[0]):
                all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
                all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
            current_poses = []
            for n in range(len(pose_entries)):
                if len(pose_entries[n]) == 0:
                    continue
                pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
                for kpt_id in range(num_keypoints):
                    if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                        pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                        pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
                pose = Pose(pose_keypoints, pose_entries[n][18])
                current_poses.append(pose)

            if track:
                track_poses(previous_poses, current_poses, smooth=smooth)
                previous_poses = current_poses
            

            for pose in current_poses:
                if(noimage == 0):
                    pose.draw(img)
                # use nose, neck to measure up & down state
                up, down = check_updown(pose.keypoints[0], pose.keypoints[2])

                # use nose, neck, right shoulders to calculate left & right angle
                angle = calc_angle(pose.keypoints[0], pose.keypoints[2], pose.keypoints[1])
                
                if(angle > 90):
                    usrMotion[2] = (angle - 90)/45
                    usrMotion[3] = 0
                else:
                    usrMotion[2] = 0
                    usrMotion[3] = (90 - angle)/45

                # self.mailbox.put(usrMotion)
                share_var.setMotionDate(usrMotion)
                cv2.putText(img,
                            "Neck & shoulder: {:.2f} degree".format(angle),
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if remind:
                    track_state(angle, img)
            img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
            for pose in current_poses:
                # cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                #               (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
                if track:
                    cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
            if(noimage == 0):
                cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
            key = cv2.waitKey(delay)
            if key == 27:  # esc
                return
            elif key == 112:  # 'p'
                if delay == 1:
                    delay = 0
                else:
                    delay = 1


def calc_angle(p1, p2, cross):
    vec1 = np.array(p1 - cross)
    vec2 = np.array(p2 - cross)
    l1 = np.sqrt(vec1.dot(vec1))
    l2 = np.sqrt(vec2.dot(vec2))
    angle = np.arccos(vec1.dot(vec2) / (l1 * l2))
    angle = math.degrees(angle)
    return angle


cal_cnt = 0
nose_neck_len = 0.0
round = 30
def check_updown(nose, neck):
    global cal_cnt, nose_neck_len
    vec = nose - neck
    len = np.sqrt(vec.dot(vec))
    # 前round帧用于校准
    if cal_cnt == 0:
        print("正在校准，请正对并平视镜头……( •̀ ω •́ )✧")
        time.sleep(2)
        cal_cnt += 1
        return 0, 0
    elif cal_cnt <= round:
        nose_neck_len += len
        cal_cnt += 1
        if cal_cnt == round+1:
            nose_neck_len /= round
        return 0, 0
    up = 0
    down = 0
    vec = nose - neck
    len = np.sqrt(vec.dot(vec))
    # get up and done
    # print(len/nose_neck_len)
    up_degree = len/nose_neck_len
    if(up_degree >= 1):
        usrMotion[0] = up_degree - 1
        usrMotion[1] = 0
    else:
        usrMotion[0] = 0
        usrMotion[1] = 1-up_degree

    if len/nose_neck_len > 1.5:
        up = 1
        # print("Up!")
    elif len/nose_neck_len < 0.7:
        down = 1
        # print("Down!")
    return up, down


class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        print('The fps is: {}'.format(self.fps))

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
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




def run_demo(net, image_provider, height_size, cpu, track, smooth, remind):
    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []
    delay = 1
    for img in image_provider:
        orig_img = img.copy()
        heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                     total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)

        if track:
            track_poses(previous_poses, current_poses, smooth=smooth)
            previous_poses = current_poses

        for pose in current_poses:
            if (noimage == 0):
                pose.draw(img)
            # use nose, neck to measure up & down state
            up, down = check_updown(pose.keypoints[0], pose.keypoints[2])

            # use nose, neck, right shoulders to calculate left & right angle
            angle = calc_angle(pose.keypoints[0], pose.keypoints[2], pose.keypoints[1])

            if (angle > 90):
                usrMotion[2] = (angle - 90) / 45
                usrMotion[3] = 0
            else:
                usrMotion[2] = 0
                usrMotion[3] = (90 - angle) / 45

            # self.mailbox.put(usrMotion)
            share_var.setMotionDate(usrMotion)
            cv2.putText(img,
                        "Neck & shoulder: {:.2f} degree".format(angle),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if remind:
                track_state(angle, img)
        img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
        for pose in current_poses:
            # cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
            #               (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
            if track:
                cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        if (noimage == 0):
            cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
        key = cv2.waitKey(delay)
        if key == 27:  # esc
            return
        elif key == 112:  # 'p'
            if delay == 1:
                delay = 0
            else:
                delay = 1


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description='''Lightweight human pose estimation python demo.
                   This is just for quick results preview.
                   Please, consider c++ demo for the best performance.''')
    parser.add_argument('--checkpoint-path', type=str,
                        default="./models/checkpoint_iter_370000.pth ", help='path to the checkpoint')
    # parser.add_argument('--checkpoint-path', type=str,
    #                     default="./models/pose_int.pth ", help='path to the checkpoint')
    parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')
    # parser.add_argument('--video', type=str, default='0', help='path to video file or camera id')
    parser.add_argument('--video', type=str, default='video.mp4', help='path to video file or camera id')
    parser.add_argument('--remind', default=False, help='whether to enable voice reminder')
    # parser.add_argument('--video', type=str, default='', help='path to video file or camera id')
    parser.add_argument('--images', nargs='+', default='', help='path to input image(s)')
    parser.add_argument('--cpu', default=True, action='store_true', help='run network inference on cpu')
    parser.add_argument('--track', type=int, default=False, help='track pose id in video')
    parser.add_argument('--smooth', type=int, default=1, help='smooth pose keypoints')
    args = parser.parse_args()

    if args.video == '' and args.images == '':
        raise ValueError('Either --video or --image has to be provided')

    net = PoseEstimationWithEffi()
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    load_state(net, checkpoint)
    # model_fp32 = PoseEstimationWithMobileNet()
    # model_fp32.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    # model_fp32_prepared = torch.quantization.prepare(model_fp32)
    # net = torch.quantization.convert(model_fp32_prepared)
    # net.load_state_dict(torch.load(args.checkpoint_path, map_location='cpu'))

    frame_provider = ImageReader(args.images)
    if args.video != '':
        frame_provider = VideoReader(args.video)
    else:
        args.track = 0

    # 初始化
    voice_init(frame_provider.fps)
    run_demo(net, frame_provider, args.height_size, args.cpu, args.track, args.smooth, args.remind)
