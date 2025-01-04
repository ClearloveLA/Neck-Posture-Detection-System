import math
import numpy as np
import cv2
import pyttsx3


class Reminder:
    engine = None
    rate = None

    def __init__(self):
        self.engine = pyttsx3.init()

    def remind(self, info):
        self.engine.say(info)
        self.engine.runAndWait()


# track how long user's pose has been out of range
ANGLE_THRESHOLD = 10
TIME_THRESHOLD = 10
INTERVAL = 5
fps = 0
frame_cnt = 0

reminder = Reminder()


def voice_init(input_fps):
    assert input_fps != 0
    global fps
    fps = input_fps


def track_state(angle, img):
    global frame_cnt, fps, reminder
    if angle > ANGLE_THRESHOLD:
        frame_cnt += 1
        if frame_cnt > TIME_THRESHOLD * fps and (frame_cnt - TIME_THRESHOLD * fps) % (INTERVAL * fps) == 0:
            reminder.remind("Hello, please watch your neck!")
    else:
        frame_cnt = 0
    cv2.putText(img,
                "Duration: {:.2f}".format(frame_cnt / fps),
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
