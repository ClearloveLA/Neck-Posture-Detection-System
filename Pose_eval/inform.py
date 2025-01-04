import threading
import time
from tkinter import *
import tkinter
import tkinter.simpledialog


from pacman_main import myGame
import share_var

import keypoints

inform_limit = 10


class inform_you(threading.Thread):
    def __init__(self, name, queue):
        threading.Thread.__init__(self, name=name)
        self.mailbox = queue
    def run(self):
        keep_time = 0
        global inform_limit
        top = tkinter.Tk()
        top.geometry('500x100')
        top.title("自定义提醒时间")
        entry = Entry(top,width=20)
        entry.pack()

        def setting():
            var = entry.get()		# 调用get()方法，将Entry中的内容获取出来
            print(var)
            global inform_limit
            inform_limit = int(var)
            top.destroy()
        Button(top,text='请输入低头时限/s',command=setting).pack()
        top.mainloop()
        print("Ready to inform you now")
        while(1):
            time.sleep(1)
            a = share_var.getMotionData()
            if(a[1] > 0.1):
                print("呀，怎么头低下去这么多~")
                keep_time = keep_time + 1
            
            if(not judge_lean_forward(keypoints.keypoints)):
                print("呀，怎么身体前倾下去这么多~")
                keep_time = keep_time + 1

            if(not judge_crooked(keypoints.keypoints)):
                print("呀，怎么身体歪斜这么多~")
                keep_time = keep_time + 1#/
            
            if(keep_time > inform_limit):
                keep_time = 0
                print("Ready to start a game now?[Y/N]")
                reply = tkinter.messagebox.askyesno("提醒时间到！","低头太久啦，来伸展一下颈椎？")
                if(reply):
                    game = myGame("game", self.mailbox)
                    game.start()
                    game.join()

def judge_lean_forward(keypoint):
    eye_distance=abs(keypoint[14][0]-keypoint[15][0])
    wrist_eye_y_distance=abs((keypoint[4][1]+keypoint[7][1])/2
                            -(keypoint[14][1]+keypoint[15][1])/2)
    if (wrist_eye_y_distance/eye_distance)>5:
        return True
    else:
        return False

def judge_crooked(keypoint):
    nose_x=keypoint[0][0]
    l_sho_x=keypoint[5][0]
    r_sho_x=keypoint[2][0]
    nose_l_sho_distance=abs(nose_x-l_sho_x)
    nose_r_sho_distance=abs(nose_x-r_sho_x)
    if nose_l_sho_distance/nose_r_sho_distance>1.25 or nose_r_sho_distance/nose_l_sho_distance>1.25:
        return False
    else:
        return True
