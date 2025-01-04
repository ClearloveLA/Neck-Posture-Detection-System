# Interactive Computer Vision Cervical Spine Prevention System 


 [Dingxi Zhang](https://kristen-z.github.io/), [Mengyin Lin](https://github.com/Cassie-Lim), Zhuoxun Chen, Hang Wang, Boling Zhai, Tianyu Zheng

This is a project for Human Computer Interaction Course at University of Chinese Academy of Sciences. We designed an efficient system to detect cervical posture in real-time, provide timely feedback to remind users to pose correctly, and an interactive game to guide the user to exercise their cervical spine scientifically. 

<img src="./resources/pose.gif" alt="pose" style="zoom:150%;" />

<p align="left">
    <a href="https://github.com/Cassie-Lim/pose_eval/tree/plus/Report/report.pdf">
        <img src='https://img.shields.io/badge/Paper-PDF-green?style=for-the-badge&logo=adobeacrobatreader&logoWidth=20&logoColor=white&labelColor=66cc00&color=94DD15' alt='Paper PDF'>
    </a>
</p>

**Abstract**: As people spend more and more time operating electronic devices such as computers, mobile phones, and tablet computers, their necks need to be kept in a fixed position for a long time. Most of the existing cervical spondylosis prevention products on the market are aimed at relieving and treating the pain and discomfort in the neck, rather than considering the neck habits of the user from a preventive perspective to avoid neck discomfort. Most prevention products are also costly and technically complex. Based on focus groups and questionnaires, this paper designs a system based on computer vision technology, which extracts the user's neck posture by monitoring the angle of the head and neck in real-time and provides timely feedback to remind the user to correct the incorrect posture, as well as an interactive game to guide the user in scientific activities to exercise the cervical spine. It is simple, portable, and user-friendly. The system is divided into three main modules, a **pose detection module** based on Lightweight Openpose, a **feedback reminder module** that generates visual and auditory feedback, and a **interactive game module** that guides the user to exercise according to professional cervical relaxation exercises. To verify the effectiveness and benefits of the system, we invited 55 subjects to evaluate the system and found that it helps users to alleviate cervical discomfort and develop good neck habits and that the gamified approach is effective in helping users to stick to the exercise and move their cervical spine.

**Keywords:** Cervical spine prevention, Posture detection, Lightweight Openpose, Gamification, Interactive guidance, Human Computer Interaction

### Requirements

- Create conda environment

```bash
conda create -n pose python=3.7
conda activate pose
```

- Installing other dependencies


```bash
pip install -r requirements.txt
```

> Note: If you haven't downloaded the Build Tools for Visual Studio, you may encounter difficulties installing pycocotools (as it doesn't have pre-built wheels and requires source code compilation). You can refer to the following links for installation:
>
> [error: Microsoft Visual C++ 14.0 is required问题解法_alicee_2012的博客-CSDN博客](https://blog.csdn.net/alicee_2012/article/details/122726986?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0-122726986-blog-89399825.pc_relevant_aa2&spm=1001.2101.3001.4242.1&utm_relevant_index=2)
>
> [错误：cl: 命令行 error D8021 :无效的数值参数“/Wno-cpp”_点亮～黑夜的博客-CSDN博客_cl: 命令行 error d8021 :无效的数值参数“/wno-cpp”](https://blog.csdn.net/weixin_41010198/article/details/94053130)

### Usage

#### Pose detection

We mainly based on [Lightweight OpenPose](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch) framework to detect the cervical pose. 

```bash
python demo.py [-h] [--checkpoint-path CHECKPOINT_PATH]                     
               [--height-size HEIGHT_SIZE] [--video VIDEO] [--remind REMIND]
               [--images IMAGES [IMAGES ...]] [--cpu] [--track TRACK]        
               [--smooth SMOOTH]
optional arguments:
  -h, --help            show this help message and exit
                        network input layer height size
  --video VIDEO         path to video file or camera id
  --remind REMIND       whether to enable voice reminder
  --images IMAGES [IMAGES ...]
                        path to input image(s)
  --cpu                 run network inference on cpu
  --track TRACK         track pose id in video
  --smooth SMOOTH       smooth pose keypoints
```

Additional explanations for important parameters:

- --video:

  You can use the `video` parameter to specify the input video file or camera ID (if it's an internal camera, set the ID to 1; if it's an external camera, you can refer to [this article](https://blog.csdn.net/babybin/article/details/122044565) to obtain the device ID).

  Of course, you can directly run `python demo.py` to use our provided default test case `video.mp4`.

- --cpu:

  We default the CPU option to enable. If you have an NVIDIA graphics card on your computer, you can use `--cpu False` to achieve better computing performance.

#### Interactive game & reminder module

We based on [tkinter](https://docs.python.org/3/library/tkinter.html) to provide timely feedback to users to correct their neck pose and [pygame](https://www.pygame.org/) to develop the interactive game. We use message queue to communicate between the detection, game and reminder thread to achieve real-time interaction with users. To enable interactive modules based on pose detection, run:

```bash
python main.py
```

For more technique details and user study experiments, pleaser refer to our [report](https://github.com/Cassie-Lim/pose_eval/tree/plus/Report/report.pdf).

## Acknowledgement

This repo is heavily based on [Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch) to detect human neck poses. Part of the code is borrowed from [Pac-Men_code](https://blog.csdn.net/dQCFKyQDXYm3F8rB0/article/details/106934740).

We express sincere gratitude to Prof. Tian Feng and Mr. Sun Wei from the Human-Computer Interaction course for their valuable advice on our project.
