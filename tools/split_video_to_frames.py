import os
import cv2
from glob import glob

input_video_path = "./YOLOX_outputs/extendkalmanfilter6_dancetrack/yolox_x_ablation_5"
video_mp4_paths = glob(os.path.join(input_video_path, "*.mp4"))

for video_path in video_mp4_paths:
    video_name = video_path.split("/")[-1].split(".")[0]
    if not os.path.exists(os.path.join(input_video_path, video_name)):
        os.mkdir(os.path.join(input_video_path, video_name))
    cap = cv2.VideoCapture(video_path)
    frame_id = 1
    success, image = cap.read()
    while success:
        cv2.imwrite(os.path.join(input_video_path, video_name, "%06d.jpg" % frame_id), image)
        success, image = cap.read()
        frame_id += 1