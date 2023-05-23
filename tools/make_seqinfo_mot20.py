import os
import configparser
from glob import glob

# [Sequence]
# name=MOT17-02-FRCNN
# imDir=img1
# frameRate=30
# seqLength=600
# imWidth=1920
# imHeight=1080
# imExt=.jpg

gt_root = "./datasets/mot20_{srate}"

for i in range(1, 11):
    some_rate_gt_dir = gt_root.format(srate=i)
    some_rate_gt_dir = os.path.join(some_rate_gt_dir, "train")

    video_path_list = glob(os.path.join(some_rate_gt_dir, "MOT20-*"))
    for video_path in video_path_list:
        gt_path = os.path.join(video_path, "gt/gt_val_half.txt")
        min_f, max_f = 9999, -9999
        with open(gt_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip("\n")
                frameid = int(line.split(",")[0])
                min_f = min(min_f, frameid)
                max_f = max(max_f, frameid)

        config=configparser.ConfigParser()
        config.read(os.path.join(video_path, "seqinfo.ini"))
        if "Sequence" not in config.sections():
            config.add_section("Sequence")
        config.set("Sequence","seqLength",f"{max_f}")
        config.set("Sequence","frameRate",f"{30//i}")
        config.write(open(os.path.join(video_path, "seqinfo.ini"), "w"))