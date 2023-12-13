import os
import random
import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy

import cv2
import lap
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from cython_bbox import bbox_overlaps as bbox_ious

from yolox.sr_tracker.sr_tracker import SRTracker

def matchtid2gid(img_size, height, width, outputs, scene_data, gt_frameid, sts):
    scale = min(img_size[0] / float(height), img_size[1] / float(width))
    bboxes = outputs.cpu().numpy()[:, :4] / scale
    before_detection_of_sts = np.array([bboxes[st.did] for st in sts], dtype=np.float)
    gt_infos = scene_data[gt_frameid]
    trackids = list(gt_infos.keys())
    before_detection_of_gts = np.array([gt_infos[trackid] for trackid in trackids], dtype=np.float)
    if len(before_detection_of_sts) == 0 or len(before_detection_of_gts) == 0:
        return {}
    iou_matrix = bbox_ious(before_detection_of_sts, before_detection_of_gts)
    
    cost, x, y = lap.lapjv(1.0 - iou_matrix, extend_cost=True, cost_limit=0.9)
    matches = []
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    tid2gid = {}
    for stid, gtid in matches:
        tid2gid[sts[stid].track_id] = [trackids[gtid], gt_infos[trackids[gtid]]]
    return tid2gid

def draw_rects(data, ax, edgecolor):
    for xmin, ymin, xmax, ymax in data:
        rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1, edgecolor=edgecolor, facecolor='none')
        ax.add_patch(rect)

parser = argparse.ArgumentParser("The Training of pk and rk")
parser.add_argument("--data_name", type=str, default="mot_1")
parser.add_argument("--suffix", type=str, default="FRCNN")
parser.add_argument("--tracker_path", type=str, default="./yolox/sr_tracker/result/model_13000.pth", help="tracker model path")
parser.add_argument("--track_thresh", type=float, default=0.6, help="tracking confidence threshold")
parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
parser.add_argument("--min-box-area", type=float, default=100, help='filter out tiny boxes')
parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
parser.add_argument("-dfn", "--data_folder_name", default="mot", type=str, help="data folder name")
parser.add_argument('-s', '--sampling_rate', help="Target sampling rate", type=int, default=1)
args = parser.parse_args()

def get_datainfo(args):
    frameboundInfo = {}
    widthheightInfo = {}
    data_dir = os.path.join("./datasets", args.data_name, "train")
    scene_list = list(filter(lambda x: x.split("-")[-1] == args.suffix, os.listdir(data_dir)))

    for scene in scene_list:
        scene_image_dir = os.path.join(data_dir, scene, "img1")
        scene_det_dir = os.path.join(data_dir, scene, "yoloxdet")
        detection_files = glob(os.path.join(scene_det_dir, "*.pkl"))
        image = cv2.imread(glob(os.path.join(scene_image_dir, "*.jpg"))[0])
        width, height = image.shape[1], image.shape[0]
        framebound = [9999, -9999]
        for detection_file in detection_files:
            frame = int(detection_file.split("/")[-1].split(".")[0])
            framebound[0] = min(framebound[0], frame)
            framebound[1] = max(framebound[1], frame)
        frameboundInfo[scene] = framebound
        widthheightInfo[scene] = [width, height]
    
    return {"data_dir": data_dir,
            "frameboundInfo": frameboundInfo,
            "widthheightInfo": widthheightInfo,
            "scene_list": scene_list}

def make_dataset(args, det_file):
    data_dir = os.path.join("./datasets", args.data_name, "train")
    scene_list = list(filter(lambda x: x.split("-")[-1] == args.suffix, os.listdir(data_dir)))

    maked_dataset = {}
    for scene in scene_list:
        scene_data, track_data = {}, {}
        frame_remap = {}
        scene_dir = os.path.join(data_dir, scene, "gt", det_file)
        scene_image_dir = os.path.join(data_dir, scene, "img1")
        image = cv2.imread(glob(os.path.join(scene_image_dir, "*.jpg"))[0])
        width, height = image.shape[1], image.shape[0]
        
        with open(scene_dir, "r") as f:
            lines = f.readlines()
            for line in lines:
                frameid, trackid, xmin, ymin, w, h, confidence, classId, visibility = line.rstrip("\n").split(",")
                frameid, trackid, xmin, ymin, w, h, confidence, classId, visibility = int(frameid), int(trackid), int(xmin), int(ymin), int(w), int(h), int(confidence), int(classId), float(visibility)
                if confidence < 1:
                    continue
                if frameid not in frame_remap:
                    frame_remap[frameid] = len(frame_remap)
                frameid = frame_remap[frameid]
                if frameid not in scene_data:
                    scene_data[frameid] = {}
                if trackid not in scene_data[frameid]:
                    scene_data[frameid][trackid] = [xmin, 
                                                    ymin,
                                                    xmin + w,
                                                    ymin + h]
                if trackid not in track_data:
                    track_data[trackid] = {}
                if frameid not in track_data[trackid]:
                    track_data[trackid][frameid] = [xmin, 
                                                    ymin,
                                                    xmin + w,
                                                    ymin + h]

        maked_dataset[scene] = [scene_data, track_data, width, height]
    return maked_dataset

def get_dataset(args, datasetRaw,
                datasetInfo, samples=10):
    img_size = (800, 1440)
    data_dir = datasetInfo["data_dir"]
    scene_list = datasetInfo["scene_list"]
    
    output_dataset = {}
    for scene in scene_list:
        scene_data, track_data, width, height = datasetRaw[scene]
        sample_rate = random.randint(1, samples)
        args.sampling_rate = sample_rate
        tracker = SRTracker(args)
        framebound = datasetInfo["frameboundInfo"][scene]
        width, height = datasetInfo["widthheightInfo"][scene]
        scene_det_dir = os.path.join(data_dir, scene, "yoloxdet")
        start_frame = random.randint(framebound[0], framebound[1])
        sts = []

        pk_trajectory = {}
        
        for frameid in range(start_frame, framebound[1], sample_rate):
            if frameid != start_frame:
                tid2gid = matchtid2gid(img_size, height, width, outputs, scene_data, 
                                       frameid - sample_rate - framebound[0], sts)
                            
            outputs = pickle.load(open(os.path.join(scene_det_dir, "{:06d}.pkl".format(frameid)), "rb"))
            info_imgs = [height, width]
            sts, predict_means, hiddens, cells = tracker.update(outputs, info_imgs,
                                                                img_size, 
                                                                output_predicted=True)
            
            if frameid != start_frame:
                newtid2gid = matchtid2gid(img_size, height, width, outputs, scene_data, 
                                          frameid - framebound[0], sts)

                for tid2 in newtid2gid:
                    for tid1 in tid2gid:
                        if tid2 == tid1:
                            x1, y1, x2, y2 = newtid2gid[tid2][1]
                            w, h = x2 - x1, y2 - y1
                            xc = x1 + w / 2
                            lastx1, lasty1, lastx2, lasty2 = tid2gid[tid1][1]
                            lastw, lasth = lastx2 - lastx1, lasty2 - lasty1
                            lastxc = lastx1 + lastw / 2
                            pos = [x1, y1, w, h,
                                   xc, y2]
                            pos += [x1 - lastx1, y1 - lasty1, w - lastw, h - lasth,
                            xc - lastxc, y2 - lasty2,
                            (x1 - lastx1) / sample_rate, (y1 - lasty1) / sample_rate, (w - lastw) / sample_rate, (h - lasth) / sample_rate,
                            (xc - lastxc) / sample_rate, (y2 - lasty2) / sample_rate,
                            sample_rate / samples, sample_rate / samples]
                            if tid1 not in pk_trajectory:
                                pk_trajectory[tid1] = {}
                            if frameid - sample_rate - framebound[0] not in pk_trajectory[tid1]:
                                pk_1 = [0] * 20
                                pk = [pos[i] - predict_means[tid1][i] for i in range(20)]
                                pk_trajectory[tid1][frameid - framebound[0]] = [pk_1, hiddens[tid2], pk]
                            else:
                                pk_1 = pk_trajectory[tid1][frameid - sample_rate - framebound[0]]
                                pk = [pos[i] - predict_means[tid1][i] for i in range(20)]
                                pk_trajectory[tid1][frameid - framebound[0]] = [pk_1, hiddens[tid2], pk]
                            
            tid2gid = {}
            
        output_dataset[scene] = pk_trajectory
    
    return output_dataset
        
datasetInfo = get_datainfo(args)
datasetRaw = make_dataset(args, "gt_val_half.txt")
datasets = get_dataset(args, datasetRaw,
                       datasetInfo)