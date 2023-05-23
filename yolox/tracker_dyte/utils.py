import os
import cv2
import time
import math
import torch
import pickle
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
from yolox.tracker import matching
from sklearn.cluster import DBSCAN
from IPython.display import clear_output, display, HTML

class IOUCLRTracker(object):
    def __init__(self, iou_threshold=0.25, max_lost=10):
        self.iou_threshold = iou_threshold

    def fit_predict(self, dets):
        edges = []
        detection_dists = matching.cious(dets, dets)
        for det_i in range(len(dets)):
            detection_dist = detection_dists[det_i]
            for det_j, dist in enumerate(detection_dist):
                if det_j <= det_i:
                    continue
                if dist > self.iou_threshold:
                    edges.append([det_i, det_j])
        clusters, cluster_remap = {}, {}
        if len(edges) > 0:
            for edge in edges:
                p1, p2 = edge
                find = False
                for cluster in clusters:
                    ps = clusters[cluster]
                    if p1 in ps or p2 in ps:
                        ps.add(p1)
                        ps.add(p2)
                        find = True
                        cluster_remap[p1] = cluster
                        cluster_remap[p2] = cluster
                        break
                if not find:
                    clusters[len(clusters)] = {p1, p2}
                    cluster_remap[p1] = len(clusters) - 1
                    cluster_remap[p2] = len(clusters) - 1
        print("clusters:", clusters)
        _label = []
        for det_i in range(len(dets)):
            if det_i not in cluster_remap:
                clusters[len(clusters)] = {det_i}
                cluster_remap[det_i] = len(clusters) - 1
            _label.append(cluster_remap[det_i])
        return _label

sampling_rate = 10
video_name = "MOT17-13-FRCNN"

data_dir = f"./datasets/mot_{sampling_rate}/train/{video_name}/"
gt_path = os.path.join(data_dir, "gt/gt.txt")
img_dir = os.path.join(data_dir, "img1")

key_frame_dict = {}
gt_frame_dict, gt_objectid_dict = {}, {}

with open(gt_path, "r") as f:
    data = f.readlines()
    for line in tqdm(data):
        frame_id, object_id, x, y, w, h, is_activate, _, conf = line.rstrip("\n").split(',')
        frame_id, object_id, x, y, w, h, is_activate, conf = int(frame_id), int(object_id), float(x), float(y), float(w), float(h), int(is_activate), float(conf)

        if is_activate < 1.0:
            continue
        
        if frame_id not in gt_frame_dict:
            gt_frame_dict[frame_id] = []
        gt_frame_dict[frame_id].append([object_id, x, y, w, h])
        
        if object_id not in gt_objectid_dict:
            gt_objectid_dict[object_id] = []
            if frame_id not in key_frame_dict:
                key_frame_dict[frame_id] = []
            key_frame_dict[frame_id].append(object_id)
        
        gt_objectid_dict[object_id].append([frame_id, x, y, w, h])

key_frame_ids = sorted(list(key_frame_dict.keys()))
key_frame_ids.remove(1)

for key_frame_id in key_frame_ids:
    object_ids = key_frame_dict[key_frame_id]
    
    now_frame = key_frame_id
    next_frame = key_frame_id + 1
    
    now_detections = gt_frame_dict[now_frame]
    next_detections = gt_frame_dict[next_frame]
    
    img_path = os.path.join(img_dir, "%06d.jpg" % key_frame_id)
    img = cv2.imread(img_path)
    now_background_img = np.zeros_like(img)
    next_background_img = np.zeros_like(img)
    clear_output(wait=True)
    
    model = IOUCLRTracker()
    now_detections_data = [[x, y, x + w, y + h] for _, x, y, w, h in now_detections]
    now_detections_label = model.fit_predict(now_detections_data)
    label_set = set(now_detections_label)
    colors = {}
    for label in list(label_set):
        color = [random.choice(range(256)) for _ in range(3)]
        if label == -1:
            colors[label] = (255, 255, 255)
            continue
        colors[label] = color
    text = "Object: %d Frame: %d" % (object_id, int(key_frame_id))
    cv2.putText(now_background_img, text, (0,25), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 2)
    for now_detection, now_detection_label in zip(now_detections, now_detections_label):
        object_id, x, y, w, h = now_detection
        color = colors[now_detection_label]
        if object_id in object_ids:
            color = (255, 0, 0)
        now_background_img = cv2.rectangle(now_background_img, 
                                           (int(x), int(y)), 
                                           (int(x + w), int(y + h)), 
                                           color, 2)
    
    # model = DBSCAN(eps=150, min_samples=1)
    next_detections_data = [[x, y, x+w, y+h] for _, x, y, w, h in next_detections]
    next_detections_label = model.fit_predict(next_detections_data)
    label_set = set(next_detections_label)
    colors = {}
    for label in list(label_set):
        color = [random.choice(range(256)) for _ in range(3)]
        if label == -1:
            colors[label] = (255, 255, 255)
            continue
        colors[label] = color
    text = "Object: %d Frame: %d" % (object_id, int(key_frame_id))
    cv2.putText(next_background_img, text, (0,25), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 2)
    for next_detection, next_detection_label in zip(next_detections, next_detections_label):
        object_id, x, y, w, h = next_detection
        color = colors[next_detection_label]
        if object_id in object_ids:
            color = (255, 0, 0)
        next_background_img = cv2.rectangle(next_background_img,
                                            (int(x), int(y)),
                                            (int(x + w), int(y + h)),
                                            color, 2)
    
    now_img = Image.fromarray(now_background_img)
    display(now_img)
    
    next_img = Image.fromarray(next_background_img)
    display(next_img)
    
    time.sleep(0.1)