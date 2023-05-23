import cv2
import numpy as np
from collections import defaultdict
from loguru import logger
from tqdm import tqdm

import torch

from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)
from yolox.utils.visualize import plot_detection, plot_tracking

from yolox.adjust_tracker.adjust_byte_tracker import AdjustTracker
from yolox.history.byte_tracker import HistoryTracker
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracker_dyte.differentiable_byte_tracker import DYTETracker, STrack
from yolox.sort_tracker.sort import Sort
from yolox.deepsort_tracker.deepsort import DeepSort
from yolox.motdt_tracker.motdt_tracker import OnlineTracker
from yolox.tracker_dyte.basetrack import BaseTrack
from yolox.tracker.basetrack import BaseTrack as BaseTrack2

import contextlib
import io
import os
import pickle
import itertools
import json
import tempfile
import time


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, track_dids, scores in results:
            for tlwh, track_id, track_did, score in zip(tlwhs, track_ids, track_dids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1))#, s=round(score, 2), did=track_did)
                f.write(line)
    logger.info('save results to {}'.format(filename))

def write_detection_result(filename, results):
    save_format = '{frame},{did},{x1},{y1},{w},{h},{s},{ifn},-1,-1\n'
    with open(filename, 'w') as f:
        for data in results:
            for frame_id, did, bbox, score, video_name, img_file_name in data:
                if score <= 0.1:
                    continue
                x1, y1, x2, y2 = bbox
                line = save_format.format(frame=frame_id, did=did, x1=str(round(x1, 1)), y1=str(round(y1, 1)), w=str(round(x2 - x1, 1)), h=str(round(y2 - y1, 1)), s=round(score, 2), ifn=img_file_name[0])
                f.write(line)
    logger.info('save detection results to {}'.format(filename))

def write_results_no_score(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class MOTEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self, args, dataloader, img_size, confthre, nmsthre, num_classes):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.args = args

    def evaluate(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        dresults = []
        det_results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
            
        tracker = BYTETracker(self.args)
        ori_thresh = self.args.track_thresh
        for cur_iter, (imgs, raw_imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]
                if video_name == 'MOT17-05-FRCNN' or video_name == 'MOT17-06-FRCNN':
                    self.args.track_buffer = 14
                elif video_name == 'MOT17-13-FRCNN' or video_name == 'MOT17-14-FRCNN':
                    self.args.track_buffer = 25
                else:
                    self.args.track_buffer = 30

                if video_name == 'MOT17-01-FRCNN':
                    self.args.track_thresh = 0.65
                elif video_name == 'MOT17-06-FRCNN':
                    self.args.track_thresh = 0.65
                elif video_name == 'MOT17-12-FRCNN':
                    self.args.track_thresh = 0.7
                elif video_name == 'MOT17-14-FRCNN':
                    self.args.track_thresh = 0.67
                elif video_name in ['MOT20-06', 'MOT20-08']:
                    self.args.track_thresh = 0.3
                else:
                    self.args.track_thresh = ori_thresh

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    BaseTrack2._count = 0
                    tracker = BYTETracker(self.args)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results(result_filename, results)
                        results = []
                        # detection_result_filename = os.path.join(result_folder, '{}-det.txt'.format(video_names[video_id - 1]))
                        # write_detection_result(detection_result_filename, dresults)
                        # dresults = []
                        out.release()
                        cv2.destroyAllWindows()
                    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # 视频编解码器
                    width, height = int(info_imgs[1]), int(info_imgs[0])  # 宽高
                    out = cv2.VideoWriter(result_folder[:-len("track_results")] + '/result{}.mp4'.format(video_names[video_id]), 
                                          fourcc, 30 // self.args.sampling_rate, 
                                          (width, height), isColor=True)
                        

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                with torch.no_grad():
                    outputs = model(imgs)
                    if decoder is not None:
                        outputs = decoder(outputs, dtype=outputs.type())

                    outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # det_results.append([output_results, frame_id, video_name, img_file_name])

            # run tracking
            if outputs[0] is not None:
                online_targets, bboxes, scores = tracker.update(outputs[0], info_imgs, self.img_size, frame_id)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                online_dids = []
                online_prediction_tlwhs = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    did = t.detection_id
                    last_tlwh = t.last_tlwh
                    if self.args.dance:
                        vertical = False
                    else:
                        vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        online_dids.append(did)
                        if len(results) != 0:
                            online_prediction_tlwhs.append(last_tlwh)
                # save results
                results.append((frame_id, online_tlwhs, online_ids, online_dids, online_scores))
                plot_image = plot_tracking(raw_imgs[0], online_tlwhs, online_prediction_tlwhs, online_ids, fps=30 // self.args.sampling_rate, frame_id=frame_id)
                out.write(plot_image)
                
                # online_detections = []
                # for did, (bbox, score) in enumerate(zip(bboxes, scores)):
                #     online_detections.append([frame_id, did, bbox, score, video_name, img_file_name])
                # dresults.append(online_detections)

                # make detections
            #     output_results = outputs[0]
            #     if output_results.shape[1] == 5:
            #         scores = output_results[:, 4]
            #         bboxes = output_results[:, :4]
            #     else:
            #         output_results = output_results.cpu().numpy()
            #         scores = output_results[:, 4] * output_results[:, 5]
            #         bboxes = output_results[:, :4]  # x1y1x2y2
            #     img_h, img_w = info_imgs[0], info_imgs[1]
            #     scale = min(self.img_size[0] / float(img_h), self.img_size[1] / float(img_w))
            #     bboxes /= scale
            #     select_inds = scores > 0.05
            #     dets = bboxes[select_inds]
            #     scores_keep = scores[select_inds]
                
            #     online_im = plot_detection(
            #         raw_imgs[0], dets, list(range(len(dets))), scores=scores_keep, frame_id=frame_id, fps=30.0
            #     )
            # else:
            #     online_im = raw_imgs[0]
            # if not os.path.exists("./visualization/%s"%video_name):
            #     os.makedirs("./visualization/%s"%video_name)
            # cv2.imwrite("./visualization/%s/%06d.jpg"%(video_name, frame_id), online_im)
                
            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results(result_filename, results)
                detection_result_filename = os.path.join(result_folder, '{}-det.txt'.format(video_names[video_id]))
                write_detection_result(detection_result_filename, dresults)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        pickle.dump(det_results, open("/".join(result_folder.split("/")[:-1]) + "/det_results.pkl", "wb"))
        return eval_results
    
    def evaluate_dyte(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        dresults = []
        det_results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1
            
        # tracker = DYTETracker(self.args)
        ori_thresh = self.args.track_thresh
        for cur_iter, (imgs, raw_imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]
                if video_name == 'MOT17-05-FRCNN' or video_name == 'MOT17-06-FRCNN':
                    self.args.track_buffer = 14
                elif video_name == 'MOT17-13-FRCNN' or video_name == 'MOT17-14-FRCNN':
                    self.args.track_buffer = 25
                else:
                    self.args.track_buffer = 30

                if video_name == 'MOT17-01-FRCNN':
                    self.args.track_thresh = 0.65
                elif video_name == 'MOT17-06-FRCNN':
                    self.args.track_thresh = 0.65
                elif video_name == 'MOT17-12-FRCNN':
                    self.args.track_thresh = 0.7
                elif video_name == 'MOT17-14-FRCNN':
                    self.args.track_thresh = 0.67
                elif video_name in ['MOT20-06', 'MOT20-08']:
                    self.args.track_thresh = 0.3
                else:
                    self.args.track_thresh = ori_thresh

                if video_name == 'MOT17-02-FRCNN':
                    self.args.match_thresh_d1 = 0.85
                    self.args.match_thresh_d2 = 0.45
                    self.args.match_thresh_d3 = 0.75
                elif video_name == 'MOT17-04-FRCNN':
                    self.args.match_thresh_d1 = 0.90
                    self.args.match_thresh_d2 = 0.55
                    self.args.match_thresh_d3 = 0.75
                elif video_name == 'MOT17-05-FRCNN':
                    self.args.match_thresh_d1 = 0.80
                    self.args.match_thresh_d2 = 0.50
                    self.args.match_thresh_d3 = 0.75
                elif video_name == 'MOT17-09-FRCNN':
                    self.args.match_thresh_d1 = 0.90
                    self.args.match_thresh_d2 = 0.50
                    self.args.match_thresh_d3 = 0.75
                elif video_name == 'MOT17-10-FRCNN':
                    self.args.match_thresh_d1 = 1.00
                    self.args.match_thresh_d2 = 0.55
                    self.args.match_thresh_d3 = 0.75
                elif video_name == 'MOT17-11-FRCNN':
                    self.args.match_thresh_d1 = 0.80
                    self.args.match_thresh_d2 = 0.50
                    self.args.match_thresh_d3 = 0.65
                elif video_name == 'MOT17-13-FRCNN':
                    self.args.match_thresh_d1 = 1.00
                    self.args.match_thresh_d2 = 0.55
                    self.args.match_thresh_d3 = 0.75

                if video_name == 'MOT17-02-FRCNN':
                    self.args.adjusted_gate = 1.10
                elif video_name == 'MOT17-04-FRCNN':
                    self.args.adjusted_gate = 1.10
                elif video_name == 'MOT17-05-FRCNN':
                    self.args.adjusted_gate = 1.70
                elif video_name == 'MOT17-09-FRCNN':
                    self.args.adjusted_gate = 1.20
                elif video_name == 'MOT17-10-FRCNN':
                    self.args.adjusted_gate = 1.90
                elif video_name == 'MOT17-11-FRCNN':
                    self.args.adjusted_gate = 1.70
                elif video_name == 'MOT17-13-FRCNN':
                    self.args.adjusted_gate = 1.50
                
                if video_name == 'MOT20-01':
                    self.args.match_thresh_d1 = 0.8
                    self.args.match_thresh_d2 = 0.65
                    self.args.match_thresh_d3 = 0.7
                elif video_name == 'MOT20-02':
                    self.args.match_thresh_d1 = 0.65
                    self.args.match_thresh_d2 = 0.6
                    self.args.match_thresh_d3 = 0.95
                elif video_name == 'MOT20-03':
                    self.args.match_thresh_d1 = 0.75
                    self.args.match_thresh_d2 = 0.5
                    self.args.match_thresh_d3 = 0.95
                elif video_name == 'MOT20-05':
                    self.args.match_thresh_d1 = 0.75
                    self.args.match_thresh_d2 = 0.5
                    self.args.match_thresh_d3 = 0.95

                if video_name == 'MOT20-01':
                    self.args.adjusted_gate = 1.2
                elif video_name == 'MOT20-02':
                    self.args.adjusted_gate = 1.1
                elif video_name == 'MOT20-03':
                    self.args.adjusted_gate = 1.3
                elif video_name == 'MOT20-05':
                    self.args.adjusted_gate = 1.1

                if video_name == 'dancetrack0004': # 4, 5, 7, 10, 14, 18, 19, 25, 26, 30, 34, 35, 41, 43, 47, 58, 63, 65, 73, 77, 79, 81, 90, 94, 97
                    self.args.adjusted_gate = 1.2
                elif video_name == 'dancetrack0005':
                    self.args.adjusted_gate = 1.7
                elif video_name == 'dancetrack0007':
                    self.args.adjusted_gate = 1.5
                elif video_name == 'dancetrack0010':
                    self.args.adjusted_gate = 1.9
                elif video_name == 'dancetrack0014':
                    self.args.adjusted_gate = 1.2
                elif video_name == 'dancetrack0018':
                    self.args.adjusted_gate = 1.9
                elif video_name == 'dancetrack0018':
                    self.args.adjusted_gate = 1.4
                elif video_name == 'dancetrack0025':
                    self.args.adjusted_gate = 1.1
                elif video_name == 'dancetrack0026':
                    self.args.adjusted_gate = 1.1
                elif video_name == 'dancetrack0030':
                    self.args.adjusted_gate = 1.4
                elif video_name == 'dancetrack0034':
                    self.args.adjusted_gate = 1.7
                elif video_name == 'dancetrack0035':
                    self.args.adjusted_gate = 1.3
                elif video_name == 'dancetrack0041':
                    self.args.adjusted_gate = 1.8
                elif video_name == 'dancetrack0043':
                    self.args.adjusted_gate = 1.3
                elif video_name == 'dancetrack0047':
                    self.args.adjusted_gate = 1.8
                elif video_name == 'dancetrack0058':
                    self.args.adjusted_gate = 1.5
                elif video_name == 'dancetrack0063':
                    self.args.adjusted_gate = 1.2
                elif video_name == 'dancetrack0065':
                    self.args.adjusted_gate = 1.8
                elif video_name == 'dancetrack0018':
                    self.args.adjusted_gate = 2.0
                elif video_name == 'dancetrack0077':
                    self.args.adjusted_gate = 1.0
                elif video_name == 'dancetrack0079':
                    self.args.adjusted_gate = 1.7
                elif video_name == 'dancetrack0081':
                    self.args.adjusted_gate = 1.3
                elif video_name == 'dancetrack0090':
                    self.args.adjusted_gate = 1.4
                elif video_name == 'dancetrack0094':
                    self.args.adjusted_gate = 1.3
                elif video_name == 'dancetrack0097':
                    self.args.adjusted_gate = 1.3
                
                
                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    BaseTrack._count = 0
                    tracker = DYTETracker(self.args, video_name)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results(result_filename, results)
                        results = []

                        out.release()
                        cv2.destroyAllWindows()
                    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # 视频编解码器
                    width, height = int(info_imgs[1]), int(info_imgs[0])  # 宽高
                    out = cv2.VideoWriter(result_folder[:-len("track_results")] + '/result{}.mp4'.format(video_names[video_id]), 
                                          fourcc, 30 // self.args.sampling_rate, 
                                          (width, height), isColor=True)

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                with torch.no_grad():
                    outputs = model(imgs)
                    if decoder is not None:
                        outputs = decoder(outputs, dtype=outputs.type())

                    outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            det_results.append([output_results, frame_id, video_name, img_file_name])

            # run tracking
            if outputs[0] is not None:
                online_targets, bboxes, scores = tracker.update(outputs[0], info_imgs, self.img_size, frame_id, np.array(raw_imgs[0]))
                online_tlwhs = []
                online_ids = []
                online_scores = []
                online_dids = []
                online_prediction_tlwhs = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    did = t.detection_id
                    last_tlwh = t.last_tlwh
                    if self.args.dance:
                        vertical = False
                    else:
                        vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        online_dids.append(did)
                        if len(results) != 0:
                           online_prediction_tlwhs.append(last_tlwh)
                # save results
                results.append((frame_id, online_tlwhs, online_ids, online_dids, online_scores))
                plot_image = plot_tracking(raw_imgs[0], online_tlwhs, online_prediction_tlwhs, online_ids, fps=30 // self.args.sampling_rate, frame_id=frame_id)
                out.write(plot_image)

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        pickle.dump(det_results, open("/".join(result_folder.split("/")[:-1]) + "/det_results.pkl", "wb"))
        return eval_results
    
    def detection(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None
    ):
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        dresults = []
        det_results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1
            
        # tracker = DYTETracker(self.args, video_name)
        ori_thresh = self.args.track_thresh
        for cur_iter, (imgs, raw_imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]
                
                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = DYTETracker(self.args, video_name)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.npy'.format(video_names[video_id - 1]))
                        # write_results(result_filename, results)
                        concat_results = np.concatenate(results, axis=0)
                        np.save(result_filename, concat_results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                with torch.no_grad():
                    outputs = model(imgs)
                    if decoder is not None:
                        outputs = decoder(outputs, dtype=outputs.type())

                    outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            det_results.append([output_results, frame_id, video_name, img_file_name])

            # run tracking
            if outputs[0] is not None:
                # online_targets, bboxes, scores = tracker.update(outputs[0], info_imgs, self.img_size, frame_id, np.array(raw_imgs[0]))
                if outputs[0].shape[1] == 5:
                    scores = outputs[0][:, 4]
                    bboxes = outputs[0][:, :4]
                else:
                    outputs[0] = outputs[0].cpu().numpy()
                    scores = outputs[0][:, 4] * outputs[0][:, 5]
                    bboxes = outputs[0][:, :4]  # x1y1x2y2
                img_h, img_w = info_imgs[0], info_imgs[1]
                scale = min(self.img_size[0] / float(img_h), self.img_size[1] / float(img_w))
                bboxes /= scale
                online_tlbrs = []
                raw_frame_id = int(img_file_name[0].split('/')[-1].split('.')[0])
                for bbox, score in zip(bboxes, scores):
                    x1, y1, x2, y2 = bbox
                    if score > 0.1:
                        online_tlbrs.append([raw_frame_id, -1, x1, y1, x2, y2, score, -1, -1, -1])
                features = tracker.encoder.inference(np.array(raw_imgs[0]), np.array([[x1, y1, x2, y2] for frameid, _, x1, y1, x2, y2, score, _, _, _ in online_tlbrs]))
                online_tlbrs = np.array(online_tlbrs)
                
                cat_features = np.concatenate((online_tlbrs, features), axis=1)
                
                # save results
                results.append(cat_features)

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.npy'.format(video_names[video_id]))
                concat_results = np.concatenate(results, axis=0)
                np.save(result_filename, concat_results)
                # write_results(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        # pickle.dump(det_results, open("/".join(result_folder.split("/")[:-1]) + "/det_results.pkl", "wb"))
        return eval_results
    
    def evaluate_akl(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        det_results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
            
        tracker = AdjustTracker(self.args)
        ori_thresh = self.args.track_thresh
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]
                if video_name == 'MOT17-05-FRCNN' or video_name == 'MOT17-06-FRCNN':
                    self.args.track_buffer = 14
                elif video_name == 'MOT17-13-FRCNN' or video_name == 'MOT17-14-FRCNN':
                    self.args.track_buffer = 25
                else:
                    self.args.track_buffer = 30

                if video_name == 'MOT17-01-FRCNN':
                    self.args.track_thresh = 0.65
                elif video_name == 'MOT17-06-FRCNN':
                    self.args.track_thresh = 0.65
                elif video_name == 'MOT17-12-FRCNN':
                    self.args.track_thresh = 0.7
                elif video_name == 'MOT17-14-FRCNN':
                    self.args.track_thresh = 0.67
                elif video_name in ['MOT20-06', 'MOT20-08']:
                    self.args.track_thresh = 0.3
                else:
                    self.args.track_thresh = ori_thresh

                if video_name == 'MOT17-02-FRCNN':
                    self.args.match_thresh_d1 = 0.85
                    self.args.match_thresh_d2 = 0.45
                    self.args.match_thresh_d3 = 0.75
                elif video_name == 'MOT17-04-FRCNN':
                    self.args.match_thresh_d1 = 0.90
                    self.args.match_thresh_d2 = 0.55
                    self.args.match_thresh_d3 = 0.75
                elif video_name == 'MOT17-05-FRCNN':
                    self.args.match_thresh_d1 = 0.80
                    self.args.match_thresh_d2 = 0.50
                    self.args.match_thresh_d3 = 0.75
                elif video_name == 'MOT17-09-FRCNN':
                    self.args.match_thresh_d1 = 0.90
                    self.args.match_thresh_d2 = 0.50
                    self.args.match_thresh_d3 = 0.75
                elif video_name == 'MOT17-10-FRCNN':
                    self.args.match_thresh_d1 = 1.00
                    self.args.match_thresh_d2 = 0.55
                    self.args.match_thresh_d3 = 0.75
                elif video_name == 'MOT17-11-FRCNN':
                    self.args.match_thresh_d1 = 0.80
                    self.args.match_thresh_d2 = 0.50
                    self.args.match_thresh_d3 = 0.65
                elif video_name == 'MOT17-13-FRCNN':
                    self.args.match_thresh_d1 = 1.00
                    self.args.match_thresh_d2 = 0.55
                    self.args.match_thresh_d3 = 0.75

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = AdjustTracker(self.args)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                with torch.no_grad():
                    outputs = model(imgs)
                    if decoder is not None:
                        outputs = decoder(outputs, dtype=outputs.type())

                    outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            det_results.append([output_results, frame_id, video_name, img_file_name])

            # run tracking
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], info_imgs, self.img_size, frame_id)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                # save results
                results.append((frame_id, online_tlwhs, online_ids, online_scores))
                
            
            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        pickle.dump(det_results, open("/".join(result_folder.split("/")[:-1]) + "/det_results.pkl", "wb"))
        return eval_results
    
    def evaluate_history(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        det_results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
            
        tracker = HistoryTracker(self.args)
        ori_thresh = self.args.track_thresh
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]
                if video_name == 'MOT17-05-FRCNN' or video_name == 'MOT17-06-FRCNN':
                    self.args.track_buffer = 14
                elif video_name == 'MOT17-13-FRCNN' or video_name == 'MOT17-14-FRCNN':
                    self.args.track_buffer = 25
                else:
                    self.args.track_buffer = 30

                if video_name == 'MOT17-01-FRCNN':
                    self.args.track_thresh = 0.65
                elif video_name == 'MOT17-06-FRCNN':
                    self.args.track_thresh = 0.65
                elif video_name == 'MOT17-12-FRCNN':
                    self.args.track_thresh = 0.7
                elif video_name == 'MOT17-14-FRCNN':
                    self.args.track_thresh = 0.67
                elif video_name in ['MOT20-06', 'MOT20-08']:
                    self.args.track_thresh = 0.3
                else:
                    self.args.track_thresh = ori_thresh

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = HistoryTracker(self.args)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                with torch.no_grad():
                    outputs = model(imgs)
                    if decoder is not None:
                        outputs = decoder(outputs, dtype=outputs.type())

                    outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            det_results.append([output_results, frame_id, video_name, img_file_name])

            # run tracking
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], info_imgs, self.img_size, frame_id)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                # save results
                results.append((frame_id, online_tlwhs, online_ids, online_scores))
                
            
            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        pickle.dump(det_results, open("/".join(result_folder.split("/")[:-1]) + "/det_results.pkl", "wb"))
        return eval_results

    def evaluate_sort(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
            
        tracker = Sort(self.args.track_thresh)
        
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = Sort(self.args.track_thresh)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results_no_score(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            online_targets = tracker.update(outputs[0], info_imgs, self.img_size)
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
            # save results
            results.append((frame_id, online_tlwhs, online_ids))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results_no_score(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def evaluate_deepsort(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None,
        model_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
            
        tracker = DeepSort(model_folder, min_confidence=self.args.track_thresh)
        
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = DeepSort(model_folder, min_confidence=self.args.track_thresh)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results_no_score(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            online_targets = tracker.update(outputs[0], info_imgs, self.img_size, img_file_name[0])
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
            # save results
            results.append((frame_id, online_tlwhs, online_ids))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results_no_score(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def evaluate_motdt(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None,
        model_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
            
        tracker = OnlineTracker(model_folder, min_cls_score=self.args.track_thresh)
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = OnlineTracker(model_folder, min_cls_score=self.args.track_thresh)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            online_targets = tracker.update(outputs[0], info_imgs, self.img_size, img_file_name[0])
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
            # save results
            results.append((frame_id, online_tlwhs, online_ids, online_scores))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def convert_to_coco_format(self, outputs, info_imgs, ids):
        data_list = []
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
        return data_list

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        track_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_track_time = 1000 * track_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "track", "inference"],
                    [a_infer_time, a_track_time, (a_infer_time + a_track_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            _, tmp = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, "w"))
            cocoDt = cocoGt.loadRes(tmp)
            '''
            try:
                from yolox.layers import COCOeval_opt as COCOeval
            except ImportError:
                from pycocotools import cocoeval as COCOeval
                logger.warning("Use standard COCOeval.")
            '''
            #from pycocotools.cocoeval import COCOeval
            from yolox.layers import COCOeval_opt as COCOeval
            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info
