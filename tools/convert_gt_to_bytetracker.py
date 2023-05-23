import os
import glob
import torch
import random
import argparse
import warnings
import imagesize
import numpy as np
from tqdm import tqdm
import motmetrics as mm
from loguru import logger
from yolox.exp import get_exp
from yolox.core import launch
from rich.progress import track
from yolox.tracker import matching
import torch.backends.cudnn as cudnn
from yolox.evaluators import MOTEvaluator
from yolox.data import MOTDataset, ValTransform
from cython_bbox import bbox_overlaps as bbox_ious
from torch.nn.parallel import DistributedDataParallel as DDP
from yolox.utils import configure_nccl, fuse_model, get_local_rank, get_model_info, setup_logger, postprocess, xyxy2xywh

def tlwh_to_tlbr(tlwh):
    ret = np.asarray(tlwh).copy()
    ret[:, 2:] += ret[:, :2]
    return ret

def make_parser():
    parser = argparse.ArgumentParser("YOLOX Eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument('-s', '--sampling_rate', help="Target sampling rate", type=int, default=None)

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "--local_rank", default=0, type=int, help="local rank for dist training"
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument(
        "--test",
        dest="test",
        default=False,
        action="store_true",
        help="Evaluating on test-dev set.",
    )
    parser.add_argument(
        "--speed",
        dest="speed",
        default=False,
        action="store_true",
        help="speed test only.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    # det args
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=0.01, type=float, help="test conf")
    parser.add_argument("--nms", default=0.7, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
    parser.add_argument("--min-box-area", type=float, default=100, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser

def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:            
            logger.info('Comparing {}...'.format(k))
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logger.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names

def convert_to_coco_format(img_size, dataloader, outputs, info_imgs, ids):
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
            img_size[0] / float(img_h), img_size[1] / float(img_w)
        )
        bboxes /= scale
        bboxes = xyxy2xywh(bboxes)

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        for ind in range(bboxes.shape[0]):
            label = dataloader.dataset.class_ids[int(cls[ind])]
            pred_data = {
                "image_id": int(img_id),
                "category_id": label,
                "bbox": bboxes[ind].numpy().tolist(),
                "score": scores[ind].numpy().item(),
                "segmentation": [],
            }  # COCO json format
            data_list.append(pred_data)
    return data_list

@logger.catch
def main(exp, args, num_gpu):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed testing. This will turn on the CUDNN deterministic setting, "
        )

    is_distributed = num_gpu > 1

    # set environment variables for distributed training
    cudnn.benchmark = True

    rank = args.local_rank
    # rank = get_local_rank()
    sampling_rate = args.sampling_rate

    file_name = os.path.join(exp.output_dir, args.experiment_name)

    if rank == 0:
        os.makedirs(file_name, exist_ok=True)

    results_folder = os.path.join(file_name, "track_results")
    os.makedirs(results_folder, exist_ok=True)

    setup_logger(file_name, distributed_rank=rank, filename="val_log.txt", mode="a")
    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    #logger.info("Model Structure:\n{}".format(str(model)))

    val_loader = exp.get_eval_loader(args.batch_size, is_distributed, args.test)

    torch.cuda.set_device(rank)
    model.cuda(rank)
    model.eval()

    if args.ckpt is None:
        ckpt_file = os.path.join(file_name, "best_ckpt.pth.tar")
    else:
        ckpt_file = args.ckpt
    logger.info("loading checkpoint")
    loc = "cuda:{}".format(rank)
    ckpt = torch.load(ckpt_file, map_location=loc)
    # load the model state dict
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")

    model = fuse_model(model)

    model = model.half()

    data_dir = './datasets'
    all_sampling_rates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    score_thre = 0.1

    progress_bar = tqdm
    for sampling_rate in all_sampling_rates:
        sampling_rate_dataset_dir = os.path.join(data_dir, "mot_%d" % sampling_rate)

        valdataset = MOTDataset(
            data_dir=sampling_rate_dataset_dir,
            json_file='train.json',
            img_size=(800, 1440),
            name='train',
            preproc=ValTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        )

        sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": 4,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = 1
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        tracker_datasets = {}

        for imgs, _, info_imgs, ids in progress_bar(val_loader):
            tensor_type = torch.cuda.HalfTensor
            imgs = imgs.type(tensor_type)
            frame_id = info_imgs[2].item()
            video_id = info_imgs[3].item()
            img_file_name = info_imgs[4]
            video_name = img_file_name[0].split('/')[0]

            with torch.no_grad():
                outputs = model(imgs)

            outputs = postprocess(outputs, exp.num_classes, exp.test_conf, exp.nmsthre)

            output_results = convert_to_coco_format((800, 1440), val_loader, outputs, info_imgs, ids)
            for output_result in output_results:
                score = output_result['score']
                if score > score_thre:
                    if video_name not in tracker_datasets:
                        tracker_datasets[video_name] = {}
                    if frame_id not in tracker_datasets[video_name]:
                        tracker_datasets[video_name][frame_id] = []
                    tracker_datasets[video_name][frame_id].append(output_result['bbox'])

        gt_datasets = {}
        sampling_rate_dir = os.path.join(data_dir, "mot_%d" % sampling_rate, "train")
        video_list = os.listdir(sampling_rate_dir)
        for video_name in video_list:
            frame_dataset = {}
            gt_path = os.path.join(sampling_rate_dir, video_name, "gt/gt.txt")
            with open(gt_path, "r") as f:
                for line in f:
                    frame_id, object_id, x, y, w, h, a, b, conf = line.rstrip("\n").split(",")         
                    if int(frame_id) not in frame_dataset:
                        frame_dataset[int(frame_id)] = []
                    frame_dataset[int(frame_id)].append([line, [int(x), int(y), int(w), int(h)]])
            for frame_id in frame_dataset:
                bytetracker_bbox = tracker_datasets[video_name][frame_id]
                gt_info = frame_dataset[frame_id]
                gt_bbox = [bbox for _, bbox in gt_info]
                bytetracker_bbox, gt_bbox = np.array(bytetracker_bbox, dtype=np.float64), np.array(gt_bbox, dtype=np.float64)
                bytetracker_tlbr = tlwh_to_tlbr(bytetracker_bbox)
                gt_tlbr = tlwh_to_tlbr(gt_bbox)

                ious = 1.0 - bbox_ious(bytetracker_tlbr, gt_tlbr)
                matches, u_track, u_detection = matching.linear_assignment(ious, thresh=0.1)
                print(matches, ious)
                for byte_id, ground_id in matches:
                    new_x, new_y, new_w, new_h = bytetracker_bbox[byte_id]
                    frame_id, object_id, x, y, w, h, a, b, conf = gt_info[ground_id][0].rstrip("\n").split(",")
                    new_line = f"{frame_id},{object_id},{new_x},{new_y},{new_w},{new_h},{a},{b},{conf}\n"
                    frame_dataset[int(frame_id)][ground_id] = [new_line, [new_x, new_y, new_w, new_h]]
            gt_datasets[video_name] = frame_dataset
        
        for video_name in gt_datasets:
            gt_byte_path = os.path.join(sampling_rate_dir, video_name, "gt/gt_byte.txt")
            with open(gt_byte_path, "w") as gfw:
                for frame_id in gt_datasets[video_name]:
                    frame_data = gt_datasets[video_name][frame_id]
                    for new_line, bbox in frame_data:
                        gfw.write(new_line)

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if args.experiment_name:
        args.experiment_name = args.experiment_name + "/" + exp.exp_name
    print("Experiment Name: ", args.experiment_name)

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=args.dist_url,
        args=(exp, args, num_gpu),
    )