import os
import random
from copy import deepcopy

import cv2
import torch
import argparse
import numpy as np
import torch.nn as nn
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from yolox.sr_tracker.mann import MANN


def draw_sequences(predict_sequence, target_sequence, width, height, length, figpath):
    # Create figure and axes
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Draw predicted sequence
    ax[0].set_xlim(0, width)
    ax[0].set_ylim(0, height)
    # Reverse the y axis as the origin point is top left in image processing
    ax[0].invert_yaxis()
    ax[0].set_title('Predicted Sequence')
    for i in range(length):
        x1, y1, w, h = predict_sequence[i]
        rect = patches.Rectangle(
            (x1, y1), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax[0].add_patch(rect)

    # Draw target sequence
    ax[1].set_xlim(0, width)
    ax[1].set_ylim(0, height)
    # Reverse the y axis as the origin point is top left in image processing
    ax[1].invert_yaxis()
    ax[1].set_title('Target Sequence')
    for i in range(length):
        x1, y1, w, h = target_sequence[i]
        rect = patches.Rectangle(
            (x1, y1), w, h, linewidth=1, edgecolor='g', facecolor='none')
        ax[1].add_patch(rect)

    plt.savefig(figpath)
    plt.close()
    plt.clf()
    plt.cla()


def make_dataset(args, det_file):
    data_dir = os.path.join("./datasets", args.data_name, "train")
    if args.suffix is not None:
        scene_list = list(filter(lambda x: x.split(
            "-")[-1] == args.suffix, os.listdir(data_dir)))
    else:
        scene_list = os.listdir(data_dir)

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
                frameid, trackid, xmin, ymin, w, h, confidence, classId, visibility = line.rstrip(
                    "\n").split(",")
                frameid, trackid, xmin, ymin, w, h, confidence, classId, visibility = int(frameid), int(
                    trackid), int(xmin), int(ymin), int(w), int(h), int(confidence), int(classId), float(visibility)
                if confidence < 1:
                    continue
                if frameid not in frame_remap:
                    frame_remap[frameid] = len(frame_remap)
                frameid = frame_remap[frameid]
                if frameid not in scene_data:
                    scene_data[frameid] = {}
                if trackid not in scene_data[frameid]:
                    scene_data[frameid][trackid] = [xmin / width,
                                                    ymin / height,
                                                    w / width,
                                                    h / height]
                if trackid not in track_data:
                    track_data[trackid] = {}
                if frameid not in track_data[trackid]:
                    track_data[trackid][frameid] = [xmin / width,
                                                    ymin / height,
                                                    w / width,
                                                    h / height]

        maked_dataset[scene] = [scene_data, track_data, width, height]
    return maked_dataset


def batch_sampler(dataset, batch_size, seq_len, samples=30, hiddens=128, eval=False):
    scene_list = list(dataset.keys())
    batch_width, batch_height = [], []
    batch_c, batch_x, batch_y, batch_lengths = [], [], [], []

    while True:
        scene = random.choice(scene_list)
        scene_data, track_data, width, height = dataset[scene]
        # print(scene, "width, height", width, height)

        c, x, y = [], [], []
        sample_rate = random.randint(1, samples)
        trackid = random.choice(list(track_data.keys()))

        frameid_list = list(track_data[trackid].keys())
        if len(frameid_list) < 2:
            continue
        start_frame = random.choice(frameid_list[:-1])
        max_frame = max(frameid_list)
        for frameid in range(start_frame, max_frame, sample_rate):
            if frameid + sample_rate not in track_data[trackid]:
                continue
            if frameid in track_data[trackid]:
                x1, y1, w, h = track_data[trackid][frameid]
                xc, y2 = x1 + w / 2, y1 + h

                pos = [x1, y1, w, h,
                       xc, y2]

                if len(x) == 0:
                    pos += [0, 0, 0, 0,
                            0, 0,
                            0, 0, 0, 0,
                            0, 0,
                            sample_rate / samples, sample_rate / samples]
                else:
                    lastx1, lasty1, lastw, lasth, lastxc, lasty2 = x[-1][:6]

                    pos += [x1 - lastx1, y1 - lasty1, w - lastw, h - lasth,
                            xc - lastxc, y2 - lasty2,
                            (x1 - lastx1) / sample_rate, (y1 - lasty1) /
                            sample_rate, (w - lastw) /
                            sample_rate, (h - lasth) / sample_rate,
                            (xc - lastxc) /
                            sample_rate, (y2 - lasty2) / sample_rate,
                            sample_rate / samples, sample_rate / samples]

                x.append(pos)
            else:
                break

            cc = [0] * hiddens
            detectids = list(scene_data[frameid].keys())
            detectids = sorted(
                detectids, key=lambda tid: scene_data[frameid][tid][0], reverse=True)
            for tid, detectid in enumerate(detectids):
                if tid >= hiddens // 4:
                    break
                x1, y1, w, h = scene_data[frameid][detectid]
                cc[tid * 4] = x1
                cc[tid * 4 + 1] = y1
                cc[tid * 4 + 2] = w
                cc[tid * 4 + 3] = h
            c.append(cc)

            if frameid + sample_rate in track_data[trackid]:
                x1, y1, w, h = track_data[trackid][frameid + sample_rate]
                y.append([x1 + w / 2, y1 + h, w / h, h])
            else:
                minimum_proximity_frame, maxmum_proximity_frame = 0, 0
                for frame in track_data[trackid]:
                    if frame > frameid:
                        maxmum_proximity_frame = frame
                        break
                for frame in reversed(track_data[trackid]):
                    if frame < frameid:
                        minimum_proximity_frame = frame
                        break
                proximity_frame_data = []
                for i in range(4):
                    proximity_frame_data.append(track_data[trackid][minimum_proximity_frame][i] +
                                                (track_data[trackid][maxmum_proximity_frame][i] -
                                                 track_data[trackid][minimum_proximity_frame][i]) * (frameid - minimum_proximity_frame) / (maxmum_proximity_frame - minimum_proximity_frame))
                x1, y1, w, h = proximity_frame_data
                y.append([x1 + w / 2, y1 + h, w / h, h])
        assert len(x) == len(y), f"Error: {len(x)} != {len(y)}"
        if len(x) < 2:
            continue
        # Truncate or Padding
        if len(x) > seq_len:
            x = x[:seq_len]
            y = y[:seq_len]
            c = c[:seq_len]
            length = seq_len
        elif len(x) < seq_len:
            length = len(x)
            for i in range(seq_len - len(x)):
                x.append([0] * len(x[0]))
                y.append([0] * len(y[0]))
                c.append([0] * len(c[0]))
        else:
            length = seq_len
        assert len(x) == len(y) and len(
            x) == seq_len, f"Error: {len(x)} != {len(y)} or {len(x)} != {seq_len}"
        batch_c.append(c)
        batch_x.append(x)
        batch_y.append(y)
        batch_lengths.append(length)
        batch_width.append(width)
        batch_height.append(height)
        if len(batch_x) == batch_size:
            if eval:
                yield batch_c, batch_x, batch_y, batch_lengths, batch_width, batch_height
            else:
                yield batch_c, batch_x, batch_y, batch_lengths
            batch_c, batch_x, batch_y, batch_lengths = [], [], [], []


parser = argparse.ArgumentParser("MANN Training")
parser.add_argument("--data_name", type=str, default="mot_1")
parser.add_argument("--suffix", type=str, default=None)
args = parser.parse_args()

train_dataset, valid_dataset = make_dataset(
    args, "gt_train_half.txt"), make_dataset(args, "gt_val_half.txt")

data_flag = args.data_name.split('_')[0]

model = MANN(input_size=20,
             hidden_size=64,
             memory_size=128,
             memory_feature_size=128)

model = model.cuda()
model.train()

min_lr = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=1000, gamma=0.8)

loss_func = nn.HuberLoss(reduction='none')

batchsize = 64
seqlength = 256
train_val_step = 500
train_steps = 50000
val_steps = 100

train_step = 0
losses = []
best_metirc = 9999

seed = 8367  # random.randint(0, 9999)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

for batch_data in batch_sampler(train_dataset, batchsize, seqlength):
    contexts, sequences, target_sequences, lengths = batch_data

    contexts = torch.FloatTensor(contexts).cuda()
    sequences = torch.FloatTensor(sequences).cuda()
    target_sequences = torch.FloatTensor(target_sequences).cuda()
    predict_sequences = model(sequences, contexts, lengths)

    lengths = torch.tensor(lengths).cuda()
    seq_index = torch.arange(seqlength).unsqueeze(
        0).expand(batchsize, -1).cuda()

    lengths = lengths.unsqueeze(1).expand(-1, seqlength)
    mask = seq_index < lengths
    mask = mask.unsqueeze(2).expand(-1, -1, 4)

    valid_predict_sequences = predict_sequences[mask].view(-1, 4)
    valid_target_sequences = target_sequences[mask].view(-1, 4)

    loss = loss_func(valid_predict_sequences,
                     valid_target_sequences).sum(dim=1).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    for param_group in optimizer.param_groups:
        param_group['lr'] = max(param_group['lr'], min_lr)

    losses.append(loss.item())
    train_step += 1

    if train_step % 10 == 0:
        plt.plot(losses)
        plt.savefig(f'./yolox/sr_tracker/{data_flag}_result/loss.png')
        plt.close()
        plt.clf()
        plt.cla()

    if train_step % train_val_step == 0:
        model.eval()
        with torch.no_grad():
            valid_step = 0
            valid_losses, infos = [], []
            for batch_data in batch_sampler(valid_dataset, batchsize, seqlength, eval=True):
                contexts, sequences, target_sequences, lengths, widths, heights = batch_data
                contexts = torch.FloatTensor(contexts).cuda()
                sequences = torch.FloatTensor(sequences).cuda()
                target_sequences = torch.FloatTensor(target_sequences).cuda()
                predict_sequences = model(sequences, contexts, lengths)

                lengths = torch.tensor(lengths).cuda()
                raw_lengths = deepcopy(lengths)
                seq_index = torch.arange(seqlength).unsqueeze(
                    0).expand(batchsize, -1).cuda()

                lengths = lengths.unsqueeze(1).expand(-1, seqlength)
                mask = seq_index < lengths
                mask = mask.unsqueeze(2).expand(-1, -1, 4)

                valid_predict_sequences = predict_sequences[mask].view(-1, 4)
                valid_target_sequences = target_sequences[mask].view(-1, 4)

                loss = loss_func(valid_predict_sequences,
                                 valid_target_sequences).sum(dim=1).mean()
                valid_losses.append(loss.item())

                if valid_step > val_steps:
                    break

                infos.append([sequences, target_sequences,
                             predict_sequences, widths, heights, raw_lengths])
                valid_step += 1
            print('Valid loss: {}'.format(np.mean(valid_losses)))
            metric = np.mean(valid_losses)

            if metric < best_metirc:
                best_metirc = metric
                torch.save(
                    model.state_dict(), f'./yolox/sr_tracker/{data_flag}_result/model_best.pth')
                print('Save model at step {}...'.format(train_step))
            torch.save(model.state_dict(
            ), f'./yolox/sr_tracker/{data_flag}_result/model_{train_step}.pth')

            # Visualize
            for ii, info in enumerate(infos):
                sequences, target_sequences, predict_sequences, widths, heights, lengths = info

                if not os.path.exists(f'./yolox/sr_tracker/{data_flag}_result/sequences'):
                    os.makedirs(
                        f'./yolox/sr_tracker/{data_flag}_result/sequences')

                if not os.path.exists(f'./yolox/sr_tracker/{data_flag}_result/sequences/{train_step}'):
                    os.makedirs(
                        f'./yolox/sr_tracker/{data_flag}_result/sequences/{train_step}')

                predict_sequences = predict_sequences.cpu().numpy()
                target_sequences = target_sequences.cpu().numpy()
                sequences = sequences.cpu().numpy()

                sequenceid = 0
                for predict_sequence, target_sequence, width, height, length in zip(predict_sequences, target_sequences, widths, heights, lengths):
                    if sequenceid > 5:
                        break
                    predict_sequence[:, 2] *= predict_sequence[:, 3]
                    target_sequence[:, 2] *= target_sequence[:, 3]

                    predict_sequence[:, 0] -= predict_sequence[:, 2] / 2
                    predict_sequence[:, 1] -= predict_sequence[:, 3]

                    target_sequence[:, 0] -= target_sequence[:, 2] / 2
                    target_sequence[:, 1] -= target_sequence[:, 3]

                    predict_sequence = predict_sequence * \
                        np.array([width, height, width, height])
                    target_sequence = target_sequence * \
                        np.array([width, height, width, height])
                    draw_sequences(predict_sequence, target_sequence, width, height, length.item(
                    ), f'./yolox/sr_tracker/{data_flag}_result/sequences/{train_step}/{ii}_{sequenceid}.png')
                    sequenceid += 1

        model.train()

    if train_step > train_steps:
        break
