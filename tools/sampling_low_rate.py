import os
import json
import argparse
from glob import glob
from tkinter import W
from unicodedata import category

def sampling_low_rate_json(gt_json_file, target_json_file, sampling_rate):
    
    with open(gt_json_file,'r') as load_f:
        load_dict = json.load(load_f)
    images_list, annotations_list, videos_list, categories = [], [], [], []
    for img in load_dict['images']:
        if img['frame_id'] % sampling_rate == 0:
            img['frame_id'] = img['frame_id'] // sampling_rate
            img['file_name'] = "/".join(img['file_name'].split('/')[:-1] + ["%06d.jpg" % img['frame_id']])
            images_list.append(img)
    for id, img in enumerate(images_list):
        img['id'] = id
    for img in images_list:
        for img_f in images_list:
            if img_f['frame_id'] == img['frame_id'] - 1:
                img['prev_image_id'] = img_f['id']
                break
    for img in images_list:
        for img_f in images_list:
            if img_f['frame_id'] == img['frame_id'] + 1:
                img['next_image_id'] = img_f['id']
                break
    
def sampling_low_rate(sampling_rate, gtfile, targetfile):
    if os.path.exists(targetfile):
        print("%s already exists" % targetfile)
        return
    gt_reader = open(gtfile, "r")
    target_writer = open(targetfile, "w")
    for gt_line in gt_reader:
        write_line = gt_line.rstrip("\n").split(",")
        write_line[0] = str(int(write_line[0]) // sampling_rate)
        write_line = ",".join(write_line)
        gt_line = gt_line.rstrip("\n")
        gt_line_split = gt_line.split(",")
        if int(gt_line_split[0]) % sampling_rate == 0:
            target_writer.write(write_line + "\n")
    target_writer.close()
    gt_reader.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='sampling low-rate images from high-rate images')
    parser.add_argument(
        '-s', '--sampling_rate', help="Target sampling rate", type=int, default=1)
    parser.add_argument(
        '-g', '--gt_file', help="Ground Truth File", type=str, default=None)
    parser.add_argument(
        '-t', '--target_file', help="Target File", type=str, default=None)
    parser.add_argument(
        '-m', '--test_mode', help="Target File", type=bool, default=False)
    args = parser.parse_args()
    
    mode = args.test_mode

    if mode:
        sample_dirs = ['img1', 'det']
    else:
        sample_dirs = ['gt', 'img1', 'det']
    
    # if args.sampling_rate == 1:
    #     print("No need to sampling")
    #     exit(0)
        
    gt_file = args.gt_file
    target_file = args.target_file
    
    gt_dirs = os.listdir(gt_file)
    if not os.path.exists(target_file):
        os.makedirs(target_file)
    
    for gt_dir in gt_dirs:
        if gt_dir not in sample_dirs:
            continue
            
        if gt_dir == 'gt' or gt_dir == 'det':
            gt_track_paths = os.listdir(os.path.join(gt_file, gt_dir))
            if not os.path.exists(os.path.join(target_file, gt_dir)):
                os.makedirs(os.path.join(target_file, gt_dir))
            for gt_track_path in gt_track_paths:
                gt_track_reader = os.path.join(gt_file, gt_dir, gt_track_path)
                target_track_writer = os.path.join(target_file, gt_dir, gt_track_path)
                sampling_low_rate(args.sampling_rate, gt_track_reader, target_track_writer)
        elif gt_dir == 'img1':
            gt_track_paths = os.listdir(os.path.join(gt_file, gt_dir))
            if not os.path.exists(os.path.join(target_file, gt_dir)):
                os.makedirs(os.path.join(target_file, gt_dir))
            for gt_track_path in gt_track_paths:
                write_track_path = "%06d.jpg" % (int(gt_track_path.split(".")[0]) // args.sampling_rate)
                if int(gt_track_path.split(".")[0]) % args.sampling_rate == 0:
                    os.system("cp %s %s" % (os.path.join(gt_file, gt_dir, gt_track_path), os.path.join(target_file, gt_dir, write_track_path)))