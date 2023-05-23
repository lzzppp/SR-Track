import os
import cv2
import argparse
from glob import glob
from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter

split_dict = {
    0.5: 10,
    0.4: 10,
    0.3: 10,
    0.2: 10,
    0.1: 10,
}

def split_videos(images, cache_dir, split_num):
    height, width = images[0].shape[:2]
    clip_length = len(images) // split_num
    for split_i in range(split_num):
        # 视频保存初始化 VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video_file = os.path.join(cache_dir, f"video_{split_i}.mp4")
        
        videowriter = cv2.VideoWriter(output_video_file, fourcc, 30, (width, height))
            
        for img in images[split_i * clip_length: (split_i + 1) * clip_length]:
            videowriter.write(img)
        videowriter.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='sampling low-rate images from high-rate images')
    parser.add_argument(
        '-k', '--key_frame_number_rate', help="key frame number", type=float, default=0.5)
    args = parser.parse_args()

    key_frame_number_rate = args.key_frame_number_rate

    gtfiles = glob("./datasets/mot_1/train/MOT17-*-FRCNN")
    if not os.path.exists("./datasets/mot_%d/train" % int(key_frame_number_rate * 1000)):
        os.makedirs("./datasets/mot_%d/train" % int(key_frame_number_rate * 1000))
    else:
        os.system("rm -rf ./datasets/mot_%d/train/*" % int(key_frame_number_rate * 1000))
    targetfiledir = "./datasets/mot_%d/train" % int(key_frame_number_rate * 1000)
    targetfiles = os.listdir("./datasets/mot_%d/train/" % int(key_frame_number_rate * 1000)) # [gtfile.split("/")[-1] for gtfile in gtfiles]

    for gtfile in gtfiles:
        if gtfile.split("/")[-1] in targetfiles or '11' in gtfile:
            continue
        tgfile = os.path.join(targetfiledir, gtfile.split("/")[-1])
        print("sampling %s to %s" % (gtfile, tgfile))
        
        gt_data = []
        frame_dict = {}
        with open(os.path.join(gtfile, "gt/gt_val_half.txt"), "r") as f:
            gt_data.extend(f.readlines())
            for data in gt_data:
                frame_id, object_id, x1, y1, w, h, conf, cls, vis = data.split(",")
                if int(frame_id) not in frame_dict:
                    frame_dict[int(frame_id)] = []
                frame_dict[int(frame_id)].append([object_id, x1, y1, w, h, conf, cls, vis])
        frame_num = int(gt_data[-1].split(",")[0])
        
        vd = Video()

        # number of images to be returned
        no_of_frames_to_returned = int(frame_num * key_frame_number_rate)

        # initialize diskwriter to save data at desired location
        diskwriter = KeyFrameDiskWriter(location="selectedframes")

        # Video file path
        video_file_path = os.path.join(gtfile, "video.mp4")
        video_name = gtfile.split("/")[-1]

        if not os.path.exists(f"{targetfiledir}/{video_name}/img1/"):
            os.makedirs(f"{targetfiledir}/{video_name}/img1/")
        
        if not os.path.exists(f"{targetfiledir}/{video_name}/gt/"):
            os.makedirs(f"{targetfiledir}/{video_name}/gt/")

        print(f"Input video file path = {video_file_path}")

        videoCapture = cv2.VideoCapture(video_file_path)
        
        extract_frame_ids = vd.extract_video_keyframes(
                                no_of_frames=no_of_frames_to_returned, file_path=video_file_path,
                                writer=diskwriter)

        extract_frames = [i + 1 for i in extract_frame_ids]
        
        gt_map_dict = {}
        
        write_file = open(f"{targetfiledir}/{video_name}/gt/gt_val_half.txt", "w")
        for frame_id, extract_frame in enumerate(extract_frames):
            videoCapture.set(cv2.CAP_PROP_POS_FRAMES, extract_frame)
            ret, frame = videoCapture.read()

            cv2.imwrite(f"{targetfiledir}/{video_name}/img1/{frame_id + 1:06d}.jpg", frame)

            object_id, x1, y1, w, h, conf, cls, vis = frame_dict[extract_frame][0]
            if object_id not in gt_map_dict:
                gt_map_dict[object_id] = len(gt_map_dict) + 1
            write_file.write(f"{frame_id + 1},{gt_map_dict[object_id]},{x1},{y1},{w},{h},{conf},{cls},{vis}")
        write_file.close()