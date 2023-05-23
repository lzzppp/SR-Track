import os
import cv2

dataset_dir = "./datasets/mot_1/train"
video_names = os.listdir(dataset_dir)

for video_name in video_names:
    if video_name == "seqmaps":
        continue
    video_dir = os.path.join(dataset_dir, video_name)
    image_names = os.listdir(os.path.join(video_dir, "img1"))
    image_names = sorted(image_names, key=lambda x: int(x.split(".")[0]))
    image_number = len(image_names)
    with open(os.path.join(video_dir, "gt/gt_val_half.txt"), "r") as f:
        lines = f.readlines()
        frame_bound = [9999, -9999]
        for line in lines:
            frame_id = int(line.split(",")[0])
            frame_bound[0] = min(frame_bound[0], frame_id)
            frame_bound[1] = max(frame_bound[1], frame_id)
        
    assert len(image_names[image_number//2+1:]) == frame_bound[1] - frame_bound[0] + 1
    
    img0 = cv2.imread(os.path.join(video_dir, "img1", image_names[0]))
    height, width, layers = img0.shape
    # 视频保存初始化 VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_file = os.path.join(video_dir, "video.mp4")
    
    videowriter = cv2.VideoWriter(output_video_file, fourcc, 30, (width, height))
        
    for image_name in image_names[image_number//2+1:]:
        image_path = os.path.join(video_dir, "img1", image_name)
        img = cv2.imread(image_path)
        videowriter.write(img)
    videowriter.release()
    cv2.destroyAllWindows()