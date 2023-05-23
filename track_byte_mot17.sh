for srate in 1 2 3 4 5 6 7 8 9 10
do
    python3 tools/track.py -f exps/example/mot/yolox_x_ablation_$srate.py -c pretrained/bytetrack_ablation.pth.tar -b 1 -d 1 --fp16 --fuse -s $srate --experiment-name bytetracker
done
