for srate in 1 2 3 4 5 6 7 8 9 10
do
    python3 tools/track_dyte.py -f exps/example/mot/yolox_x_ablation_20$srate.py \
    -c pretrained/bytetrack_ablation.pth.tar -b 1 -d 1 --fp16 --fuse -s $srate --experiment-name extendkalmanfilter_mot20 --mot20 --stdp 0.05 --stdv 0.0125 --stda 0.00025
done
