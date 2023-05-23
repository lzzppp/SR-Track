for srate in 1 2 3 4 5 6 7 8 9 10
do
    python3 tools/track_dyte.py -f exps/example/dance/yolox_x_ablation_$srate.py \
    -c pretrained/bytetrack_model.pth.tar -b 1 -d 1 --fp16 --fuse -s $srate --experiment-name extendkalmanfilter_dancetrack --dance \
    --stdp 0.25 --stda 0.015 --match_thresh_d1 1.0
done
