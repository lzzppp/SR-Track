for srate in 1 2 3 4 5 6 7 8 9 10
do
    python3 tools/convert_mot17_to_low_rate.py -s $srate
    python3 tools/convert_mot17_to_coco.py -d ./datasets/mot_$srate
done
