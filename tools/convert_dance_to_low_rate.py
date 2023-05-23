import os
import argparse
from glob import glob

from tools.sampling_low_rate import sampling_low_rate_json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='sampling low-rate images from high-rate images')
    parser.add_argument(
        '-s', '--sampling_rate', help="Target sampling rate", type=int, default=1)
    args = parser.parse_args()

    # if args.sampling_rate == 1:
    #     print("No need to sampling")
    #     exit(0)

    gtfiles = glob("./datasets/DanceTrack/train/dancetrack*")
    if not os.path.exists("./datasets/dancetrack_%d/train" % args.sampling_rate):
        os.makedirs("./datasets/dancetrack_%d/train" % args.sampling_rate)
    else:
        os.system("rm -rf ./datasets/dancetrack_%d/train/*" % args.sampling_rate)
    targetfiledir = "./datasets/dancetrack_%d/train" % args.sampling_rate
    targetfiles = os.listdir("./datasets/dancetrack_%d/train/" % args.sampling_rate) # [gtfile.split("/")[-1] for gtfile in gtfiles]

    for gtfile in gtfiles:
        if gtfile.split("/")[-1] in targetfiles:
            continue
        print("sampling %s" % gtfile)
        os.system("python3 ./tools/sampling_low_rate.py -s %d -g %s -t %s" % (args.sampling_rate, gtfile, os.path.join(targetfiledir, gtfile.split("/")[-1])))

    gtfiles = glob("./datasets/DanceTrack/val/dancetrack*")
    if not os.path.exists("./datasets/dancetrack_%d/val" % args.sampling_rate):
        os.makedirs("./datasets/dancetrack_%d/val" % args.sampling_rate)
    else:
        os.system("rm -rf ./datasets/dancetrack_%d/val/*" % args.sampling_rate)
    targetfiledir = "./datasets/dancetrack_%d/val" % args.sampling_rate
    targetfiles = os.listdir("./datasets/dancetrack_%d/val/" % args.sampling_rate) # [gtfile.split("/")[-1] for gtfile in gtfiles]

    for gtfile in gtfiles:
        if gtfile.split("/")[-1] in targetfiles:
            continue
        print("sampling %s" % gtfile)
        os.system("python3 ./tools/sampling_low_rate.py -s %d -g %s -t %s" % (args.sampling_rate, gtfile, os.path.join(targetfiledir, gtfile.split("/")[-1])))
