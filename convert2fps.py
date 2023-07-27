import os
import argparse
import shutil
import json, subprocess, random, string
import cv2
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', help='dataset path', default='./videos', type=str)
parser.add_argument('--fps', help='Frame per second', default=25, type=int)
args = parser.parse_args()

# cmd = f"ffmpeg -y -r 25 -i {new_name} 25fps.mp4"

def convertVideo2Fps(src):
    cmd = f"ffmpeg -i {src} -r 25 25fps.mp4 -y"
    # cmd = f"ffmpeg -y -r 25 -i {src} 25fps.mp4"
    os.system(cmd)
    print(f'remove src: {src}')
    os.rename(src, 'temp.mp4')
    try:
        shutil.move('25fps.mp4', f"{src}")
        os.remove('temp.mp4')
    except:
        os.rename('temp.mp4', src)
        print(f'file {src} wrong processing')
    
def check_fps(filename):
    cap = cv2.VideoCapture(filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps
    

data_root = args.data_root
fps = args.fps
identities = os.listdir(data_root)
identities = [x if not x.endswith('.DS_Store') else os.remove(os.path.join(data_root, x)) for x in identities]
identities.sort()
for identity in identities:
    # if int(identity) < 6:
    #     continue
    fullPathID = os.path.join(data_root, identity)
    videos = os.listdir(fullPathID)
    videos = [x if x.endswith('.mp4') else os.remove(os.path.join(data_root, identity, x)) for x in videos]
    i = 0
    for video in videos:
        src = os.path.join(fullPathID, video)
        if check_fps(src) == fps:
            continue
        print(f'Convert video: {src} to 25 fps')
        convertVideo2Fps(src)
        
