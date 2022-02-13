import cv2
import os
import shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from mtcnn import MTCNN


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', help='dataset', default='videos', type=str)
args = parser.parse_args()

detector = MTCNN()
def convertVideo2Fps(src):
    cmd = f"ffmpeg -y -r 25 -i {src} 25fps.mp4"
    os.system(cmd)
    print(f'remove src: {src}')
    shutil.move(src, 'temp.mp4')
    try:
        shutil.move('25fps.mp4', f"{src}")
        os.remove('temp.mp4')
        return True
    except:
        os.rename('temp.mp4', src)
        print(f'file {src} wrong processing')
    return False

def get_portion_video(video):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps
    if duration//60 > 2:
        print('fps: ', fps)
        flag = True
        if int(fps) != 25:
            flag = convertVideo2Fps(video)
        if flag:
            ret, frame = cap.read()
            count = 0
            skip = 60*fps
            start = None
            splitTimes = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                count += 1
                if count < skip or count%20 != 0:
                    continue
                print(f'processing frame {count}th !!!')
                # cv2.imshow('frame', frame)
                result = detector.detect_faces(frame)
                
                print(f'frame {count}th has {len(result)} faces')
                if len(result) != 0 and start is None:
                    if result[0]['confidence'] < 0.8:
                        continue
                    start = count//fps
                elif (len(result) == 0 and start) or (start != None and (count//fps - start > 60)):
                    if count//fps - 1 - start > 10:
                        splitTimes.append((start, count//fps - 1))
                        print(f'split at {start} - {count//fps -1}')
                    start = None
                # if cv2.waitKey(25) & 0xFF == ord('q'):
                #     break
            cap.release()
            return splitTimes
def splitVideoByTime(splitTimes, video):
    count = 0
    baseName = video.split('.')[0]
    if len(splitTimes) > 0:
        for splitTime in splitTimes:
            new_name = f"{baseName+'-'+str(count)}.mp4"
            cmd = f"ffmpeg -i {video} -ss {splitTime[0]} -to {splitTime[1]}{new_name}"
            print(f'CMD: {cmd}')
            os.system(cmd)
            count += 1
        # os.remove(video)

data_root = args.data_root
identities = os.listdir(data_root)
identities = [x if not x.endswith('.DS_Store') else os.remove(os.path.join(data_root, x)) for x in identities]
identities.sort()
for identity in identities:
    print(f'Processing dataset {identity}')
    videos = os.listdir(os.path.join(data_root, identity))
    videos = [x if x.endswith('.mp4') else os.remove(os.path.join(data_root, identity, x))  for x in videos]
    videos.sort()
    root_manifest = os.path.join('manifest', identity)
    if not os.path.exists(root_manifest):
        os.mkdir(root_manifest)
    for video in videos:
        fullPathVideo = os.path.join(data_root, identity, video)
        print(f'processing video {fullPathVideo}')
        manifest = os.path.join(root_manifest, video.split('.')[0] + '.text')
        splitTimes = []
        if not os.path.exists(manifest):
            splitTimes = get_portion_video(fullPathVideo)
            with open(manifest, 'w') as f:
                for time in splitTimes:
                    f.write(f"{time[0]} {time[1]}\n")
        else:
            # splitTimes = []
            with open(manifest, 'r') as f:
                content = f.readlines()
                for line in content:
                    numbers = line.split(' ')
                    start, end = int(float(numbers[0])), int(float(numbers[1]))
                    splitTimes.append((start, end))
        print(f'run split video {fullPathVideo}')
        print(f'list time: {splitTimes}')
        splitVideoByTime(splitTimes, fullPathVideo)
    
