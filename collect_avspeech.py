import shutil
from pytube import YouTube
from pytube import exceptions
import os
import argparse
import csv
import time

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_file', help='url video', default='dataset_csv/vietnam_train.csv', type=str)
parser.add_argument('--is_vn', help='if vietnam dataset',  action='store_true', default=False)
parser.add_argument('--dataset_out', help='folder contain out dataset', default='', type=str)
parser.add_argument('--log_file', help='log_file', default='', type=str)
args = parser.parse_args()

def downloadYouTube(youtube_link, youtube_id, path, start_segment, end_segment, downloadedYoutubeID):

    try:
        print(f'Try to download video: {youtube_link}')
        yt_obj = YouTube(youtube_link)
        filters = yt_obj.streams.filter(progressive=True, file_extension='mp4').get_highest_resolution()
    except exceptions.VideoUnavailable:
        print(f'Video {youtube_link} is unavaialable, skipping.')
        addNewFiles.write(f"{youtube_id}\n")
        return 0
    except KeyError as e:
        print(e)
        addNewFiles.write(f"{youtube_id}\n")
        return 0
    else:
        out_file = filters.download(filename = 'temp_download_video.mp4')
        print(f'outfile: {out_file}')
        print(f'start_time {start_segment} to {end_segment}')
        cmd = f"ffmpeg -i temp_download_video.mp4 -ss {start_segment} -to {end_segment} temp.mp4"
        os.system(cmd)
        os.remove(f'temp_download_video.mp4')
        cnt = 0
        new_name = str(cnt).zfill(5) + '.mp4'
        while os.path.exists(os.path.join(output_id,  new_name)):
            cnt += 1
            new_name = str(cnt).zfill(5) + '.mp4'
        
        print(f"Move file temp.mp4 to {os.path.join(output_id, new_name)}")
        shutil.move('temp.mp4', os.path.join(output_id, new_name))

        print("Video Downloaded Successfully")
    addNewFiles.write(f"{youtube_id}\n")
    return 1


videos = []
dataset_out = args.dataset_out
dataset_file = args.dataset_file
log_file = args.log_file
if not os.path.exists(log_file):
    print(f"Create log file {log_file}")
    os.system(f'touch {log_file}')
downloadedYoutubeID = open(log_file, 'r').read().split("\n")
print(f"Downloaded links {downloadedYoutubeID}")

addNewFiles = open(log_file, 'a')
if args.is_vn:
    print("Download Vietnam dataset")

with open(f"{dataset_file}", "r") as f:
    reader = csv.reader(f, delimiter="\t")
   
    for i, line in enumerate(reader):
        
        # youtube_link = f"https://www.youtube.com/watch?v={1Uf_F74fBns}"
        if args.is_vn:
            print(line[0])
            channel, youtube_id, start_segment, end_segment = line[0].split(",")
            output_id = os.path.join(dataset_out, channel)
        else:
            youtube_id, start_segment, end_segment, x, y = line[0].split(",")
            output_id = os.path.join(dataset_out, youtube_id)

        if youtube_id in downloadedYoutubeID:
            print("file downloaded, so continue...")
            continue
        if not os.path.exists(output_id):
            os.mkdir(output_id)
        else:
            if len(os.listdir(output_id)) > 0:
                print(f'{youtube_id} is downloaded so continue')
                continue
        youtube_link = f"https://www.youtube.com/watch?v={youtube_id}"
        status = downloadYouTube(youtube_link, youtube_id, output_id, start_segment, end_segment, addNewFiles)
        if status == 0:
            print(f'VideoUnavailable so delete {youtube_id}')
            os.rmdir(output_id)
        time.sleep(10)

addNewFiles.close()