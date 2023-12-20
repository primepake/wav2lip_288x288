from os.path import dirname, join, basename, isfile
from tqdm import tqdm
from time import time
import datetime

import math
import random

from models import SyncNet_color_384 as SyncNet
import audio
import pandas as pd
import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np

from glob import glob

import os, random, cv2, argparse
from hparams import hparams, get_image_list

import torch.multiprocessing as mp
import torch.distributed as dist
from pytorch_lightning.loggers import CSVLogger


parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator')

parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=False,default="checkpoints/syncnet/",type=str)
parser.add_argument('--exp_num', help='ID number of the experiment', required=False, default="actor", type=str)
parser.add_argument('--history_train', help='Save history training', required=False,default="logs/syncnet/",type=str)
parser.add_argument('--checkpoint_path', help='Resumed from this checkpoint', default=None, type=str)
args = parser.parse_args()


global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
best_loss = 1000
print('use_cuda: {}'.format(use_cuda))

syncnet_T = 5
syncnet_mel_step_size = 16
format_video = 'mov'
hparams.set_hparam("img_size", 384)

# mel augmentation
def mask_mel(crop_mel):
    block_size = 0.1
    time_size = math.ceil(block_size * crop_mel.shape[0])
    freq_size = math.ceil(block_size * crop_mel.shape[1])
    time_lim = crop_mel.shape[0] - time_size
    freq_lim = crop_mel.shape[1] - freq_size

    time_st = random.randint(0, time_lim)
    freq_st = random.randint(0, freq_lim)

    mel = crop_mel.copy()
    mel[time_st:time_st+time_size] = -4.
    mel[:, freq_st:freq_st + freq_size] = -4.

    return mel
def get_audio_length(audio_path):
    """Get the length of the audio file in seconds"""
    cmd = 'ffprobe -i {} -show_entries format=duration -v quiet -of csv="p=0"'.format(audio_path)
    audio_length = float(os.popen(cmd).read().strip())
    return audio_length


class Dataset(object):
    def __init__(self, file_list):
        self.all_videos = get_image_list(file_list)
    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)
        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, f'{frame_id:05}.jpg')
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames


    def crop_audio_window(self, spec, start_frame):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx : end_idx, :]
    
    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]
            img_names = list(glob(join(vidname, '*.jpg')))
                
            # print("len(img_names)):", len(img_names))
            if len(img_names) <= 3 * syncnet_T:
                continue
            
            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            
            count_same = 0
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)
                count_same += 1
                if count_same > 10:
                    break
            if count_same > 10:
                continue
            
            if random.choice([True, False]):
                y = torch.ones(1).float()
                chosen = img_name
            else:
                y = torch.zeros(1).float()
                chosen = wrong_img_name

            window_fnames = self.get_window(chosen)
            if window_fnames is None:
                # print("window_fnames")
                continue

            window = []
            all_read = True
            is_flip = random.random() < 0.5
            for fname in window_fnames:
                try:
                    img = cv2.imread(fname)

                    if is_flip:
                        img = cv2.flip(img, 1)
                    img = cv2.resize(img, (hparams.img_size, hparams.img_size))
                except Exception as e:
                    print(e)
                    all_read = False
                    break
                window.append(img)

            if not all_read:
                print("if not all_read:")
                continue


            try:
                mel_out_path = join(vidname, "mel.npy")
                if os.path.isfile(mel_out_path):  # x50 times faster - 0.002 -> 0.01s
                    with open(mel_out_path, "rb") as f:
                        orig_mel = np.load(f)
                else:
                    wavpath = os.path.join(vidname, "synced.wav")
                    wav = audio.load_wav(wavpath, hparams.sample_rate)

                    orig_mel = audio.melspectrogram(wav).T  # 0.2 -> 0.9s
                    with open(mel_out_path, "wb") as f:
                        np.save(f, orig_mel)
            except Exception as e:
                # print("mel", vidname)
                continue

            mel = self.crop_audio_window(orig_mel.copy(), img_name)

            # mel augmentation
            if random.random() < 0.3:
                mel = mask_mel(mel)

            del orig_mel

            if (mel.shape[0] != syncnet_mel_step_size):
                # print("Mel shape")
                continue

            # H x W x 3 * T
            # x = np.concatenate(window, axis=2) / 255. # [0, 1]
            x = (np.concatenate(window, axis=2) / 255.0)
            x = x.transpose(2, 0, 1)
            x = x[:, x.shape[1]//2:]

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)

            return x, mel, y

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss

def train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):

    global global_step, global_epoch
    resumed_step = global_step
    logger = CSVLogger(args.history_train, name=args.exp_num)

    stop_training = False
    while global_epoch < nepochs:
        st_e = time()
        try:
            print('Starting Epoch: {}'.format(global_epoch))
            running_loss = 0.
            for step, (x, mel, y) in enumerate(train_data_loader):
                
                st = time()
                model.train()
                optimizer.zero_grad()

                x = x.to(device)
            
                mel = mel.to(device)
                y = y.to(device)
                a, v = model(mel, x)
                loss = cosine_loss(a, v, y)
                loss.backward()
                optimizer.step()

                d = nn.functional.cosine_similarity(a, v)

                global_step += 1

                cur_session_steps = global_step - resumed_step
                running_loss += loss.item()

                print(f"Step {global_step} | out_of_sync_distance: {d.detach().cpu().clone().numpy().mean():.8f} | Loss: {running_loss/(step+1):.8f} | Elapsed: {(time() - st):.5f}")
                # if global_step == 1 or global_step % checkpoint_interval == 0:

                if global_step % 500 == 0:
                    with torch.no_grad():
                        model.eval()
                        eval_loss = eval_model(test_data_loader, global_step, device, model, checkpoint_dir)
                        # if eval_loss < 0.25:
                        #     stop_training = True
                    save_checkpoint(model, optimizer, global_step, checkpoint_dir, global_epoch, eval_loss)
                    logger.log_metrics({
                        "train_loss": running_loss / (step + 1),
                        "eval_loss": eval_loss
                        },step=global_step)
                    logger.save()
                    model.train()

                # prog_bar.set_description('Loss: {}'.format(running_loss / (step + 1)))
                #delete(x,mel,y)
                del x, mel, y
            if stop_training:
                print("The model has converged, stop training.")
                break
            print("Epoch time:", time() - st_e)
            global_epoch += 1
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            break
    save_checkpoint(model, optimizer, global_step, checkpoint_dir, global_epoch, 1000)
    logger.save()


def eval_model(test_data_loader, global_step, device, model, checkpoint_dir):
    eval_steps = 10
    print('Evaluating for {} steps'.format(eval_steps))
    losses = []
    for step, (x, mel, y) in enumerate(test_data_loader):
        print("Eval step", step)
        # model.eval()

        # Transform data to CUDA device
        x = x.to(device)

        mel = mel.to(device)

        a, v = model(mel, x)
        y = y.to(device)

        loss = cosine_loss(a, v, y)
        print("Eval loss", loss.item())
        losses.append(loss.item())

    averaged_loss = sum(losses) / len(losses)
    print(averaged_loss)

    return averaged_loss

def upload_file(path):
    pass

def save_ckpt(model, optimizer, step, checkpoint_dir, epoch, model_name):
    checkpoint_path = join(checkpoint_dir, model_name)
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
        "best_loss": best_loss,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch, loss_val):
    # save best.pth
    global best_loss
    date = str(datetime.datetime.now()).split(" ")[0]
    post_fix = f'checkpoint_{hparams.img_size}_{hparams.syncnet_batch_size}_{global_step:09d}_{date}.pth'
    if loss_val < best_loss:
        best_loss = loss_val
        save_ckpt(model, optimizer, step, checkpoint_dir, epoch, f"best_syncnet_{args.exp_num}.pth")

    # last model
    save_ckpt(model, optimizer, step, checkpoint_dir, epoch, f"last_syncnet_{args.exp_num}.pth")

    prefix = "syncnet_"
    save_ckpt(model, optimizer, step, checkpoint_dir, epoch, f"{prefix}{post_fix}")

    ckpt_list = os.listdir(checkpoint_dir)
    ckpt_list = [file for file in ckpt_list if prefix in file and "checkpoint_" in file and "syncnet_" in file]
    num_ckpts = hparams.num_checkpoints
    if len(ckpt_list) <= num_ckpts*2:
        return

    ckpt_list.sort(key=lambda x: int(x.replace(".pth", "").split("_")[-2]))
    num_elim = len(ckpt_list) - num_ckpts
    elim_ckpt = ckpt_list[:num_elim]
    for ckpt in elim_ckpt:
        ckpt_path = os.path.join(checkpoint_dir, ckpt)
        os.remove(ckpt_path)
        print("Deleted", ckpt_path)

def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_checkpoint(path, model, optimizer, reset_optimizer=False):
    global global_step
    global global_epoch
    global best_loss

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]
    best_loss = checkpoint["best_loss"]

    return model


def run():
    # global global_step

    checkpoint_dir = os.path.join(args.checkpoint_dir, args.exp_num)
    checkpoint_path = args.checkpoint_path

    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)

    train_dataset = Dataset('filelists/train.txt')
    test_dataset = Dataset('filelists/test.txt')
    hparams.set_hparam("syncnet_batch_size", 64)
    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=hparams.syncnet_batch_size, shuffle=True,
        num_workers=hparams.num_workers,
        drop_last=True
    )

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.syncnet_batch_size,
        num_workers=hparams.num_workers,
        drop_last=True
    )

    device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    model = nn.DataParallel(SyncNet()).to(device)
    # model = SyncNet().to(device)

    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))


    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.syncnet_lr)

    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False)
    else:
        print("Training From Scratch !!!")

    # model = nn.DataParallel(model).to(device)

    train(device, model, train_data_loader,test_data_loader, optimizer,
          checkpoint_dir=checkpoint_dir,
          checkpoint_interval=hparams.syncnet_checkpoint_interval,
          nepochs=hparams.nepochs)


if __name__ == "__main__":
    run()