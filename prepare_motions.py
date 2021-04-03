import json
import os
import random
import copy
from PIL import Image
import torch
import math
import numpy as np
import multiprocessing
from multiprocessing import Pool
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from auto_augment import AutoAugment, auto_augment_policy
from utils import get_logger
from tqdm import tqdm


with open('data/train-tracks.json') as f:
        tracks = json.load(f)
list_of_uuids = list(tracks.keys())
list_of_tracks = list(tracks.values())
train_num = len(list_of_uuids)
#cv
with open('data/test-tracks.json') as f:
    unlabel_tracks = json.load(f)
list_of_uuids.extend(unlabel_tracks.keys())
list_of_tracks.extend(unlabel_tracks.values())

print('#track id (class): %d '%len(list_of_tracks))
#motion
def save_motion(data):
            track_idx, track  = data
            interpret = max(math.floor(len(track["frames"])/5), 1)
            for frame_idx, frame in enumerate(track["frames"]):
                frame_path = os.path.join('data/cityflow/MTMC', track["frames"][frame_idx])
                frame = Image.open(frame_path).convert('RGB')
                if frame_idx==0:
                    mean_frame = np.asarray(frame)/255
                else:
                    mean_frame += np.asarray(frame)/255
            mean_frame = mean_frame/len(track["frames"]) * 255

'''
            for frame_idx, frame in enumerate(track["frames"]):
                if not frame_idx % interpret==0:
                    continue
                frame_path = os.path.join('data/cityflow/MTMC', track["frames"][frame_idx])
                frame = Image.open(frame_path).convert('RGB')
                box = track['boxes'][frame_idx]
                w = frame.size[0]
                h = frame.size[1]
                pad_w, pad_h = round(box[2]/10), round(box[3]/10)
                x1, y1 = max(0, box[0]-pad_w), max(0, box[1]-pad_h)
                x2, y2 = min(box[0] + box[2]+pad_w, w-1), min(box[1] + box[3]+pad_h, h-1)
                mean_frame[y1:y2, x1:x2,:] = 0 # black edge
                pad = 0
                x1, y1 = max(0, box[0]-pad), max(0, box[1]-pad)
                x2, y2 = min(box[0] + box[2]+pad, w-1), min(box[1] + box[3]+pad, h-1)
                crop = frame.crop( (x1,y1,x2,y2))
                next_crop = np.asarray(crop)
                mean_frame[y1:y2, x1:x2,:] = next_crop
'''
            mean_frame = Image.fromarray(np.uint8(mean_frame))
            mean_frame = mean_frame.resize((w//2, h//2) , Image.BICUBIC)
            save_path = './motions/%04d.jpg'%track_idx
            mean_frame.save(save_path)

track_iter = enumerate(list_of_tracks)
with Pool(16) as p:
    p.map(save_motion, track_iter )
