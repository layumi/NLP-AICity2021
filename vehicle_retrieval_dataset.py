#!/usr/bin/env python
# COPYRIGHT 2020. Fred Fung. Boston University.
"""
PyTorch dataset for CityFlow-NL.
"""
import json
import os
import random

#import cv2
#cv2.setNumThreads(0)
import copy
from PIL import Image
import torch
import math
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from auto_augment import AutoAugment, auto_augment_policy 
from utils import get_logger

class CityFlowNLDataset(Dataset):
    def __init__(self, data_cfg, multi=10):
        """
        Dataset for training.
        :param data_cfg: CfgNode for CityFlow NL.
        """
        self.multi = multi
        self.data_cfg = data_cfg
        self.aug = AutoAugment(auto_augment_policy(name='v0r', hparams=None))
        with open(self.data_cfg.JSON_PATH) as f:
            tracks = json.load(f)
        f.close()
        self.list_of_uuids = list(tracks.keys())
        self.list_of_tracks = list(tracks.values())
        self.list_of_crops = list()
        self.transform = transforms.Compose(
                       [transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 
        print('#track id (class): %d '%len(self.list_of_tracks))
        for track_idx, track in enumerate(self.list_of_tracks):
            for frame_idx, frame in enumerate(track["frames"]):
                frame_path = os.path.join(self.data_cfg.CITYFLOW_PATH, frame)
                #nl_idx = int(random.uniform(0, 3))
                nl = track["nl"]#[nl_idx]
                box = track["boxes"][frame_idx]
                crop = {"track_id": track_idx, "frame": frame_path, "nl": nl, "box": box}
                self.list_of_crops.append(crop)
        self._logger = get_logger()

    def __len__(self):
        return len(self.list_of_tracks)*self.multi

    def __getitem__(self, index):
        """
        Get pairs of NL and cropped frame.
        """
        index = math.ceil(index/self.multi) 
        if random.uniform(0, 1) > self.data_cfg.POSITIVE_THRESHOLD:
            label = 1 #positive
        else:
            label = 0 #negative
        track = self.list_of_tracks[index]
        frame_idx = int(random.uniform(0, len((track["frames"]))))
        frame_path = os.path.join(self.data_cfg.CITYFLOW_PATH, track["frames"][frame_idx])
        if not os.path.isfile(frame_path):
            self._logger.warning("Missing Image File: %s" % track["frames"][frame_idx])
            label = 0
            crop = torch.randn(size=(3,) + self.data_cfg.CROP_SIZE)
        else:
            frame = Image.open(frame_path).convert('RGB')
            box = track["boxes"][frame_idx]
            #try:
            w = frame.size[0]
            h = frame.size[1]
            pad = 5
            crop_id =  index # dp["track_id"]
            crop = frame.crop( (max(0, box[1]-pad) , max(0, box[0]-pad),
               min(box[1] + box[3]+pad, w-1), min(box[0] + box[2]+pad, h-1) ))
            frame.close() # clean
            crop = crop.resize((self.data_cfg.CROP_SIZE, self.data_cfg.CROP_SIZE) , Image.BICUBIC)
            crop = self.aug(crop)
            crop = self.transform(crop)
            #crop = torch.from_numpy(crop).permute([2, 0, 1]).to(
            #    dtype=torch.float32)
        nl_id = index #dp["track_id"]
        nl_idx = int(random.uniform(0, 3))
        nl = track["nl"][nl_idx]
        if label != 1:
            nsample = random.sample(self.list_of_crops, 1)
            nl_idx = int(random.uniform(0, len(nsample[0]["nl"])))
            nl = nsample[0]["nl"][nl_idx]
            nl_id = nsample[0]["track_id"]
            del nsample
        nl = '[CLS]' + nl + '[SEP]'
        #label = torch.Tensor([label]).to(dtype=torch.float32) # only 0,1
        #dp["token"] = self.bert_tokenizer.batch_encode_plus([dp["nl"]], padding='longest',return_tensors='pt')
        return nl, crop, nl_id, crop_id, label


class CityFlowNLInferenceDataset(Dataset):
    def __init__(self, data_cfg):
        """Dataset for evaluation. Loading tracks instead of frames."""
        self.data_cfg = data_cfg
        with open(self.data_cfg.EVAL_TRACKS_JSON_PATH) as f:
            tracks = json.load(f)
        self.list_of_uuids = list(tracks.keys())
        self.list_of_tracks = list(tracks.values())
        self._logger = get_logger()

    def __len__(self):
        return len(self.list_of_uuids)

    def __getitem__(self, index):
        """
        :return: a dictionary for each track:
        id: uuid for the track
        frames, boxes, nl are untouched from the input json file.
        crops: A Tensor of cropped images from the track of shape
            [length, 3, crop_w, crop_h].
        """
        dp = {"id": self.list_of_uuids[index]}
        dp.update(self.list_of_tracks[index])
        cropped_frames = []
        for frame_idx, frame_path in enumerate(dp["frames"]):
            frame_path = os.path.join(self.data_cfg.CITYFLOW_PATH, frame_path)
            if not os.path.isfile(frame_path):
                continue
            #frame = cv2.imread(frame_path)
            #box = dp["boxes"][frame_idx]
            #crop = frame[box[1]:box[1] + box[3], box[0]: box[0] + box[2], :]
            #crop = cv2.resize(crop, dsize=self.data_cfg.CROP_SIZE)
            #crop = torch.from_numpy(crop).permute([2, 0, 1]).to(
            #    dtype=torch.float32)
            frame = Image.open(dp["frame"]).convert('RGB')
            box = dp["box"]
            #try:
            w = frame.size[0]
            h = frame.size[1]
            pad = 5
            crop = frame.crop( (max(0, box[1]-pad) , max(0, box[0]-pad),
               min(box[1] + box[3]+pad, w-1), min(box[0] + box[2]+pad, h-1) ))
            #except:
            #    print(dp["frame"])
            crop = crop.resize((self.data_cfg.CROP_SIZE, self.data_cfg.CROP_SIZE) , Image.BICUBIC)
            crop = self.transform(crop)
            cropped_frames.append(crop)
        dp["crops"] = torch.stack(cropped_frames, dim=0)
        return dp
