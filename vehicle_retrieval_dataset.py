#!/usr/bin/env python
# COPYRIGHT 2020. Fred Fung. Boston University.
"""
PyTorch dataset for CityFlow-NL.
"""
import json
import os
import random
import copy
from PIL import Image
import torch
import math
from random_erasing import RandomErasing
import numpy as np
import multiprocessing
from multiprocessing import Pool
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from auto_augment import AutoAugment, auto_augment_policy 
from utils import get_logger
from tqdm import tqdm

class CityFlowNLDataset(Dataset):
    def __init__(self, data_cfg, multi=10):
        """
        Dataset for training.
        :param data_cfg: CfgNode for CityFlow NL.
        """
        self.multi = multi
        self.all3 = data_cfg.all3
        self.motion = data_cfg.motion
        self.nseg = data_cfg.nseg
        self.data_cfg = data_cfg
        self.aug = AutoAugment(auto_augment_policy(name='v0r', hparams=None))
        with open(self.data_cfg.JSON_PATH) as f:
            tracks = json.load(f)
        f.close()
        self.list_of_uuids = list(tracks.keys())
        self.list_of_tracks = list(tracks.values())
        self.list_of_crops = list()
        train_num = len(self.list_of_uuids)
        self.transform = transforms.Compose(
                       [
                        transforms.Pad(10),
                        transforms.RandomCrop((data_cfg.CROP_SIZE, self.data_cfg.CROP_SIZE)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        RandomErasing(probability=0.5) ])

        if data_cfg.semi:
            #cv
            with open(self.data_cfg.EVAL_TRACKS_JSON_PATH) as f:
                unlabel_tracks = json.load(f)
            f.close()
            self.list_of_uuids.extend(unlabel_tracks.keys())
            self.list_of_tracks.extend(unlabel_tracks.values())
            #nl
            with open("data/test-queries.json", "r") as f:
                unlabel_nl = json.load(f)
            unlabel_nl_key = list(unlabel_nl.keys())

        print('#track id (class): %d '%len(self.list_of_tracks))
        count = 0
        # add id and nl, -1 for unlabeled data
        for track_idx, track in enumerate(self.list_of_tracks):
            track["track_id"] = track["id"] # Using clear trainning json Or if you want to test old json, using track_idx
            track["nl_id"] = track["id"]
            # from 0 to train_num-1 is the id of the original training set. 
            if track_idx>=train_num:
                track["nl_id"] = -1
                track["nl"] = unlabel_nl[unlabel_nl_key[count]]
                count = count+1
        self._logger = get_logger()

    def __len__(self):
        return len(self.list_of_tracks)*self.multi

    def __getitem__(self, index):
        """
        Get pairs of NL and cropped frame.
        """
        index = math.floor(index/self.multi) 
        #if random.uniform(0, 1) > self.data_cfg.POSITIVE_THRESHOLD:
        label = 1 #positive
        #else:
        #    label = 0 #negative
        track = self.list_of_tracks[index]
        
        nseg = self.nseg
        length = len((track["frames"])) // nseg
        nmotion = torch.zeros((nseg, 3, self.data_cfg.CROP_SIZE, self.data_cfg.CROP_SIZE))
        ncrops = torch.zeros((nseg, 3, self.data_cfg.CROP_SIZE, self.data_cfg.CROP_SIZE))
        for i in range(nseg):
            if i*length <=len(track["frames"]):
                frame_idx = int(random.uniform(i*length, min( (i+1)*length, len((track["frames"])))))
            else: 
                frame_idx = len((track["frames"])) -1
            frame_path = os.path.join(self.data_cfg.CITYFLOW_PATH, track["frames"][frame_idx])
            frame = Image.open(frame_path).convert('RGB')
            if self.motion:
                motion = frame.resize((self.data_cfg.CROP_SIZE, self.data_cfg.CROP_SIZE) , Image.BICUBIC)
                motion = self.transform(motion)
                nmotion[i,:,:,:] = motion
        #frame_idx = int(random.uniform(0, len(track["frames"])))
        #frame_path = os.path.join(self.data_cfg.CITYFLOW_PATH, track["frames"][frame_idx])
        #if not os.path.isfile(frame_path):
        #    self._logger.warning("Missing Image File: %s" % track["frames"][frame_idx])
        #    crop = torch.randn(size=(3,) + self.data_cfg.CROP_SIZE)
        #else:
            #frame = Image.open(frame_path).convert('RGB')
            box = track["boxes"][frame_idx]
            #try:
            w = frame.size[0]
            h = frame.size[1]
            pad = 5
            x1, y1 = max(0, box[0]-pad), max(0, box[1]-pad)
            x2, y2 = min(box[0] + box[2]+pad, w-1), min(box[1] + box[3]+pad, h-1)
            crop_id = track["track_id"]
            crop = frame.crop( ( x1, y1, x2, y2 ) )
            frame.close() # clean
            crop = crop.resize((self.data_cfg.CROP_SIZE, self.data_cfg.CROP_SIZE) , Image.BICUBIC)
            crop = self.transform(crop)
            ncrops[i,:,:,:] = crop
            #crop = torch.from_numpy(crop).permute([2, 0, 1]).to(
            #    dtype=torch.float32)
        nl_id = track["nl_id"]
        nl_total = '[CLS]'
        if self.all3: 
            rand_idx = np.random.permutation(len(track["nl"]))
            for j in range(3): 
                idx = rand_idx[j]
                nl = track["nl"][idx]
                nl_total += nl.replace('Sedan', 'sedan').replace('suv','SUV').replace('Suv','SUV').replace('Jeep','jeep').replace('  ',' ') + '[SEP]'
        else:
            nl_idx = int(random.uniform(0, len(track["nl"])))
            nl = track["nl"][nl_idx]
            nl_total += nl.replace('Sedan', 'sedan').replace('suv','SUV').replace('Suv','SUV').replace('Jeep','jeep').replace('  ',' ') + '[SEP]'
        #label = torch.Tensor([label]).to(dtype=torch.float32) # only 0,1
        if self.motion:
            return nl_total, ncrops, nmotion, nl_id, crop_id, label
        return nl_total, ncrops, nl_id, crop_id, label

class CityFlowNLInferenceDataset(Dataset):
    def __init__(self, data_cfg):
        """Dataset for evaluation. Loading tracks instead of frames."""
        self.data_cfg = data_cfg
        with open(self.data_cfg.EVAL_TRACKS_JSON_PATH) as f:
            tracks = json.load(f)
        self.list_of_uuids = list(tracks.keys())
        self.list_of_tracks = list(tracks.values())
        self._logger = get_logger()
        self.motion = data_cfg.motion
        self.all3 = data_cfg.all3
        self.nseg = data_cfg.nseg
        self.transform = transforms.Compose(
                       [transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    def __len__(self):
        return len(self.list_of_uuids)

    def get_concat_h_cut(self, im1, im2):
        dst = Image.new('RGB', (im1.width + im2.width, min(im1.height, im2.height)))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst


    def read_img(self, data):
        frame_idx, frame_path, frame_box = data
        if not os.path.isfile(frame_path):
            print('missing %s'%frame_path)
        frame = Image.open(frame_path).convert('RGB')
        box = frame_box
        w = frame.size[0]
        h = frame.size[1]
        pad = 5
        crop = frame.crop( (max(0, box[0]-pad), max(0, box[1]-pad), 
               min(box[0] + box[2]+pad, w-1), min(box[1] + box[3]+pad, h-1)))
        crop = crop.resize((self.data_cfg.CROP_SIZE, self.data_cfg.CROP_SIZE) , Image.BICUBIC)
        #if frame_idx == 0:
        save_path = './crops/%s.jpg'%self.one_id
        if not os.path.isfile(save_path):
            frame_save = frame.resize((self.data_cfg.CROP_SIZE, self.data_cfg.CROP_SIZE) , Image.BICUBIC)
            save_path_motion = './crops/%s.jpg'%self.one_id
            frame_save = self.get_concat_h_cut(crop, frame_save)
            frame_save.save(save_path)
        frame.close()
        crop = self.transform(crop)
        self.cropped_frames[frame_idx,:,:,:] = crop
        #motion = self.transform(motion)
        #self.cropped_motions[frame_idx,:,:,:] = motion
        return


    def __getitem__(self, index):
        """
        :return: a dictionary for each track:
        id: uuid for the track
        frames, boxes, nl are untouched from the input json file.
        crops: A Tensor of cropped images from the track of shape
            [length, 3, crop_w, crop_h].
        """
        dp = {"id": self.list_of_uuids[index]}
        self.one_id = self.list_of_uuids[index]
        dp.update(self.list_of_tracks[index])
        track = dp
        num = len((track["frames"]))
        if self.nseg == 1: # we sample 4 images average
            nseg = 4
            nmotion = torch.zeros((min(nseg, num), 3, self.data_cfg.CROP_SIZE, self.data_cfg.CROP_SIZE))
            self.cropped_frames = torch.zeros((min(nseg, num), 3, self.data_cfg.CROP_SIZE, self.data_cfg.CROP_SIZE))
        else:
            nseg = self.nseg # we sample the same images as training
            nmotion = torch.zeros((nseg, 3, self.data_cfg.CROP_SIZE, self.data_cfg.CROP_SIZE))
            self.cropped_frames = torch.zeros((nseg, 3, self.data_cfg.CROP_SIZE, self.data_cfg.CROP_SIZE))
        length = len((track["frames"])) // nseg
        frame_idx_iter, frame_path_iter, frame_box_iter = [],[],[]
        if self.nseg!=1 or len(track["frames"]) > nseg:
            idx = np.floor(np.linspace(0, len(track["frames"]), num=nseg, endpoint=False))
            for i in range(nseg):
                frame_idx = idx[i].astype(int)
                frame_path = os.path.join(self.data_cfg.CITYFLOW_PATH, track["frames"][frame_idx])
                if self.motion:
                    print('load motion')
                    frame = Image.open(frame_path).convert('RGB')
                    motion = frame.resize((self.data_cfg.CROP_SIZE, self.data_cfg.CROP_SIZE) , Image.BICUBIC)
                    motion = self.transform(motion)
                    nmotion[i,:,:,:] = motion
                frame_idx_iter.append(i)
                frame_path_iter.append(frame_path)
                frame_box_iter.append(dp["boxes"][frame_idx])
        else:
            for i in range(len(track["frames"])):
                frame_path = os.path.join(self.data_cfg.CITYFLOW_PATH, track["frames"][i])
                if self.motion:
                    frame = Image.open(frame_path).convert('RGB')
                    motion = frame.resize((self.data_cfg.CROP_SIZE, self.data_cfg.CROP_SIZE) , Image.BICUBIC)
                    motion = self.transform(motion)
                    nmotion[i,:,:,:] = motion
                frame_idx_iter.append(i)
                frame_path_iter.append(frame_path)
                frame_box_iter.append(dp["boxes"][i])

        if nseg<80:
            for data in zip(frame_idx_iter, frame_path_iter, frame_box_iter):
                self.read_img(data)
        else:
            with Pool(4) as p:
                p.map(self.read_img, zip(frame_idx_iter, frame_path_iter, frame_box_iter) )
        crops = self.cropped_frames
        #motions = self.cropped_motions
        if self.motion:
            return [crops, nmotion], self.list_of_uuids[index]
        return crops, self.list_of_uuids[index]


class VAL_CityFlowNLDataset(Dataset):
    def __init__(self, data_cfg, multi=1, nl=False):
        """
        Dataset for training.
        :param data_cfg: CfgNode for CityFlow NL.
        """
        self.nl = nl 
        self.multi = multi
        self.motion = data_cfg.motion
        self.nseg = data_cfg.nseg
        self.all3 = data_cfg.all3
        self.data_cfg = data_cfg
        self.aug = AutoAugment(auto_augment_policy(name='v0r', hparams=None))
        with open(self.data_cfg.JSON_PATH) as f:
            tracks = json.load(f)
        f.close()
        self.list_of_uuids = list(tracks.keys())
        self.list_of_tracks = list(tracks.values())
        self.list_of_crops = list()
        train_num = len(self.list_of_uuids)
        self.transform = transforms.Compose(
                       [
                        transforms.Pad(10),
                        transforms.RandomCrop((data_cfg.CROP_SIZE, self.data_cfg.CROP_SIZE)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        RandomErasing(probability=0.5) ])

        if data_cfg.semi:
            #cv
            with open(self.data_cfg.EVAL_TRACKS_JSON_PATH) as f:
                unlabel_tracks = json.load(f)
            f.close()
            self.list_of_uuids.extend(unlabel_tracks.keys())
            self.list_of_tracks.extend(unlabel_tracks.values())
            #nl
            with open("data/test-queries.json", "r") as f:
                unlabel_nl = json.load(f)
            unlabel_nl_key = list(unlabel_nl.keys())

        print('#track id (class): %d '%len(self.list_of_tracks))
        count = 0
        # add id and nl, -1 for unlabeled data
        for track_idx, track in enumerate(self.list_of_tracks):
            track["track_id"] = track_idx
            track["nl_id"] = track_idx
            # from 0 to train_num-1 is the id of the original training set. 
            if track_idx>=train_num:
                track["nl_id"] = -1
                track["nl"] = unlabel_nl[unlabel_nl_key[count]]
                count = count+1
        self._logger = get_logger()

    def __len__(self):
        return len(self.list_of_tracks)*self.multi

    def get_concat_h_cut(self, im1, im2):
        dst = Image.new('RGB', (im1.width + im2.width, min(im1.height, im2.height)))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    def read_img(self, data):
        frame_idx, frame_path, frame_box = data
        if not os.path.isfile(frame_path):
            print('missing %s'%frame_path)
        frame = Image.open(frame_path).convert('RGB')
        box = frame_box
        w = frame.size[0]
        h = frame.size[1]
        pad = 5
        crop = frame.crop( (max(0, box[0]-pad), max(0, box[1]-pad),
               min(box[0] + box[2]+pad, w-1), min(box[1] + box[3]+pad, h-1)))
        crop = crop.resize((self.data_cfg.CROP_SIZE, self.data_cfg.CROP_SIZE) , Image.BICUBIC)
        save_path = './val_crops/%s.jpg'%self.one_id
        if not os.path.isfile(save_path):
            frame_save = frame.resize((self.data_cfg.CROP_SIZE, self.data_cfg.CROP_SIZE) , Image.BICUBIC)
            save_path_motion = './val_crops/%s.jpg'%self.one_id
            frame_save = self.get_concat_h_cut(crop, frame_save)
            frame_save.save(save_path)
        frame.close()
        crop = self.transform(crop)
        self.cropped_frames[frame_idx,:,:,:] = crop
        #motion = self.transform(motion)
        #self.cropped_motions[frame_idx,:,:,:] = motion
        return

    def __getitem__(self, index):
        """
        Get pairs of NL and cropped frame.
        """
        index = math.floor(index/self.multi) 
        #if random.uniform(0, 1) > self.data_cfg.POSITIVE_THRESHOLD:
        label = 1 #positive
        #else:
        #    label = 0 #negative
        track = self.list_of_tracks[index]
        num = len((track["frames"]))
        if self.nseg == 1: # we sample 4 images average
            nseg = 4
            nmotion = torch.zeros((min(nseg, num), 3, self.data_cfg.CROP_SIZE, self.data_cfg.CROP_SIZE))
            self.cropped_frames = torch.zeros((min(nseg, num), 3, self.data_cfg.CROP_SIZE, self.data_cfg.CROP_SIZE))
        else:
            nseg = self.nseg # we sample the same images as training
            nmotion = torch.zeros((nseg, 3, self.data_cfg.CROP_SIZE, self.data_cfg.CROP_SIZE))
            self.cropped_frames = torch.zeros((nseg, 3, self.data_cfg.CROP_SIZE, self.data_cfg.CROP_SIZE))
        self.one_id = index
        length = len((track["frames"])) // nseg
        frame_idx_iter, frame_path_iter, frame_box_iter = [],[],[]
        if not self.nl: 
            if self.nseg!=1 or len(track["frames"]) > nseg:
                idx = np.floor(np.linspace(0, len(track["frames"]), num=nseg, endpoint=False))
                for i in range(nseg):
                    frame_idx = idx[i].astype(int)
                    frame_path = os.path.join(self.data_cfg.CITYFLOW_PATH, track["frames"][frame_idx])
                    if self.motion:
                        frame = Image.open(frame_path).convert('RGB')
                        motion = frame.resize((self.data_cfg.CROP_SIZE, self.data_cfg.CROP_SIZE) , Image.BICUBIC)
                        motion = self.transform(motion)
                        nmotion[i,:,:,:] = motion
                    frame_idx_iter.append(i)
                    frame_path_iter.append(frame_path)
                    frame_box_iter.append(track["boxes"][frame_idx])
            else:
                for i in range(len(track["frames"])):
                    frame_path = os.path.join(self.data_cfg.CITYFLOW_PATH, track["frames"][i])
                    if self.motion:
                        frame = Image.open(frame_path).convert('RGB')
                        motion = frame.resize((self.data_cfg.CROP_SIZE, self.data_cfg.CROP_SIZE) , Image.BICUBIC)
                        motion = self.transform(motion)
                        nmotion[i,:,:,:] = motion
                    frame_idx_iter.append(i)
                    frame_path_iter.append(frame_path)
                    frame_box_iter.append(track["boxes"][i])

        if not self.nl:
            if nseg<80:
                for data in zip(frame_idx_iter, frame_path_iter, frame_box_iter):
                    self.read_img(data)
            else:
                with Pool(4) as p:
                    p.map(self.read_img, zip(frame_idx_iter, frame_path_iter, frame_box_iter) )
            crop = self.cropped_frames
        else:
            crop = torch.ones(1) # for fast load nlp data
        crop_id = track["track_id"]
        nl_id = track["nl_id"]
        nl = track["nl"]
        if self.motion:
            return nl, [crop, nmotion], nl_id, crop_id, label
        return nl, crop, nl_id, crop_id, label
