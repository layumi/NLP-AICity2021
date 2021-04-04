# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#from PIL import Image
import time
import math
from tqdm import tqdm
import os
import numpy as np
from siamese_baseline_model import SiameseBaselineModel
from vehicle_retrieval_dataset import CityFlowNLDataset
from vehicle_retrieval_dataset import CityFlowNLInferenceDataset
import yaml
from shutil import copyfile
import random
import json
import scipy.io
from DeBERTa import deberta
import pickle
from transformers import AutoTokenizer
from utils import get_model_list, load_network, save_network, make_weights_for_balanced_classes
from circle_loss import CircleLoss, convert_label_to_similarity

version =  torch.__version__
#fp16
try:
    from apex.fp16_utils import *
    from apex import amp, optimizers
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')

# make the output
if not os.path.isdir('./data/outputs'):
    os.mkdir('./data/outputs')


os.environ["TOKENIZERS_PARALLELISM"] = "false"
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--adam', action='store_true', help='use all training data' )
parser.add_argument('--names',default='ft_ResNet50', type=str, help='output model name')
parser.add_argument('--init_name',default='imagenet', type=str, help='initial with ImageNet')
parser.add_argument('--CITYFLOW_PATH',default="data/cityflow/MTMC",type=str, help='training dir path')
parser.add_argument('--JSON_PATH',default="data/train-tracks.json",type=str, help='training dir path')
parser.add_argument('--EVAL_TRACKS_JSON_PATH',default="data/test-tracks.json",type=str, help='training dir path')
parser.add_argument('--CROP_SIZE', default=256, type=int, help='batchsize')
parser.add_argument('--POSITIVE_THRESHOLD', default=0.5, type=float, help='batchsize')
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
parser.add_argument('--batchsize', default=12, type=int, help='batchsize')
parser.add_argument('--h', default=299, type=int, help='height')
parser.add_argument('--w', default=299, type=int, help='width')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--pool',default='avg', type=str, help='last pool')
parser.add_argument('--autoaug', action='store_true', help='use Color Data Augmentation' )
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--deberta', action='store_true', help='use deberta' )
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
parser.add_argument('--PCB', action='store_true', help='use PCB+ResNet50' )
parser.add_argument('--CPB', action='store_true', help='use Center+ResNet50' )
parser.add_argument('--fp16', action='store_true', help='use float16 instead of float32, which will save about 50% memory' )
parser.add_argument('--balance', action='store_true', help='balance sample' )
parser.add_argument('--angle', action='store_true', help='use angle loss' )
parser.add_argument('--arc', action='store_true', help='use arc loss' )
parser.add_argument('--circle', action='store_true', help='use Circle loss' )
parser.add_argument('--motion', action='store_true', help='use Circle loss' )
parser.add_argument('--netvlad', action='store_true', help='use Circle loss' )
parser.add_argument('--noisy', action='store_true', help='use model trained with noisy student' )
parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--resume', action='store_true', help='use arc loss' )
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
opt = parser.parse_args()

str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature_l(model,dataloaders):
    features = {}
    count = 0
    for query_id in tqdm(dataloaders):
        nl3 = dataloaders[query_id]
        ff = torch.FloatTensor(1,512).zero_().cuda()
        nl = []
        for i in range(len(nl3)):
            nl.append( '[CLS]' + nl3[i].replace('Sedan', 'sedan').replace('suv','SUV').replace('Suv','SUV').replace('Jeep','jeep').replace('  ',' ') + '[SEP]')
        
        tokens = bert_tokenizer.batch_encode_plus(nl, padding='longest',
                                                       return_tensors='pt')
        input_ids, attention_mask = tokens['input_ids'].cuda(), tokens['attention_mask'].cuda()
        lang_embeds = model.compute_lang_embed(input_ids, attention_mask = attention_mask)

        fnorm = torch.norm(lang_embeds, p=2, dim=1, keepdim=True)
        lang_embeds = lang_embeds.div(fnorm.expand_as(lang_embeds))

        ff = torch.sum(lang_embeds, dim =0)
        fnorm = torch.norm(ff, p=2, dim=0, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features[query_id] = torch.squeeze(ff).cpu().numpy()
    return features

def extract_feature_v(model, dataloaders):
    features = {}
    motion_features = {}
    count = 0
    for data, gallery_id in tqdm(dataloaders): # return one track
        frame_data = data[0]
        if model.motion:
            motion_data = data[1]
            frame_data = frame_data[0]
            n, c, h, w = frame_data.size()
        else:
            n, c, h, w = frame_data.size()
        #print(frame_data.size(), gallery_id)
        for j in range(0, n, opt.batchsize):
            img = frame_data[j:min(j+opt.batchsize,n),:,:,:]
            ff = torch.FloatTensor(1,512).zero_().cuda()
            #for i in range(2):
                #if(i==1): 
                #    img = fliplr(img)
            input_img = Variable(img.cuda())
            for scale in ms:
                if scale != 1:
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bilinear', align_corners=False)
                if model.motion:
                    outputs = model.compute_visual_embed(input_img, motion_data.cuda() )
                else:
                    outputs = model.compute_visual_embed(input_img)
                fnorm = torch.norm(outputs, p=2, dim=1, keepdim=True)
                outputs = outputs.div(fnorm.expand_as(outputs))
                ff += torch.sum(outputs, dim=0)
            if gallery_id in features:
                features[gallery_id] +=ff
            else:
                features[gallery_id] =ff
        names.append(gallery_id)
    # Normalize
    for gallery_id in features:
        ff = features[gallery_id]
        fnorm = torch.norm(ff, p=2, dim=0, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features[gallery_id] = torch.squeeze(ff).cpu().numpy()
    return features

######################################################################

names = opt.names.split(',')
models = nn.ModuleList()

for name in names:
    model_tmp, _, epoch = load_network(name, opt)
    #model_tmp = torch.nn.DataParallel(model_tmp)
    models.append(model_tmp.cuda().eval())

bert_tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# Query loader
with open("data/test-queries.json", "r") as f:
    queries = json.load(f)

# Gallery loader
galleries = CityFlowNLInferenceDataset(opt)

dataloaders = {}
dataloaders['query'] = queries
dataloaders['gallery'] = torch.utils.data.DataLoader(galleries, batch_size=1, num_workers=4, shuffle=False)

def save_pkl(name,data):
    f = open(name, "wb")
    pickle.dump(data,f)
    f.close()

def load_pkl(name):
    with open(name, "rb") as fp:   #Pickling
        mydict = pickle.load(fp) 
    return mydict

# Extract feature\
snapshot_feature_mat = './feature/submit_result_%s'%opt.names
print('Feature Output Path: %s'%snapshot_feature_mat)
if not os.path.isfile(snapshot_feature_mat+'_gallery.pkl'):
    with torch.no_grad():
        gallery_feature, query_feature  = {}, {}
        for model in models:
            q_f = extract_feature_l(model,dataloaders['query']) 
            for q_id in q_f.keys():
                if q_id in query_feature:
                    query_feature[q_id] = np.concatenate((query_feature[q_id],q_f[q_id]), 0)
                else:
                    query_feature[q_id] = q_f[q_id]
            g_f = extract_feature_v(model,dataloaders['gallery']) 
            for g_id in g_f.keys():
                if g_id in gallery_feature:
                    gallery_feature[g_id] = np.concatenate((gallery_feature[g_id],g_f[g_id]), 0)
                else:
                    gallery_feature[g_id] = g_f[g_id]

    save_pkl(snapshot_feature_mat+'_gallery.pkl', gallery_feature)
    save_pkl(snapshot_feature_mat+'_query.pkl', query_feature)
else:
    print('load feature from disk.')
    query_feature = load_pkl(snapshot_feature_mat+'_query.pkl')
    gallery_feature = load_pkl(snapshot_feature_mat+'_gallery.pkl')

gf_name = []
gf_tensor = np.ndarray( (len(gallery_feature),512*len(models))) 
count = 0
for k,v in gallery_feature.items():
    gf_tensor[count,:] = v
    gf_name.append(k)
    count +=1

#print(gf_tensor)
gf_tensor = torch.FloatTensor(gf_tensor).cuda() # 530, 512
fnorm = torch.norm(gf_tensor, p=2, dim=1, keepdim=True) # 530
#print(fnorm)
gf_tensor = gf_tensor.div(fnorm.expand_as(gf_tensor))

result = {}
for qk in query_feature.keys(): 
    qv = torch.FloatTensor(query_feature[qk]).view(-1,1).cuda()
    score = torch.mm(gf_tensor,qv)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    #print(gf_tensor)
    #print(qv)
    print(max(score))
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    result[qk] = []
    for i in index:
        result[qk].append(gf_name[i][0])
with open("results.json", "w") as f:
        json.dump(result, f, indent=4)
