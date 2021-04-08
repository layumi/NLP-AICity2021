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
import math
from sam import SAM
matplotlib.use('agg')
import matplotlib.pyplot as plt
#from PIL import Image
import time
from tqdm import tqdm
import os
from siamese_baseline_model import SiameseBaselineModel
from vehicle_retrieval_dataset import CityFlowNLDataset
from vehicle_retrieval_dataset import CityFlowNLInferenceDataset
import yaml
from shutil import copyfile
import random
import numpy as np
#from DeBERTa import deberta
from transformers import AutoTokenizer
import utils_T2
from utils import get_model_list, load_network, save_network, ContrastiveLoss, make_weights_for_balanced_classes
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
parser.add_argument('--name',default='ft_ResNet50', type=str, help='output model name')
parser.add_argument('--init_name',default='imagenet', type=str, help='initial with ImageNet')
parser.add_argument('--CITYFLOW_PATH',default="data/cityflow/MTMC",type=str, help='training dir path')
parser.add_argument('--JSON_PATH',default="data/train-tracks-clear.json",type=str, help='training dir path')
parser.add_argument('--EVAL_TRACKS_JSON_PATH',default="data/test-tracks.json",type=str, help='training dir path')
parser.add_argument('--CROP_SIZE', default=256, type=int, help='batchsize')
parser.add_argument('--POSITIVE_THRESHOLD', default=0.5, type=float, help='batchsize')
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
parser.add_argument('--batchsize', default=12, type=int, help='batchsize')
parser.add_argument('--h', default=299, type=int, help='height')
parser.add_argument('--w', default=299, type=int, help='width')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--nseg', default=1, type=int, help='nseg')
parser.add_argument('--pool',default='avg', type=str, help='last pool')
parser.add_argument('--autoaug', action='store_true', help='use Color Data Augmentation' )
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--deberta', action='store_true', help='use deberta' )
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
parser.add_argument('--PCB', action='store_true', help='use PCB+ResNet50' )
parser.add_argument('--CPB', action='store_true', help='use Center+ResNet50' )
parser.add_argument('--fp16', action='store_true', help='use float16 instead of float32, which will save about 50% memory' )
parser.add_argument("--sam", action='store_true', help="enable sam.")
parser.add_argument('--balance', action='store_true', help='balance sample' )
parser.add_argument('--angle', action='store_true', help='use angle loss' )
parser.add_argument('--arc', action='store_true', help='use arc loss' )
parser.add_argument('--track2', action='store_true', help='use arc loss' )
parser.add_argument('--circle', action='store_true', help='use Circle loss' )
parser.add_argument('--motion', action='store_true', help='use motion' )
parser.add_argument('--ddloss', action='store_true', help='use ddloss' )
parser.add_argument('--xhloss', action='store_true', help='use conloss' )
parser.add_argument('--netvlad', action='store_true', help='use netvlad' )
parser.add_argument('--all3', action='store_true', help='use netvlad' )
parser.add_argument('--fixed', action='store_true', help='use netvlad' )
parser.add_argument('--noisy', action='store_true', help='use model trained with noisy student' )
parser.add_argument('--warm_epoch', default=10, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--num_epoch', default=80, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--resume', action='store_true', help='use arc loss' )
parser.add_argument('--semi', action='store_true', help='transductive learning' )
opt = parser.parse_args()

if opt.all3: 
    opt.JSON_PATH = "data/train-tracks-clear-all3.json"

if opt.resume:
    model, opt, start_epoch = load_network(opt.name, opt)
    model.bert_model.pooler = torch.nn.Sequential()
else:
    start_epoch = 0

print(start_epoch)


fp16 = opt.fp16
name = opt.name

if not opt.resume:
    str_ids = opt.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >=0:
            gpu_ids.append(gid)
    opt.gpu_ids = gpu_ids

# set gpu ids
if len(opt.gpu_ids)>0:
    cudnn.enabled = True
    cudnn.benchmark = True
######################################################################
# Load Data
# ---------
#

dataset = CityFlowNLDataset(opt)
dataset_size = len(dataset)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchsize,
                            shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

use_gpu = torch.cuda.is_available()

#since = time.time()
#inputs, classes = next(iter(dataloaders['train']))
#print(time.time()-since)

######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []

def l2_norm(v):
    fnorm = torch.norm(v, p=2, dim=1, keepdim=True) + 1e-6
    v = v.div(fnorm.expand_as(v))
    return v

xhloss = ContrastiveLoss()
def compute_loss(model, input_ids, attention_mask, crop, motion, nl_id, crop_id, label, warm):
    crop = crop.view(opt.nseg*opt.batchsize,3,opt.CROP_SIZE, opt.CROP_SIZE)
    if opt.motion:
        visual_embeds, lang_embeds, predict_class_v, predict_class_l = model.forward(input_ids, attention_mask, crop, motion.cuda())
    else:
        visual_embeds, lang_embeds, predict_class_v, predict_class_l = model.forward(input_ids, attention_mask, crop)
    #print(similarity.shape, predict_class_v.shape, predict_class_l.shape)
    #print(label.shape, nl_id.shape)
    #label = label.float()

    visual_embeds = l2_norm(visual_embeds)    
    lang_embeds = l2_norm(lang_embeds)    
    if opt.xhloss:
        loss_xh = xhloss(torch.mm(visual_embeds, torch.t(lang_embeds))) *opt.batchsize
    if opt.ddloss:
        visual_embeds = visual_embeds.t()
        lang_embeds =lang_embeds.t()
        # dense triplet loss
        sim1 = torch.mm(visual_embeds*torch.exp(model.module.logit_scale1), torch.t(lang_embeds)) 
    else: 
        mask  = torch.ones((opt.batchsize, opt.batchsize))
        for i in range(opt.batchsize):
            for j in range(opt.batchsize):
                if i!=j and crop_id[i] == crop_id[j]:
                    mask[i,j] = 0.
                    mask[j,i] = 0.
                    #print(mask)
        sim1 = torch.mm(visual_embeds*torch.exp(model.module.logit_scale1), torch.t(lang_embeds)) 
        sim1 = sim1 * mask.cuda()
    sim2 = sim1.t()
    sim_label = torch.arange(sim1.size(0)).cuda().detach()
    sim_label[np.argwhere(nl_id==-1)] = -1 
    loss_con = F.cross_entropy(sim1, sim_label, ignore_index = -1) + F.cross_entropy(sim2, sim_label, ignore_index = -1)
    if opt.xhloss:
        loss_con = 0
    else: 
        loss_xh  = 0
    loss_cv =  F.cross_entropy(predict_class_v, crop_id.cuda(), ignore_index = -1)
    loss_nl =  F.cross_entropy(predict_class_l, nl_id.cuda(), ignore_index = -1)

    #if opt.motion:  
    #    loss_mt =  F.cross_entropy(predict_class_motion, crop_id.cuda(), ignore_index = -1)
    #else:
    loss_mt = 0.0
    loss_total = loss_con/2 + loss_cv + loss_nl + loss_mt + loss_xh
    print('\r\rtotal:%.4f  loss_con:%.4f loss_cv:%.4f loss_nl:%.4f loss_mt:%.4f loss_xh:%.4f  warmup:%.4f'%(loss_total, loss_con, loss_cv, loss_nl, loss_mt, loss_xh, warm), end="" )
    return loss_total


def train_model(model, criterion, optimizer, scheduler, start_epoch=0, num_epochs=25):
    bert_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    since = time.time()

    warm_up = 0.1 # We start from the 0.1*lrRate
    gamma = 0.0 #auto_aug
    warm_iteration = round(dataset_size/opt.batchsize)*opt.warm_epoch # first 5 epoch
    print(warm_iteration)
    total_iteration = round(dataset_size/opt.batchsize)*num_epochs

    best_model_wts = model.state_dict()
    best_loss = 9999
    best_epoch = 0
    if opt.circle:
        criterion_circle = CircleLoss(m=0.25, gamma=32)
        
    for epoch in range(num_epochs-start_epoch):
        epoch = epoch + start_epoch
        print('gamma: %.4f'%gamma)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            # Iterate over data.
            with tqdm(dataloader, ascii=True) as tq:
                for data in tq:
                # zero the parameter gradients
                    if opt.motion:
                        nl, crop, motion, nl_id, crop_id, label = data
                    else:
                        nl, crop, nl_id, crop_id, label = data
                        motion = None
                    tokens = bert_tokenizer.batch_encode_plus(nl, padding='longest',
                                                       return_tensors='pt')

                    optimizer.zero_grad()
                    loss = compute_loss(model, tokens['input_ids'].cuda(), tokens['attention_mask'].cuda(), crop.cuda(), motion, nl_id, crop_id, label, warm_up)
                # backward + optimize only if in training phase
                    if epoch<opt.warm_epoch and phase == 'train': 
                        warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                        loss *= warm_up
                # backward + optimize only if in training phase
                    if phase == 'train':
                        if fp16: # we use optimier to backward loss
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()

                        if opt.sam:
                            optimizer.first_step(zero_grad=True)
                            loss.backward()
                            optimizer.second_step(zero_grad=True)
                        else:
                            optimizer.step()

                # statistics
                    if int(version[0])>0 or int(version[2]) > 3: # for the new version like 0.4.0, 0.5.0 and 1.0.0
                        running_loss += loss.item() * opt.batchsize
                    else :  # for the old version like 0.3.0 and 0.3.1
                        running_loss += loss.data[0] * now_batch_size
                
                    del(loss, tokens, data, nl, crop, nl_id, crop_id, label)
            epoch_loss = running_loss / dataset_size
            
            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))
            
            y_loss[phase].append(epoch_loss)
            # deep copy the model
            #if len(opt.gpu_ids)>1:
            #    save_network(model.module, opt.name, epoch+1)
            #else:
            if epoch %10 ==0:
                save_network(model, opt.name, epoch+1)
            draw_curve(epoch)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch
            last_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best epoch: {:d} Best Train Loss: {:4f}'.format(best_epoch, best_loss))

    # load best model weights
    model.load_state_dict(last_model_wts)
    save_network(model, opt.name, 'last')
    return model


######################################################################
# Draw Curve
#---------------------------
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")
def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    #ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    #ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    #ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        #ax1.legend()
    fig.savefig( os.path.join('./data/outputs',name,'train.png'))


######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#
if not opt.resume:
    init_model = None

    if opt.track2: 
        old_opt = parser.parse_args()
        init_model, old_opt, _ = utils_T2.load_network('ft_2021SE_imbalance_s1_384_p0.5_lr1_mt_d0.2_b36_wa_sam_gem', old_opt )

    model = SiameseBaselineModel(opt, init_model)

model = model.cuda()
##########################
#Put model parameter in front of the optimizer!!!

# For resume:
if start_epoch>=round(0.8*opt.num_epoch-start_epoch):
    opt.lr = opt.lr*0.1
if start_epoch>=round(0.95*opt.num_epoch-start_epoch):
    opt.lr = opt.lr*0.1

model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids).cuda()

ignored_params = list(map(id, model.module.lang_fc.parameters() )) + list(map(id, model.module.visual_fc.parameters() )) +\
         list(map(id, model.module.resnet50.classifier.parameters() ))+ list(map(id, model.module.bert_model.parameters() ))

base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

if opt.sam:
    optim_method = SAM
else:
    optim_method = optim.SGD


optimizer = optim_method([
             {'params': base_params, 'lr': 0.1*opt.lr,},
             {'params': model.module.bert_model.parameters(), 'lr': 0.1*opt.lr,},
             {'params': model.module.lang_fc.parameters(), 'lr': opt.lr,},
             {'params': model.module.visual_fc.parameters(), 'lr': opt.lr,},
             {'params': model.module.resnet50.classifier.parameters(), 'lr': opt.lr,}
         ], weight_decay=5e-4, momentum=0.9, nesterov=True)

if opt.fixed:
    optimizer = optim_method([
             {'params': model.module.resnet50.model.parameters(), 'lr': 0.1*opt.lr,},
             {'params': model.module.lang_fc.parameters(), 'lr': opt.lr,},
             {'params': model.module.visual_fc.parameters(), 'lr': opt.lr,},
             {'params': model.module.resnet50.classifier.parameters(), 'lr': opt.lr,}
         ], weight_decay=5e-4, momentum=0.9, nesterov=True)

if opt.adam:
    optimizer_ft = optim.Adam(model.parameters(), opt.lr, weight_decay=5e-4)

# Decay LR by a factor of 0.1 every 40 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)
exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(0.8*opt.num_epoch-start_epoch), round(0.95*opt.num_epoch)-start_epoch], gamma=0.1)

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 1-2 hours on GPU. 
#
dir_name = os.path.join('./data/outputs',name)

if not opt.resume:
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
#record every run
    copyfile('./train.py', dir_name+'/train.py')
    copyfile('./model.py', dir_name+'/model.py')
    copyfile('./siamese_baseline_model.py', dir_name+'/siamese_baseline_model.py')
# save opts
    with open('%s/opts.yaml'%dir_name,'w') as fp:
        yaml.dump(vars(opt), fp, default_flow_style=False)

# model to gpu
if fp16:
    #model = network_to_half(model)
    #optimizer_ft = FP16_Optimizer(optimizer_ft, dynamic_loss_scale=True)
    model, optimizer = amp.initialize(model, optimizer, opt_level = "O1")


if opt.angle:
    criterion = AngleLoss()
elif opt.arc:
    criterion = ArcLoss()
else:
    criterion = nn.CrossEntropyLoss()

print(model)
model = train_model(model, criterion, optimizer, exp_lr_scheduler,
                    start_epoch=start_epoch, num_epochs=opt.num_epoch)

