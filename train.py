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
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#from PIL import Image
import time
import os
from siamese_baseline_model import SiameseBaselineModel
from vehicle_retrieval_dataset import CityFlowNLDataset
from vehicle_retrieval_dataset import CityFlowNLInferenceDataset
import yaml
from shutil import copyfile
import random
from autoaugment import ImageNetPolicy
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
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--adam', action='store_true', help='use all training data' )
parser.add_argument('--name',default='ft_ResNet50', type=str, help='output model name')
parser.add_argument('--init_name',default='imagenet', type=str, help='initial with ImageNet')
parser.add_argument('--data_dir',default='./data/pytorch2021',type=str, help='training dir path')
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--inputsize', default=299, type=int, help='batchsize')
parser.add_argument('--h', default=299, type=int, help='height')
parser.add_argument('--w', default=299, type=int, help='width')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--pool',default='avg', type=str, help='last pool')
parser.add_argument('--autoaug', action='store_true', help='use Color Data Augmentation' )
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--use_NAS', action='store_true', help='use nasnetalarge' )
parser.add_argument('--use_SE', action='store_true', help='use se_resnext101_32x4d' )
parser.add_argument('--use_DSE', action='store_true', help='use senet154' )
parser.add_argument('--use_IR', action='store_true', help='use InceptionResNetv2' )
parser.add_argument('--use_EF4', action='store_true', help='use EF4' )
parser.add_argument('--use_EF5', action='store_true', help='use EF5' )
parser.add_argument('--use_EF6', action='store_true', help='use EF6' )
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
parser.add_argument('--PCB', action='store_true', help='use PCB+ResNet50' )
parser.add_argument('--CPB', action='store_true', help='use Center+ResNet50' )
parser.add_argument('--fp16', action='store_true', help='use float16 instead of float32, which will save about 50% memory' )
parser.add_argument('--balance', action='store_true', help='balance sample' )
parser.add_argument('--angle', action='store_true', help='use angle loss' )
parser.add_argument('--arc', action='store_true', help='use arc loss' )
parser.add_argument('--circle', action='store_true', help='use Circle loss' )
parser.add_argument('--noisy', action='store_true', help='use model trained with noisy student' )
parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--resume', action='store_true', help='use arc loss' )
opt = parser.parse_args()

if opt.resume:
    model, opt, start_epoch = load_network(opt.name, opt)
else:
    start_epoch = 0

print(start_epoch)


fp16 = opt.fp16
data_dir = opt.data_dir
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

dataset = CityFlowNLDataset(train_cfg.DATA)
dataset_size = len(dataset)
dataloader = DataLoader(dataset, batch_size=opt.batchsize,
                            num_workers=4, pin_memory=True, drop_last=True)

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

def train_model(model, criterion, optimizer, scheduler, start_epoch=0, num_epochs=25):
    since = time.time()

    warm_up = 0.1 # We start from the 0.1*lrRate
    gamma = 0.0 #auto_aug
    warm_iteration = round(dataset_sizes['train']/opt.batchsize)*opt.warm_epoch # first 5 epoch
    total_iteration = round(dataset_sizes['train']/opt.batchsize)*num_epochs

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
            for data in dataloaders[phase]:
                # get the inputs
                if opt.autoaug:
                    inputs, inputs2, labels = data
                    if random.uniform(0, 1)> gamma:
                        inputs = inputs2
                    gamma = min(1.0, gamma + 1.0 / total_iteration)
                else:
                    inputs, labels = data

                now_batch_size,c,h,w = inputs.shape
                if now_batch_size<opt.batchsize: # skip the last batch
                    continue
                #print(inputs.shape)
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda().detach())
                    labels = Variable(labels.cuda().detach())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                # if we use low precision, input also need to be fp16
                #if fp16:
                #    inputs = inputs.half()
 
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                if phase == 'val':
                    with torch.no_grad():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)


                if opt.PCB:
                    part = {}
                    sm = nn.Softmax(dim=1)
                    num_part = 6
                    for i in range(num_part):
                        part[i] = outputs[i]

                    score = sm(part[0]) + sm(part[1]) +sm(part[2]) + sm(part[3]) +sm(part[4]) +sm(part[5])
                    _, preds = torch.max(score.data, 1)

                    loss = criterion(part[0], labels)
                    for i in range(num_part-1):
                        loss += criterion(part[i+1], labels)
                elif opt.CPB:
                    part = {}
                    sm = nn.Softmax(dim=1)
                    num_part = 4
                    for i in range(num_part):
                        part[i] = outputs[i]

                    score = sm(part[0]) + sm(part[1]) +sm(part[2]) + sm(part[3]) 
                    _, preds = torch.max(score.data, 1)

                    loss = criterion(part[0], labels)
                    for i in range(num_part-1):
                        loss += criterion(part[i+1], labels)
                elif opt.circle: 
                    logits, ff = outputs
                    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                    ff = ff.div(fnorm.expand_as(ff))
                    loss = criterion(logits, labels) + criterion_circle(*convert_label_to_similarity( ff, labels))/now_batch_size
                    _, preds = torch.max(logits.data, 1)                        
                else:
                    loss = criterion(outputs, labels)
                    if opt.angle or opt.arc:
                        outputs = outputs[0]
                    _, preds = torch.max(outputs.data, 1)

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
                    optimizer.step()

                #print('Iteration: loss:%.2f accuracy:%.2f'%(loss.item(), float(torch.sum(preds == labels.data))/now_batch_size ) )
                # statistics
                if int(version[0])>0 or int(version[2]) > 3: # for the new version like 0.4.0, 0.5.0 and 1.0.0
                    running_loss += loss.item() * now_batch_size
                else :  # for the old version like 0.3.0 and 0.3.1
                    running_loss += loss.data[0] * now_batch_size
                running_corrects += float(torch.sum(preds == labels.data))
                
                del(loss, outputs, inputs, preds)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0-epoch_acc)            
            # deep copy the model
            if len(opt.gpu_ids)>1:
                save_network(model.module, opt.name, epoch+1)
            else:
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
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    #ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig( os.path.join('./data/outputs',name,'train.png'))


######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#

model = SiameseBaselineModel().cuda()

if opt.init_name != 'imagenet':
    old_opt = parser.parse_args()
    init_model, old_opt, _ = load_network(opt.init_name, old_opt)
    print(old_opt)
    opt.stride = old_opt.stride
    opt.pool = old_opt.pool
    opt.use_dense = old_opt.use_dense
    if opt.use_dense:
        model = ft_net_dense(opt.nclasses, droprate=opt.droprate, stride=opt.stride, init_model=init_model, pool = opt.pool, circle = opt.circle)
    else:
        model = ft_net(opt.nclasses, droprate=opt.droprate, stride=opt.stride, init_model=init_model, pool = opt.pool, circle = opt.circle)


##########################
#Put model parameter in front of the optimizer!!!

# For resume:
if start_epoch>=60:
    opt.lr = opt.lr*0.1
if start_epoch>=75:
    opt.lr = opt.lr*0.1

model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids).cuda()

ignored_params = list(map(id, model.module.lang_fc.parameters() )) + list(map(id, model.module.resnet50.classifier.parameters() ))
base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
optimizer = torch.optim.SGD([
             {'params': base_params, 'lr': 0.1*opt.lr,},
             {'params': model.module.lang_fc.parameters(), 'lr': opt.lr,},
             {'params': model.module.resnet50.classifier.parameters(), 'lr': opt.lr,}
         ], weight_decay=5e-4, momentum=0.9, nesterov=True)
if opt.adam:
    optimizer_ft = optim.Adam(model.parameters(), opt.lr, weight_decay=5e-4)

# Decay LR by a factor of 0.1 every 40 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)
exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[60-start_epoch, 75-start_epoch], gamma=0.1)

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
# save opts
    with open('%s/opts.yaml'%dir_name,'w') as fp:
        yaml.dump(vars(opt), fp, default_flow_style=False)

# model to gpu
if fp16:
    #model = network_to_half(model)
    #optimizer_ft = FP16_Optimizer(optimizer_ft, dynamic_loss_scale=True)
    model, optimizer_ft = amp.initialize(model, optimizer_ft, opt_level = "O1")


if opt.angle:
    criterion = AngleLoss()
elif opt.arc:
    criterion = ArcLoss()
else:
    criterion = nn.CrossEntropyLoss()

print(model)
model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                    start_epoch=start_epoch, num_epochs=80)

