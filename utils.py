#!/usr/bin/env python
# COPYRIGHT 2020. Fred Fung. Boston University.
"""
Logging utilities.
"""
import io
import logging
import os

import colorlog

import torch
import yaml
import torch.nn as nn
import parser
import swa_utils
from model import ft_net, ft_net_dense, ft_net_EF4, ft_net_EF5, ft_net_EF6, ft_net_IR, ft_net_NAS, ft_net_SE, ft_net_DSE, PCB, CPB, ft_net_angle, ft_net_arc
from siamese_baseline_model import SiameseBaselineModel

def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1 # count the image number in every class
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight

# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        print('no dir: %s'%dirname)
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pth" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name

######################################################################
# Save model
#---------------------------
def save_network(network, dirname, epoch_label):
    if isinstance(epoch_label, int):
        save_filename = 'net_%03d.pth'% epoch_label
    else:
        save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join('./data/outputs',dirname,save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.cuda()


######################################################################
#  Load model for resume
#---------------------------
def load_network(name, opt):
    # Load config
    dirname = os.path.join('./data/outputs',name)
    last_model_name = os.path.basename(get_model_list(dirname, 'net'))
    epoch = last_model_name.split('_')[1]
    epoch = epoch.split('.')[0]
    if not epoch=='last' and not epoch=='waverage':
       epoch = int(epoch)
    config_path = os.path.join(dirname,'opts.yaml')
    with open(config_path, 'r') as stream:
        config = yaml.load(stream)

    opt.name = config['name']
    opt.CROP_SIZE = config['CROP_SIZE']
    opt.CITYFLOW_PATH = config['CITYFLOW_PATH']
    opt.JSON_PATH = config['JSON_PATH']
    opt.droprate = config['droprate']
    opt.color_jitter = config['color_jitter']
    opt.batchsize = config['batchsize']
    opt.stride = config['stride']
    opt.pool = config['pool']
    opt.h = config['h']
    opt.w = config['w']
    opt.gpu_ids = config['gpu_ids']
    opt.erasing_p = config['erasing_p']
    opt.deberta = config['deberta']
    opt.lr = config['lr']
    opt.erasing_p = config['erasing_p']
    opt.PCB = config['PCB']
    opt.CPB = config['CPB']
    opt.fp16 = config['fp16']
    opt.balance = config['balance']
    opt.angle = config['angle']
    opt.arc = config['arc']

    # load model
    if isinstance(epoch, int):
        save_filename = 'net_%03d.pth'% epoch
    else:
        save_filename = 'net_%s.pth'% epoch

    save_path = os.path.join('./data/outputs',name,save_filename)
    print('Load the model from %s'%save_path)
    network = SiameseBaselineModel(opt).cuda()
 
    try:
        network.load_state_dict(torch.load(save_path))
    except:
        network = torch.nn.DataParallel(network)
        if epoch=='waverage':
            network = swa_utils.AveragedModel(network)
        network.load_state_dict(torch.load(save_path))
        network = network.module
        if epoch=='waverage':
            network = network.module
    return network, opt, epoch

class TqdmToLogger(io.StringIO):
    logger = None
    level = None
    buf = ''

    def __init__(self):
        super(TqdmToLogger, self).__init__()
        self.logger = get_logger('tqdm')

    def write(self, buf):
        self.buf = buf.strip('\r\n\t ')

    def flush(self):
        self.logger.info(self.buf)


def get_logger(logger_name='default', debug=False, save_to_dir=None):
    if debug:
        log_format = (
            '%(asctime)s - '
            '%(levelname)s : '
            '%(name)s - '
            '%(pathname)s[%(lineno)d]:'
            '%(funcName)s - '
            '%(message)s'
        )
    else:
        log_format = (
            '%(asctime)s - '
            '%(levelname)s : '
            '%(name)s - '
            '%(message)s'
        )
    bold_seq = '\033[1m'
    colorlog_format = f'{bold_seq} %(log_color)s {log_format}'
    colorlog.basicConfig(format=colorlog_format, datefmt='%y-%m-%d %H:%M:%S')
    logger = logging.getLogger(logger_name)

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    if save_to_dir is not None:
        fh = logging.FileHandler(os.path.join(save_to_dir, 'log', 'debug.log'))
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(log_format)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        fh = logging.FileHandler(
            os.path.join(save_to_dir, 'log', 'warning.log'))
        fh.setLevel(logging.WARNING)
        formatter = logging.Formatter(log_format)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        fh = logging.FileHandler(os.path.join(save_to_dir, 'log', 'error.log'))
        fh.setLevel(logging.ERROR)
        formatter = logging.Formatter(log_format)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
