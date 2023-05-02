import argparse
from logging import log
import os
import random
import shutil
import time
import warnings
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import logging
from config import cfg
from model import build_model  
logger = logging.getLogger(cfg['logger']['name'])





def create_model(cfg):
    return build_model(cfg)


def set_model(cfg,ngpus_per_node, model):
    logger.info('set model device')
    if not torch.cuda.is_available():
        logger.info('using CPU, this will be slow')
    elif cfg['device']['distributed']:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if cfg['device']['gpu'] is not None:
            torch.cuda.set_device(cfg['device']['gpu'])
            model.cuda(cfg['device']['gpu'])
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            cfg['loader']['batch_size'] = int(cfg['loader']['batch_size'] / ngpus_per_node)
            cfg['loader']['workers']= int((cfg['loader']['workers']+ ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg['device']['gpu']])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif cfg['device']['gpu'] is not None:
        torch.cuda.set_device(cfg['device']['gpu'])
        model = model.cuda(cfg['device']['gpu'])
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if cfg['model']['backbone'].startswith('alexnet') or cfg['model']['backbone'].startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()


# def get_model(cfg,ngpus_per_node):
#     model = create_model(cfg)
#     set_model(cfg,ngpus_per_node,model)
#     return model