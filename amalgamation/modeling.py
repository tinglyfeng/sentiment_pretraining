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
from model import build_model
logger = logging.getLogger('senti_pre')


def create_model(cfg):
    model = build_model(cfg)
    return model

def set_model(gpu ,ngpus_per_node, model):
    logger.info('set model device')
    if not torch.cuda.is_available():
        logger.info('using CPU, this will be slow')
    torch.cuda.set_device(gpu)
    model = model.cuda(gpu)

def get_model(gpu, cfg ,ngpus_per_node):
    model = create_model(cfg)
    set_model(gpu, ngpus_per_node,model)
    return model