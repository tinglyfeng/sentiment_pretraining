## modified from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
import os
from torch._C import Value
# os.chdir(os.path.dirname(__file__))
import argparse
from logging import Logger
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
from train import *
from solver import *
from utils import *
from val import *
from loader import get_dataloader
from modeling import get_model
from pprint import pformat
from PIL import ImageFile
from utils import logging_lr
ImageFile.LOAD_TRUNCATED_IMAGES = True


parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--backbone_id", type=int, default=1)
args = parser.parse_args()
print(args)

logger_path = './exp/log/fi.log'
logger = get_logger('senti_pre', logger_path)

best_acc1 = 0

def main_worker(ngpus_per_node):
    backbone_zoo = ["resnet18", "resnet50", "resnet101" , "vgg16_bn" ,"vgg19_bn"]
    gpu = args.gpu_id
    backbone = backbone_zoo[args.backbone_id]
    global best_acc1
    logger.info("Use GPU: {} for training".format(gpu))
    model_setting = {
        "model":{
            "num_class": 8,
            "teachers":{
                "low":{ 
                    "backbone": backbone,
                    "pretrained_path": "./pretrained_models/" + backbone + "_low.pth.tar",
                    "target_loss": True,
                    "logits_reg": False,
                    "feature_reg": ['f2'],
                },
                "mid":{
                    "backbone": backbone,
                    "pretrained_path": "./pretrained_models/" + backbone + "_mid.pth.tar",
                    "target_loss": True,
                    "logits_reg": False,
                    "feature_reg": ['f3'],
                },
                "high":{
                    "backbone": backbone,
                    "pretrained_path": "./pretrained_models/" + backbone + "_high.pth.tar",
                    "target_loss": True,
                    "logits_reg": True,
                    "feature_reg": ['f4'],
                },
            },
            "student":{
                "backbone": backbone,
                "pretrained_path": "./pretrained_models/" + backbone + "_high.pth.tar",
                },
            }
    }
    logger.info(model_setting)

    dataset_path = "/home/ubuntu16/ljx/datasets/FI"
    batch_size = 32
    lr = 0.001

    model = get_model(gpu, model_setting, ngpus_per_node)
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(0)
    # optionally resume from a checkpoint
    optimizer = get_optimizer(model, model_setting, lr)   
    lr_sheduler = torch.optim.lr_scheduler.StepLR(optimizer,  step_size=10, gamma=0.1, last_epoch=-1)
    cudnn.benchmark = True
    train_loader, val_loader, _ = get_dataloader(dataset_path, batch_size)

    for epoch in range(30):
        logging_lr(optimizer)
        # train for one epoch
        train_cls(train_loader, model, criterion, optimizer, epoch, gpu, model_setting)
        # evaluate on validation set
        acc1 = validate_cls(val_loader, model, criterion, gpu)
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        lr_sheduler.step()


if __name__ == '__main__':

    logger.info("\n\n\n\n" + "*" * 50 + 'new_expr' + "*" * 50)
    ngpus_per_node = torch.cuda.device_count()
    backbone_zoo = ["resnet18", "resnet50", "resnet101" , "vgg16" ,"vgg19"]

    main_worker(ngpus_per_node)
