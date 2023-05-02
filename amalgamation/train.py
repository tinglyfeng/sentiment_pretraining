import torch
from torch._C import _to_dlpack
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
from utils import AverageMeter,ProgressMeter,accuracy,Summary
import time
import metric
import numpy as np
import logging 
import torch.nn.functional as F

logger = logging.getLogger('senti_pre')

def to_log_prob_logits(output,cfg):
    
    output['stu']['logits'] = torch.log(torch.softmax(output['stu']['logits'], dim=1))

    teacher_infos = cfg['model']['teachers']
    if 'low' in teacher_infos:
        output['low']['logits'] = torch.log(torch.softmax(output['low']['logits'], dim=1))

    if 'mid' in teacher_infos:
        output['mid']['logits'] = torch.log(torch.softmax(output['mid']['logits'], dim=1))

    if 'high' in teacher_infos:
        output['high']['logits'] = torch.log(torch.softmax(output['high']['logits'], dim=1))

    
def kd_loss(output, target, criterion, cfg):
    
    def single_level_loss(level):
        level_info = teacher_infos[level]
        level_output = output[level]
        level_logits = level_output['logits']
        stu_output = output['stu']
        stu_logits = stu_output['logits']
        level_loss = 0
        
        if level_info['target_loss']:
            level_target_loss = criterion(level_logits, target)
            level_loss += level_target_loss
            
        if level_info['logits_reg']:
            level_logits_loss = F.mse_loss(level_logits, stu_logits)
            level_loss += level_logits_loss
            
        if level_info['feature_reg']:
            feature_stages_info = level_info['feature_reg']
            level_feature_loss = 0
            for stage in feature_stages_info:
                t_fm = level_output[stage]
                s_fm = stu_output[stage]
                stage_feature_loss = F.mse_loss(t_fm, s_fm)
                level_feature_loss += stage_feature_loss
            level_loss += level_feature_loss
            
        if 'loss_factor' in level_info:
            level_loss *= level_info['loss_factor']
            
        return level_loss
                
    teacher_infos = cfg['model']['teachers']
    total_loss = 0
    main_target_loss = criterion(output['stu']['logits'], target)
    total_loss += main_target_loss
    
    if 'low' in teacher_infos:
        total_loss += single_level_loss('low')
        
    if 'mid' in teacher_infos:
        total_loss += single_level_loss('mid')
    
    if 'high' in teacher_infos:
        total_loss += single_level_loss('high')
    return total_loss

def train_cls(train_loader, model, criterion, optimizer, epoch, gpu, model_setting):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (data) in enumerate(train_loader):
        # measure data loading time
        images = data['img']
        target = data['label']
        data_time.update(time.time() - end)

        images = images.cuda(gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = kd_loss(output, target, criterion, model_setting)

        # measure accuracy and record loss
        output = output['stu']['logits']
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            progress.display(i)      