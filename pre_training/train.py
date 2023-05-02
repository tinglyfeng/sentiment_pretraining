from importlib_metadata import Sectioned
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
from zmq import device
from utils import AverageMeter,ProgressMeter,accuracy,Summary
import time
import metric
import numpy as np
from config import cfg as cfg_
import logging 
logger = logging.getLogger(cfg_['logger']['name'])

def train_cls(train_loader, model, criterion, optimizer, epoch, cfg):
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

        if cfg['device']['gpu'] is not None:
            images = images.cuda(cfg['device']['gpu'], non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(cfg['device']['gpu'], non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
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

        if i % cfg['misc']['print_freq'] == 0:
            progress.display(i)



def train_ldl(train_loader, model, criterion, optimizer, epoch, cfg):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (data) in enumerate(train_loader):
        # measure data loading time
        images = data['img']
        target = data['label']
        data_time.update(time.time() - end)

        if cfg['device']['gpu'] is not None:
            images = images.cuda(cfg['device']['gpu'], non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(cfg['device']['gpu'], non_blocking=True)

        # compute output
        output = model(images)
        ## convert the 
        prob_output =  torch.softmax(output,dim = 1)
        if isinstance(criterion,torch.nn.modules.loss.KLDivLoss):
            loss = criterion(torch.log(prob_output), target)   ### beaware the input format of KLDivLoss, the predictions should be log-probabilities and target should be probabilities by default
        else:
            raise ValueError 
        losses.update(loss.item(), images.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg['misc']['print_freq'] == 0:
            progress.display(i)
            

def train_mll(train_loader, model, criterion, optimizer, epoch, cfg):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (data) in enumerate(train_loader):
        # measure data loading time
        images = data['img']
        target = data['label']
        data_time.update(time.time() - end)

        if cfg['device']['gpu'] is not None:
            images = images.cuda(cfg['device']['gpu'], non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(cfg['device']['gpu'], non_blocking=True)

        # compute output
        output = model(images)

        loss = criterion(output, target)   

        losses.update(loss.item(), images.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg['misc']['print_freq'] == 0:
            progress.display(i)



def train_pretrain(train_loader, model, criterions, optimizer, epoch, cfg):
    
    def set_device_and_dimension(input):
        if cfg['device']['gpu'] is not None:
            device_id = cfg['device']['gpu']
            # if 'gene' in input and 'img' in input['gene']['img']:
            if 'gene' in input and 'img' in input['gene']:
                input['gene']['img'] = input['gene']['img'].to(device_id)
            
            if 'color' in input:
                input['color']['gray'] = input['color']['gray'].to(device_id).permute((0,3,1,2))
                input['color']['ab'] = input['color']['ab'].to(device_id).permute(0,3,1,2)
            
            if 'sr' in input:
                input['sr']['img'] = input['sr']['img'].to(device_id)
                input['sr']['tar'] = input['sr']['tar'].to(device_id).permute(0,3,1,2)
            
            if 'jigsaw' in input:
                input['jigsaw']['img'] = input['jigsaw']['img'].to(device_id)
                input['jigsaw']['order'] = input['jigsaw']['order'].to(device_id)
            
            if 'scene' in input:
                # input['scene']['label'] = input['scene']['label'].to(device)
                input['scene']['label'] = input['scene']['label'].to(device_id)
            
            if 'anp' in input:
                input['anp']['img'] = input['anp']['img'].to(device_id)
                input['anp']['label'] = input['anp']['label'].to(device_id)
            
            if 'caption' in input:
                input['caption']['image'] = input['caption']['image'].to(device_id)
                input['caption']['tokens'] = input['caption']['tokens'].to(device_id)
                
                
    def get_loss_meter():
        tasks_info = cfg['tasks_info']
        task_loss_meter_dict = {}
        if 'color' in tasks_info:
            task_loss_meter_dict['color'] = AverageMeter('loss_color', ":4f")
        if 'sr' in tasks_info:
            task_loss_meter_dict['sr'] = AverageMeter('loss_sr', ":4f")
        if 'jigsaw' in tasks_info:
            task_loss_meter_dict['jigsaw'] = AverageMeter('loss_jigsaw', ":4f")
        if 'scene' in tasks_info:
            task_loss_meter_dict['scene'] = AverageMeter('loss_scene', ":4f")
        if 'anp' in tasks_info:
            task_loss_meter_dict['anp'] = AverageMeter('loss_anp', ":4f")
        if 'caption' in tasks_info:
            task_loss_meter_dict['caption'] = AverageMeter('loss_caption', ":4f")
        return task_loss_meter_dict
            

    def calc_loss(res, data, meter_dict):
        loss = 0
        if 'color' in res:
            loss_color = criterions['color'](res['color'], data['color']['ab'])
            loss += loss_color
            meter_dict['color'].update(loss_color.item(), cfg['loader']['batch_size'])
        if 'sr' in res:
            loss_sr = criterions['sr'](res['sr'],data['sr']['tar'])
            loss += loss_sr
            meter_dict['sr'].update(loss_sr.item(), cfg['loader']['batch_size'])
        if 'jigsaw' in res:
            loss_jigsaw = criterions['jigsaw'](res['jigsaw'], data['jigsaw']['order'])
            loss += loss_jigsaw
            meter_dict['jigsaw'].update(loss_jigsaw.item(), cfg['loader']['batch_size'])
        if 'scene' in res:
            loss_scene = criterions['scene'](res['scene'], data['scene']['label'])
            loss += loss_scene
            meter_dict['scene'].update(loss_scene.item(), cfg['loader']['batch_size'])
        if 'anp' in res:
            loss_anp = criterions['anp'](res['anp'], data['anp']['label'])
            loss += loss_anp
            meter_dict['anp'].update(loss_anp.item(), cfg['loader']['batch_size'])
        if 'caption' in res:
            logits = res['caption']['logits']
            targets = res['caption']['targets']
            loss_caption = criterions['caption'](logits.data,
                        targets.data)
            alpha_c = cfg['tasks_info']['caption']['decoder']['alpha_c']
            if  alpha_c > 0:
                decoder_length = res['caption']['decode_lengths']
                alphas = res['caption']['alphas']
                device_type = res['caption']['logits'].data.device
                total_energy = torch.from_numpy(
                    np.array(decoder_length)
                    ) / alphas.shape[-1]
                total_energy.unsqueeze_(-1)  # B x 1
                total_energy = total_energy.to(device_type)
                loss_caption_d_atn = alpha_c * (
                    (total_energy -  alphas.sum(dim=1)) ** 2
                    ).mean()
                loss_caption += loss_caption_d_atn
            loss += loss_caption
            meter_dict['caption'].update(loss_caption.item(), 
                                         cfg['loader']['batch_size'])
                       
        return loss
        
        
    batch_time = AverageMeter('Time', ':6.3f')      
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss_Sum', ':.4f')
    task_loss_meter_dict = get_loss_meter()
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, *list(task_loss_meter_dict.values())],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (data) in enumerate(train_loader):
        set_device_and_dimension(data)
        # compute output
        optimizer.zero_grad()
        
        res = model(data)

        #loss = criterions(output, target)   
        loss = calc_loss(res, data, task_loss_meter_dict)

        losses.update(loss.item(),cfg['loader']['batch_size'])
        # compute gradient and do SGD step
        # s_bp = time.time()
        loss.backward()
        # print('loss bp time : ', time.time() - s_bp)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg['misc']['print_freq'] == 0 and (
            not cfg['device']['mulp_dist_enable'] or cfg['device']['gpu'] == 0):
            progress.display(i)
            