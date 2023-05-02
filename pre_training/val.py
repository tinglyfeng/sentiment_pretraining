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
from config import cfg as cfg_
import logging 
logger = logging.getLogger(cfg_['logger']['name'])

from metric import get_mll_metric_ret


def validate_cls(val_loader, model, criterion, cfg):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (data) in enumerate(val_loader):
            images = data['img']
            target = data['label']
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

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % cfg['misc']['print_freq'] == 0:
                progress.display(i)

        progress.display_summary()

    return top1.avg



def validate_ldl(val_loader, model, criterion, cfg):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    outs = []
    tags = []
    with torch.no_grad():
        end = time.time()
        for i, (data) in enumerate(val_loader):
            images = data['img']
            target = data['label']
            if cfg['device']['gpu'] is not None:
                images = images.cuda(cfg['device']['gpu'], non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(cfg['device']['gpu'], non_blocking=True)

            # compute output
            output = model(images)
            prob_output =  torch.softmax(output,dim = 1)

            # measure accuracy and record loss
            loss = criterion(torch.log(prob_output), target)
            losses.update(loss.item(), images.size(0))
            outs += prob_output.detach().cpu().tolist()
            tags += target.detach().cpu().tolist()
            
                   
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % cfg['misc']['print_freq'] == 0:
                progress.display(i)
                
        prob_output_np = np.array(outs)
        target_np = np.array(tags)
        chebyshev =  metric.chebyshev(target_np,prob_output_np)
        clark = metric.clark(target_np,prob_output_np)
        canberra = metric.canberra(target_np,prob_output_np)
        kl = metric.kl(target_np,prob_output_np)
        cosine = metric.cosine(target_np,prob_output_np)
        intersection = metric.intersection(target_np,prob_output_np)
        logger.info('chebyshev: {}, clark: {}, canberra: {}, kl: {}, cosine: {}, intersection: {}'.format(chebyshev, clark, canberra, kl, cosine, intersection))
        progress.display_summary()
    return 




def validate_mll(val_loader, model, criterion, cfg):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    outs = []
    tags = []
    with torch.no_grad():
        end = time.time()
        for i, (data) in enumerate(val_loader):
            images = data['img']
            target = data['label']
            if cfg['device']['gpu'] is not None:
                images = images.cuda(cfg['device']['gpu'], non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(cfg['device']['gpu'], non_blocking=True)

            # compute output
            output = model(images)
            out_sigmoid = torch.sigmoid(output)

            # measure accuracy and record loss
            loss = criterion(output, target)
            losses.update(loss.item(), images.size(0))
            
            outs += out_sigmoid.detach().cpu().tolist()
            tags += target.detach().cpu().tolist()
            
                   
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % cfg['misc']['print_freq'] == 0:
                progress.display(i)
                
        out_sigmoid_np = np.array(outs)
        target_np = np.array(tags)
        ret = get_mll_metric_ret(out_sigmoid_np, target_np)
       
        logger.info('ranking: {}, hamming: {}, f1_micro: {}, f1_macro: {}'.format(ret['ranking'],ret['hamming'], ret['f1_micro'], ret['f1_macro']))
        progress.display_summary()
    return 



def val_pretrain(test_loader, model, criterions, optimizer, epoch, cfg):
    
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
                
    def get_metric_meter():
        metric_dict = {}
        tasks_info = cfg['tasks_info']
        if 'color' in tasks_info:
            loss_meter = AverageMeter('color_MSE', ':6.6f', Summary.AVERAGE)
            # progress = ProgressMeter(
            #     len(test_loader),loss_meter, prefix='Test: '
            # )
            metric_dict['color'] = loss_meter
            
        if 'sr' in tasks_info:
            loss_meter = AverageMeter('sr_MSE', ':6.6f', Summary.AVERAGE)
            # progress = ProgressMeter(
            #     len(test_loader),loss_meter, prefix='Test: '
            # )
            metric_dict['sr'] = loss_meter
        
        if 'jigsaw' in tasks_info:
            acc_meter = AverageMeter('jigsaw_ACC', ':6.2f', Summary.AVERAGE)
            # progress = ProgressMeter(
            #     len(test_loader),acc_meter, prefix='Test: '
            # )
            metric_dict['jigsaw'] = acc_meter
            
        if 'scene' in tasks_info:
            acc_meter = AverageMeter('scene_ACC', ':6.2f', Summary.AVERAGE)
            # progress = ProgressMeter(
            #     len(test_loader),acc_meter, prefix='Test: '
            # )
            metric_dict['scene'] = acc_meter
        
        if 'anp' in tasks_info:
            acc_meter = AverageMeter('anp_ACC', ':6.2f', Summary.AVERAGE)
            # progress = ProgressMeter(
            #     len(test_loader),acc_meter, prefix='Test: '
            # )
            metric_dict['anp'] = acc_meter
        
        if 'caption' in tasks_info:
            loss_meter = AverageMeter('caption_LOSS', ':6.2f', Summary.AVERAGE)
            # progress = ProgressMeter(
            #     len(test_loader),loss_meter, prefix='Test: '
            # )
            metric_dict['caption'] = loss_meter
            
        return metric_dict
        

    def calc_performance(res, data, metric_dict):

        if 'color' in res:
            loss_color = criterions['color'](res['color'], data['color']['ab'])
            metric_dict['color'].update(loss_color.item(), cfg['loader']['batch_size'])
            
        if 'sr' in res:
            loss_sr = criterions['sr'](res['sr'],data['sr']['tar'])
            metric_dict['sr'].update(loss_sr.item(), cfg['loader']['batch_size'])
            
        if 'jigsaw' in res:
            logits = res['jigsaw']
            preds = logits.argmax(1)
            #print('debug: ', preds.shape, cfg['loader']['batch_size'])
            acc = (data['jigsaw']['order'] == preds).sum() /  cfg['loader']['batch_size']
            metric_dict['jigsaw'].update(acc, cfg['loader']['batch_size'])
            
        if 'scene' in res:
            logits = res['scene']
            preds = logits.argmax(1)
            acc = (data['scene']['label'] == preds).sum() /  cfg['loader']['batch_size']
            metric_dict['scene'].update(acc, cfg['loader']['batch_size'])
        if 'anp' in res:
            logits = res['anp']
            preds = logits.argmax(1)
            acc = (data['anp']['label'] == preds).sum() /  cfg['loader']['batch_size']
            metric_dict['anp'].update(acc.item(), cfg['loader']['batch_size'])
        if 'caption' in res:
            logits = res['caption']['logits']
            targets = res['caption']['targets']
            loss_caption = criterions['caption'](logits.data,
                        targets.data)
            metric_dict['caption'].update(loss_caption.item(), 
                                         cfg['loader']['batch_size'])
                
                    
                    
        
        
    metric_dict = get_metric_meter()
    progress = ProgressMeter(
        len(test_loader), 
        [*list(metric_dict.values())],
        prefix="Epoch: [{}]".format(epoch)
    )
    # switch to train mode
    model.eval()

    with torch.no_grad():
        for i, (data) in enumerate(test_loader):
            # measure data loading time
            set_device_and_dimension(data)

            # compute output
            optimizer.zero_grad()
            
            res = model(data)

            #loss = criterions(output, target)   
            calc_performance(res, data, metric_dict)


            # measure elapsed time

            if i % cfg['misc']['print_freq'] == 0 and (
                not cfg['device']['mulp_dist_enable'] or cfg['device']['gpu'] == 0):
                progress.display(i)
