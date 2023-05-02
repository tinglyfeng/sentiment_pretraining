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


def set_parameter_group(model,cfg,lr):
    teacher_infos = cfg['model']['teachers']
    parameter_groups = [
        {'params' : model.stu.parameters(), 
            'lr' : lr},             
    ]
    if 'low' in teacher_infos:
        if 'lr_factor' in teacher_infos['low']:
            cur_lr = lr * teacher_infos['low']['lr_factor']
        else:
            cur_lr = lr
        parameter_groups.append(
            {
                'params' :  model.t_low.parameters(),
                'lr':  cur_lr
                }
        )
        
    if 'mid' in teacher_infos:
        if 'lr_factor' in teacher_infos['mid']:
            cur_lr = lr * teacher_infos['mid']['lr_factor']
        else:
            cur_lr = lr
        parameter_groups.append(
            {
                'params' :  model.t_mid.parameters(),
                'lr':  cur_lr
                }
        )
        
    if 'high' in teacher_infos:
        if 'lr_factor' in teacher_infos['high']:
            cur_lr = lr * teacher_infos['high']['lr_factor']
        else:
            cur_lr = lr
        parameter_groups.append(
            {
                'params' :  model.t_high.parameters(),
                'lr':  cur_lr
                }
        )
    return parameter_groups

def get_optimizer(model, cfg, lr):
    parameter_group = set_parameter_group(model,cfg,lr)
    optimizer =  torch.optim.SGD(parameter_group, momentum=0.9, weight_decay=0.0001)
    return optimizer


if __name__ == '__main__':
    pass