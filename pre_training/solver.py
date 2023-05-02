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


def get_optimizer(model, cfg):
    optimizer =  torch.optim.SGD(model.parameters(), cfg['solver']['base_lr'],
                                momentum=cfg['solver']['momentum'],
                                weight_decay=cfg['solver']['weight_decay'])
    return optimizer


def get_criterions(cfg):
    tasks_info = cfg['tasks_info']
    criterions = dict()
    if 'color' in tasks_info:
        criterions['color'] = nn.MSELoss()
    if 'sr' in tasks_info:
        criterions['sr'] = nn.MSELoss()
    if 'jigsaw' in tasks_info:
        criterions['jigsaw'] = nn.CrossEntropyLoss()
    if 'scene' in tasks_info:
        criterions['scene'] = nn.CrossEntropyLoss()
    if 'anp' in tasks_info:
        criterions['anp'] = nn.CrossEntropyLoss()
    if 'caption' in tasks_info:
        criterions['caption'] = nn.CrossEntropyLoss()
    return criterions


if __name__ == '__main__':
    pass