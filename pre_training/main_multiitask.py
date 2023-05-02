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
from modeling import create_model,set_model
from pprint import pformat
from PIL import ImageFile
from artemis.neural_models.word_embeddings import init_token_bias
ImageFile.LOAD_TRUNCATED_IMAGES = True


from config import cfg
logger = get_logger(cfg['logger']['name'],cfg['logger']['path'])
best_acc1 = 0

def main():

    if cfg['device']['gpu'] is not None:
        warnings.warn('You have chosen single specific GPU. This will completely ' 'disable data parallelism.')
        cfg['device']['mulp_dist_enable'] = False


    if cfg['device']['url'] == "env://" and cfg['device']['world_size']  == -1:
        cfg['device']['world_size'] = int(os.environ["WORLD_SIZE"])

    ## if world size > 1, distributed is implicitly implying enabled
    cfg['device']['distributed'] = cfg['device']['world_size'] > 1 or cfg['device']['mulp_dist_enable']
    

    ngpus_per_node = torch.cuda.device_count()
    if cfg['device']['mulp_dist_enable']:
        # Since we have ngpus_per_node processes per node, the total world_size needs to be adjusted accordingly
        cfg['device']['world_size'] = ngpus_per_node * cfg['device']['world_size']
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg))
    else:
        # Simply call main_worker function
        main_worker(cfg['device']['gpu'], ngpus_per_node, cfg)


def main_worker(gpu, ngpus_per_node, cfg):
    global best_acc1
    cfg['device']['gpu'] = gpu

    if cfg['device']['gpu'] is not None:
        logger.info("Use GPU: {} for training".format(cfg['device']['gpu']))

    if cfg['device']['distributed'] :
        if cfg['device']['url'] == "env://" and cfg['device']['rank'] == -1:
            cfg['device']['rank'] = int(os.environ["RANK"])

        if cfg['device']['mulp_dist_enable']:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            cfg['device']['rank'] = cfg['device']['rank'] * ngpus_per_node + gpu
        dist.init_process_group(backend=cfg['device']['backend'],           init_method=cfg['device']['url'],
                                world_size=cfg['device']['world_size'], rank=cfg['device']['rank'])

    if not cfg['device']['mulp_dist_enable'] or cfg['device']['gpu'] == 0:
        logger.info("\n\n\n\n" + "*" * 50 + 'new_expr' + "*" * 50)
        logger.info('\n' + pformat(cfg))
        
    model = create_model(cfg)

    # define loss function (criterion) and optimizer
    criterions = get_criterions(cfg)

    optimizer = get_optimizer(model,cfg)
    # optionally resume from a checkpoint
    model, optimizer, best_acc1 = resume(cfg, model, optimizer, best_acc1)

    cudnn.benchmark = True

    train_loader, test_loader, train_sampler = get_dataloader(cfg)
    
    if 'caption' in cfg['tasks_info'].keys():
        token_bias = init_token_bias(train_loader.dataset.art_dataset.tokens,
                                     model.caption_head.vocab)
        model.caption_head.decoder.next_word.bias = token_bias
        
    set_model(cfg,ngpus_per_node,model)
        

    if cfg['misc']['evaluate']:
        if cfg['task'] == 'cls':
            validate_cls(test_loader, model, criterions,cfg)
            return
        elif cfg['task'] == 'ldl':
            pass  ## not implemented now
        elif cfg['task'] == 'pretrain':
            val_pretrain(test_loader, model, criterions, optimizer, 0, cfg)
        else:
            raise ValueError
        

    for epoch in range(cfg['solver']['start_epoch'], cfg['solver']['epochs']):
        if cfg['device']['distributed']:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, cfg)

        # train for one epoch
        if cfg['task'] == 'cls':
            train_cls(train_loader, model, criterions, optimizer, epoch, cfg)
            # evaluate on validation set
            acc1 = validate_cls(test_loader, model, criterions,cfg)
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
        elif cfg['task'] == 'ldl':
            train_ldl(train_loader, model, criterions, optimizer, epoch, cfg)
            validate_ldl(test_loader, model, criterions,cfg)
        elif cfg['task'] == 'mll':
            train_mll(train_loader, model, criterions, optimizer, epoch, cfg)
            # validate_mll(test_loader, model, criterions,cfg)
        elif cfg['task'] == 'pretrain':
            train_pretrain(train_loader, model, criterions, optimizer, epoch, cfg)
            val_pretrain(test_loader, model, criterions, optimizer, epoch, cfg)
        else:
            raise ValueError

        if not cfg['device']['mulp_dist_enable'] or (cfg['device']['mulp_dist_enable']
                and cfg['device']['rank'] % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': cfg['model']['backbone'],
                'state_dict': model.state_dict() if not hasattr(model,'module') else model.module.state_dict() ,
                'best_acc1': None,
                'optimizer' : optimizer.state_dict(),
            }, False, filename= cfg['misc']['model_save_name'])


if __name__ == '__main__':
    #%%
    import yaml
    from collections import OrderedDict
    main()