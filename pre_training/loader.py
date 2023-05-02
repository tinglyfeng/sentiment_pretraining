import torch 
from torchvision import models
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms,datasets
import os
from dataloader import dataset



def get_dataloader(cfg):
    train_dataset =  dataset.multiTaskDataset('train',cfg)
    test_dataset = dataset.multiTaskDataset('test', cfg)
    
    if cfg['device']['distributed']:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg['loader']['batch_size'], shuffle=(train_sampler is None),
        num_workers=cfg['loader']['workers'], pin_memory=True, sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg['loader']['batch_size'],
        shuffle = False,
        num_workers=cfg['loader']['workers'],
        pin_memory=True
    )
    
    return train_loader, test_loader, train_sampler
    