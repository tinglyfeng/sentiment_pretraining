import torch 
from torchvision import models
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms,datasets
import os
from datasets import clc_dataset_folder

proto_train_transforms1 =  transforms.Compose([
    transforms.Resize(size= (256,256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
])

proto_test_transforms1 =  transforms.Compose([
    transforms.Resize(size = (256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
])


def get_dataloader(dataset_path, batch_size):
    train_dataset = clc_dataset_folder(dataset_path, "train", proto_train_transforms1)
    test_dataset = clc_dataset_folder(dataset_path, "test", proto_test_transforms1)

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=8, pin_memory=True, sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True)
    
    return train_loader, test_loader, train_sampler