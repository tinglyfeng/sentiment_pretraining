from audioop import bias
from termios import CKILL
from turtle import forward

from numpy import repeat
from sklearn import linear_model
from torch import nn
import torch
import torchvision
import torchvision.models as models
import copy
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from collections import OrderedDict


class custom_resnet(nn.Module):
    def __init__(self,cfg) -> None:
        super(custom_resnet, self).__init__()
        self.pretrained_path = cfg['model']['pretrained_path']
        self.backbone = cfg['model']['backbone']
        self.num_class = cfg['model']['num_class']
        if self.pretrained_path == 'tvof':
            resnet_tv =  models.__dict__[self.backbone](pretrained= True)
        else:
            resnet_tv =  models.__dict__[self.backbone](pretrained= False)
        self.stage1 = nn.Sequential(*(list(resnet_tv.children())[:5]))
        self.stage2 = list(resnet_tv.children())[5]
        self.stage3 = list(resnet_tv.children())[6]
        self.stage4 = list(resnet_tv.children())[7]
        if self.pretrained_path is not None and self.pretrained_path != 'tvof':
            self.load_custom_backbone()
        
        feature_dim = resnet_tv.fc.in_features
        self.head = nn.Linear(feature_dim, self.num_class)


    def load_custom_backbone(self):
        state_dict = torch.load(self.pretrained_path)
        if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
        filtered_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'stage' in k:
                filtered_state_dict[k] = v
        self.load_state_dict(filtered_state_dict)
        
    
    def forward(self,x):
        x = self.stage4(self.stage3(self.stage2(self.stage1(x))))
        x = F.adaptive_avg_pool2d(x, (1,1))
        x = x.squeeze()
        out = self.head(x)
        return out


        


if __name__ == '__main__':
    print()
    
        