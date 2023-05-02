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


class custom_vgg(nn.Module):
    def __init__(self,cfg) -> None:
        super(custom_vgg, self).__init__()
        self.pretrained_path = cfg['model']['pretrained_path']
        self.backbone = cfg['model']['backbone']
        self.num_class = cfg['model']['num_class']
        
        if self.pretrained_path == 'tvof':
            vgg_tv =  models.__dict__[self.backbone](pretrained= True)
        else:
            vgg_tv =  models.__dict__[self.backbone](pretrained= False)
        vgg_tv = vgg_tv.features
        if self.backbone == 'vgg16_bn':
            self.stage1 =  vgg_tv[:13]   ## 2x
            self.stage2 =  vgg_tv[13:23] ## 4x
            self.stage3 = vgg_tv[23:33]  ## 8x
            self.stage4 = vgg_tv[33:43]  ## 16X
        elif self.backbone == 'vgg19_bn':
            self.stage1 = vgg_tv[:13]   ## 2x
            self.stage2 = vgg_tv[13:26] ## 4x
            self.stage3 = vgg_tv[26:39] ## 8x
            self.stage4 = vgg_tv[39:52] ## 16x
            
        if self.pretrained_path is not None and self.pretrained_path != 'tvof':
            self.load_custom_backbone()
        
        feature_dim = 512
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
    
        