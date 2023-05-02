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


class single_resnet(nn.Module):
    def __init__(self,info, num_class) -> None:
        super(single_resnet, self).__init__()
        self.pretrained_path = info['pretrained_path']
        self.backbone = info['backbone']
        self.num_class = num_class
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
        state_dict = torch.load(self.pretrained_path, map_location=torch.device('cpu'))
        if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
        filtered_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'stage' in k:
                filtered_state_dict[k] = v
        self.load_state_dict(filtered_state_dict,strict=True)
        
    
    def forward(self,x):
        x1 =self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        
        logits = F.adaptive_avg_pool2d(x4, (1,1))
        logits = logits.squeeze(dim=2).squeeze(dim=2)
        logits = self.head(logits)
        
        
        # x = self.stage4(self.stage3(self.stage2(self.stage1(x))))
        # x = F.adaptive_avg_pool2d(x, (1,1))
        # x = x.squeeze()
        # out = self.head(x)
        return {'f1': x1, 
                'f2' : x2,
                'f3' :  x3, 
                'f4' : x4, 
                'logits': logits}



class resnet_kd(nn.Module):
    def __init__(self,cfg):
        super(resnet_kd, self).__init__()
        self.model_info = cfg['model']
        self.num_class = self.model_info['num_class']
        self.build_teachers()
        self.build_student()
        
    def build_teachers(self):
        teachers_info  = self.model_info['teachers']
        self.teachers = list(teachers_info)
        if 'low' in teachers_info:
            self.t_low = single_resnet(teachers_info['low'],
                                   self.num_class)
        if 'mid' in teachers_info:
            self.t_mid = single_resnet(teachers_info['mid'],
                                   self.num_class)
        if 'high' in teachers_info:
            self.t_high = single_resnet(teachers_info['high'],
                                    self.num_class)
        
    
    def build_student(self):
        self.stu = single_resnet(self.model_info['student'],
                                self.num_class)
    
    def forward(self,x):
        res = dict()
        if 'low' in self.teachers:
            res['low'] = self.t_low(x)
            
        if 'mid' in self.teachers:
            res['mid'] = self.t_mid(x)
            
        if 'high' in  self.teachers:
            res['high'] = self.t_high(x)
        
        res['stu'] = self.stu(x)
        return res
        
        

if __name__ == '__main__':
    print()
    
        