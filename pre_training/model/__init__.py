from .resnet import multiHeadResNet
from .vgg import multiHeadVgg

def build_model(cfg):
    if 'resnet' in cfg['model']['backbone']:
        return multiHeadResNet(is_pretrain = cfg['model']['pretrain'], tasks_info=cfg['tasks_info'], model_info= cfg['model'] )
    elif 'vgg' in cfg['model']['backbone']:
        return multiHeadVgg(is_pretrain = cfg['model']['pretrain'], tasks_info=cfg['tasks_info'], model_info= cfg['model'])