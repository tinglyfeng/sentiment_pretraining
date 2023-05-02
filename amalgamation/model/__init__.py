from .resnet import custom_resnet
from .resnet_kd import resnet_kd
from .vgg import custom_vgg
from .vgg_kd import vgg_kd


def build_model(cfg):
    if 'resnet' in cfg['model']['student']['backbone']:
        return resnet_kd(cfg)
    elif 'vgg' in cfg['model']['student']['backbone']:
        return vgg_kd(cfg)

    

