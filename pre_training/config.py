import yaml
import argparse

from yaml.events import NodeEvent

parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', default= './config/config_pretrain.yaml')

parser.add_argument('--backbone', default= None)
args = parser.parse_args()

f = open(args.cfg_file)
cfg = yaml.safe_load(f)
f.close()

if args.backbone is not None:
    cfg['model']['backbone'] = args.backbone

