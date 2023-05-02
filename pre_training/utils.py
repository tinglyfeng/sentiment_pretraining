from enum import Enum
import torch
import shutil
import os
import logging
import sys
from config import cfg  
from datetime import datetime
logger = logging.getLogger(cfg['logger']['name'])


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        logger.info(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, cfg):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = cfg['solver']['base_lr'] * (0.1 ** (epoch // cfg['solver']['decay_inter']))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



def resume(cfg, model, optimizer, best_acc1):
    if cfg['misc']['resume_path']:
        if os.path.isfile(cfg['misc']['resume_path']):
            logger.info("=> loading checkpoint '{}'".format(cfg['misc']['resume_path']))
            if cfg['device']['gpu'] is None:
                checkpoint = torch.load(cfg['misc']['resume_path'])
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(cfg['device']['gpu'])
                checkpoint = torch.load(cfg['misc']['resume_path'], map_location=loc)
            cfg['solver']['start_epoch'] = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if cfg['device']['gpu'] is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(cfg['device']['gpu'])
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(cfg['misc']['resume_path'], checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(cfg['misc']['resume_path']))
    return model, optimizer, best_acc1


def get_logger(log_name,file_path):
    if logging.getLogger(cfg['logger']['name']).hasHandlers():
        return logging.getLogger(cfg['logger']['name'])
    if cfg['logger']['time_stamp']:
        timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        basename = os.path.basename(file_path)
        dirname = os.path.dirname(file_path)
        basename_with_timestampe = timestamp + '_' + basename
        file_path = os.path.join(dirname, basename_with_timestampe)
        
    log = logging.getLogger(log_name)
    log.setLevel(logging.INFO)
    file_format = logging.Formatter("%(asctime)s %(levelname)s : %(message)s","%Y-%m-%d %H:%M:%S")
    stdout_format = logging.Formatter("%(asctime)s : %(message)s","%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(stdout_format)  
    log.addHandler(ch)
    if file_path is not None:
        fh = logging.FileHandler(file_path)
        fh.setFormatter(file_format)
        log.addHandler(fh)
    return log

