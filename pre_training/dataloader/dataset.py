from multiprocessing.sharedctypes import Value
import traceback
from tty import CFLAG
from sklearn.feature_extraction import img_to_graph
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import os
import torchvision.io as tio
from yaml import parse
from PIL import Image
import torch
from dataloader.transform import *
import pandas as pd
import copy
from artemis.in_out import neural_net_oriented as nno
from artemis.utils.vocabulary import Vocabulary
from ast import literal_eval


def read_txt(path):
    f = open(path)
    lines = f.readlines()
    lines = [line.strip() for line in lines]
    f.close()
    return lines


class multiTaskDataset(Dataset):
    def __init__(self,split,cfg):
        self.split = split
        self.tasks_info = cfg['tasks_info']
        self.tasks = list(self.tasks_info)
        
        # if split == 'train':
        #     self.split_txt = cfg['dataset']['scene']['train_split']
        # elif split == 'test':
        #     self.split_txt = cfg['dataset']['scene']['test_split']
        # else:
        #     raise ValueError


        self.parse_data(cfg)
        self.parse_transform(cfg)


    def parse_data(self,cfg):
        if any([task in ['color', 'sr', 'scene', 'jigsaw'] for task in self.tasks]):
            if self.split == 'train':
                split_txt = cfg['dataset']['scene']['train_split']
            elif self.split == 'test':
                split_txt = cfg['dataset']['scene']['test_split']
            else:
                raise ValueError
            self.scene_data_root = cfg['dataset']['scene']['data_root']
            self.scene_img_paths = []
            self.scene_labels = []
            lines = read_txt(os.path.join(self.scene_data_root, split_txt))
            for line in lines:
                elems = line.split('   ')
                img_name = elems[0]
                label = int(elems[1])
                img_path = os.path.join(self.scene_data_root, img_name)
                self.scene_img_paths.append(img_path)
                self.scene_labels.append(label)
                
        if 'anp' in self.tasks:
            if self.split == 'train':
                split_txt = cfg['dataset']['anp']['train_split']
            elif self.split == 'test':
                split_txt = cfg['dataset']['anp']['test_split']
            else:
                raise ValueError
            self.anp_data_root = cfg['dataset']['anp']['data_root']
            self.anp_img_paths = []
            self.anp_labels = []
            lines = read_txt(os.path.join(self.anp_data_root, split_txt))
            for line in lines:
                elems = line.split('   ')
                img_name = elems[0]
                label = int(elems[1])
                img_path = os.path.join(self.anp_data_root, img_name)
                self.anp_img_paths.append(img_path)
                self.anp_labels.append(label)
            self.anp_data_len = len(self.anp_img_paths)
            
        if 'caption' in self.tasks:
            #vocab = Vocabulary.load(os.path.join(cfg['dataset']['caption'], './processed/vocabulary.pkl'))
            self.art_df = pd.read_csv(os.path.join(cfg['dataset']['caption']['data_root'], './processed/artemis_preprocessed.csv'))
            self.art_df.tokens_encoded = self.art_df.tokens_encoded.apply(literal_eval)
            img_transforms = nno.image_transformation(img_dim=
                self.tasks_info['caption']['resize_size'], lanczos=
                self.tasks_info['caption']['lanczos'])

            self.art_dataset = nno.pass_artemis_splits_to_datasets(
                df = self.art_df, load_imgs=True, img_transforms= img_transforms,
                top_img_dir = os.path.join(cfg['dataset']['caption']['data_root'],
                                          'wikiart' ),
                one_hot_emo= True   
            )[self.split]
            self.art_data_len = len(self.art_dataset)

    
    def parse_transform(self,cfg):
        
        self.norm_rgb_t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std =[0.229, 0.224, 0.225]),
        ])
        self.data_template = dict()
        for task in self.tasks:
            self.data_template[task] = dict()
        if any([task in ['color', 'sr', 'scene', 'jigsaw'] for task in self.tasks]):
            self.pre_t_r = transforms.Resize(self.tasks_info['gene']['pre_t']['resize_size'])
            self.pre_t_cf = transforms.Compose([
                # transforms.Resize(self.tasks_info['gene']['pre_t']['resize_size']),
                transforms.RandomCrop(self.tasks_info['gene']['pre_t']['crop_size']),
                transforms.RandomHorizontalFlip(),
            ])
            self.data_template['gene'] = dict()  ## general img for place365 
                
            if 'color' in self.tasks:
                self.color_t = Color(self.tasks_info['color']['size'])
                self.norm_l_t =transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485,],std =[0.229,]),
                ])
                
                self.norm_ab_t =transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.456, 0.406],std =[0.224, 0.225]),
                ])
        
            if 'sr' in self.tasks:
                self.sr_t = SR(self.tasks_info['sr']['size'])
                
            if 'jigsaw' in self.tasks:
                self.jigsaw_t = JigSaw(self.tasks_info['jigsaw']['size'],self.tasks_info['jigsaw']['permu_path'])
                
            if 'scene' in self.tasks:
                pass
            
            
        if 'anp' in self.tasks:
            self.anp_t = transforms.Compose([
                transforms.Resize(self.tasks_info['anp']['resize_size']),
                transforms.RandomCrop(self.tasks_info['anp']['crop_size']),
                transforms.RandomHorizontalFlip(p = 0.5),
                self.norm_rgb_t
            ])
            
       
    def __getitem__(self, index):
        res = copy.deepcopy(self.data_template)
        if any([task in ['color', 'sr', 'scene', 'jigsaw'] for task in self.tasks]):
            ## notice pil format image and numpy format image and keeping the consistency of rgb order in conversion:
            # https://note.nkmk.me/en/python-opencv-bgr-rgb-cvtcolor/#:~:text=When%20the%20image%20file%20is,to%20convert%20BGR%20and%20RGB.
           
            img_path = self.scene_img_paths[index]
                
            img_pil = Image.open(img_path).convert('RGB')
            img_pil_r = self.pre_t_r(img_pil)
            img_pil_rcf = self.pre_t_cf(img_pil_r)  ## rgb order 
            
            img_np_r = np.array(copy.deepcopy(img_pil_r))[:,:,::-1]
            img_np_rcf = np.array(copy.deepcopy(img_pil_rcf))[:,:,::-1]  ## bgr order
            
            # check if rgb or bgr order is consistent with pil and opencv
            # cv2.imwrite('zz_cv2.jpg', img_np)
            # img_pil.save('zz_pil.jpg')


            if 'color' in self.tasks:  
                img_l, img_ab = self.color_t(img_np_rcf)
                res['color']['gray'] = (img_l-50) / 50  ## input, norm to [-1,1]
                res['color']['ab'] = (img_ab + 127) / 255 ## target, norm to [0,1]
                
            if 'sr' in self.tasks:
                img_sr_input = self.sr_t(img_pil_r) ## resize to low resolution
                img_sr_target = img_np_r / 255 ## target, norm to [0,1]
                img_sr_target = img_sr_target.astype('float32')
                res['sr']['img'] = self.norm_rgb_t(img_sr_input) ## norm to [0,1]
                res['sr']['tar'] = img_sr_target
                
            if 'jigsaw' in self.tasks:
                img_jigsaw, order = self.jigsaw_t(img_pil_rcf)
                res['jigsaw']['order'] = order  
                res['jigsaw']['img'] = torch.stack([self.norm_rgb_t(img) for img in img_jigsaw])
            
            # if 'scene' in self.tasks or 'anp' in self.tasks or 'caption' in self.tasks:
                
            if 'scene' in self.tasks:
                res['scene']['label'] = self.scene_labels[index]
                
            res['gene']['img'] = self.norm_rgb_t(img_pil_rcf)
            res['gene']['path'] = img_path
            
            # check out if img_np or img_pil was inplacely modified by the above transforms
            # cv2.imwrite('zz_cv2.jpg', img_np)
            # img_pil.save('zz_pil.jpg')
        
        if 'anp' in self.tasks:
            anp_index = index % self.anp_data_len   
            img_path = self.anp_img_paths[anp_index]
            img_pil = Image.open(img_path).convert('RGB')
            res['anp']['img'] = self.anp_t(img_pil)
            res['anp']['label'] = self.anp_labels[anp_index]
        
        if 'caption' in self.tasks:
            art_index  = index % self.art_data_len
            res['caption'] = self.art_dataset[art_index]
        
        return res
    
        

    
    def __len__(self):
        if any([task in ['color', 'sr', 'scene', 'jigsaw'] for task in self.tasks]):
            return len(self.scene_img_paths) ## based on scene
        else:
            return self.art_data_len


if __name__ == '__main__':
    pass
    
   
            