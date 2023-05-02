from audioop import bias
from termios import CKILL
from turtle import forward
from numpy import repeat
from torch import nn
import torch
import torchvision
from torchvision import models
import copy
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from artemis.utils.vocabulary import Vocabulary
from artemis.neural_models.attentive_decoder import AttentiveDecoder
import time


class convUpBlock(nn.Module):
    def __init__(self, scale, in_c, out_c) -> None:
        super(convUpBlock, self).__init__()
        self.up_2x = nn.Upsample(scale_factor=scale, mode = 'bilinear')
        self.conv1x1 = nn.Conv2d(in_c, out_c, 1)
    
    def forward(self,x):
        x = self.conv1x1(x)
        x = self.up_2x(x) ## conv first, scale next
        return x

class upCatBranch(nn.Module):
    def __init__(self, backbone):
        super(upCatBranch,self).__init__()
        if backbone == 'resnet18':
            self.up1 = convUpBlock(4, 64, 64)
            self.up2 = convUpBlock(8, 128, 64)
            self.up3 = convUpBlock(16, 256, 64)
            self.up4 = convUpBlock(32, 512, 64)
        elif backbone == 'resnet50' or backbone == 'resnet101':
            self.up1 = convUpBlock(4, 256, 256)
            self.up2 = convUpBlock(8, 512, 256)
            self.up3 = convUpBlock(16, 1024, 256)
            self.up4 = convUpBlock(32, 2048, 256)
    
    def forward(self, x1, x2, x3, x4):
        x1 = self.up1(x1)
        x2 = self.up2(x2)
        x3 = self.up3(x3)
        x4 = self.up4(x4)
        x = torch.cat((x1, x2, x3, x4), dim = 1)
        return x


class colorHead(nn.Module):
    def __init__(self,backbone) :
        super(colorHead,self).__init__()
        if backbone == 'resnet18':
            self.conv = nn.Conv2d(256, 2, kernel_size=3, bias = True, padding = 3 //2)
        elif backbone == 'resnet50' or backbone == 'resnet101':
            self.conv = nn.Conv2d(1024, 2, kernel_size=3, bias = True, padding = 3 //2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        return self.sigmoid(self.conv(x)) 
    
class srHead(nn.Module):
    def __init__(self, backbone):
        super(srHead,self).__init__()
        if backbone == 'resnet18':
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=3 // 2, output_padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=3 // 2, output_padding=1),
                nn.ReLU(inplace=True)
            )
            self.reconstruction =nn.Sequential(
                nn.Conv2d(256, 3, kernel_size=3, padding=3 // 2), 
                nn.Sigmoid()
            )
        elif backbone == 'resnet50' or backbone == 'resnet101':
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(1024, 1024, kernel_size=3, stride=2, padding=3 // 2, output_padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(1024, 1024, kernel_size=3, stride=2, padding=3 // 2, output_padding=1),
                nn.ReLU(inplace=True)
            )
            self.reconstruction =nn.Sequential(
                nn.Conv2d(1024, 3, kernel_size=3, padding=3 // 2), 
                nn.Sigmoid()
            )

                                            
    def forward(self,x):
        return self.reconstruction(self.deconv(x))


class sceneHead(nn.Module):
    def __init__(self, backbone, classes = 365) :
        super(sceneHead, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        if backbone == 'resnet18':
            self.cls = nn.Linear(512, classes)
        elif backbone == 'resnet50' or backbone == 'resnet101':
            self.cls = nn.Linear(2048, classes)
    def forward(self,x):
        # x = self.avg_pool(x).squeeze()
        return self.cls(x)
    
class jigsawHead(nn.Module):
    def __init__(self, backbone, classes = 1000):
        super(jigsawHead, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        if backbone == 'resnet18':
            self.cls = nn.Linear(512 * 9, classes)
        elif backbone == 'resnet50' or backbone == 'resnet101':
            self.cls = nn.Linear(2048 * 9, classes)
            
        
    def forward(self,x):
        B,T,C,H,W = x.size() ## T = 9
        x = self.avg_pool(x).squeeze()
        x = x.view(B, -1)
        return self.cls(x)
    
class anpHead(nn.Module):
    def __init__(self, backbone, classes = 1513) :
        super(anpHead, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        if backbone == 'resnet18':
            self.cls = nn.Linear(512, classes)
        elif backbone == 'resnet50' or backbone == 'resnet101':
            self.cls = nn.Linear(2048, classes)
            
    def forward(self,x):
        # x = self.avg_pool(x).squeeze()
        return self.cls(x)
    
class captionHead(nn.Module):
    def __init__(self,backbone, caption_info):
        super(captionHead, self).__init__()
        self.vocab = Vocabulary.load(caption_info['vocabulary_path'])
        word_embedding = nn.Embedding(len(self.vocab),caption_info['decoder']['word_embedding_dim'], padding_idx=self.vocab.pad)
        if backbone == 'resnet18':
            encoder_out_dim = 512
        elif backbone == 'resnet50' or backbone == 'resnet101':
            encoder_out_dim = 2048

        emo_ground_dim = 0
        emo_projection_net = None 

        self.decoder = AttentiveDecoder(word_embedding,
                                caption_info['decoder']['rnn_hidden_dim'],
                                encoder_out_dim,
                                caption_info['decoder']['attention_dim'],
                                self.vocab,
                                dropout_rate=caption_info['decoder']['dropout_rate'],
                                teacher_forcing_ratio=caption_info['decoder']['teacher_forcing_ratio'],
                                auxiliary_net=emo_projection_net,
                                auxiliary_dim=emo_ground_dim)
    def forward(self,x,caps):
        return self.decoder(x,caps)


class multiHeadResNet(nn.Module):
    def __init__(self, is_pretrain, tasks_info, model_info) -> None:
        super(multiHeadResNet, self).__init__()
        self.is_pretrain = is_pretrain
        self.tasks_info = tasks_info
        self.model_info = model_info
        self.backbone = model_info['backbone']
        self.tasks = list(self.tasks_info)
        self.build_model()
        self.make_fwd_template()

    def make_fwd_template(self):
        self.fwd_template = dict()
        

    def build_model(self):
        resnet_tv = models.__dict__[self.backbone](pretrained= self.is_pretrain)
        self.stage1 = nn.Sequential(*(list(resnet_tv.children())[:5]))
        self.stage2 = list(resnet_tv.children())[5]
        self.stage3 = list(resnet_tv.children())[6]
        self.stage4 = list(resnet_tv.children())[7]
    
        if 'color' in self.tasks:
            self.color_catUp = upCatBranch(self.backbone)
        
        if 'sr' in self.tasks:
            self.sr_catUp = upCatBranch(self.backbone)
            
        if 'color' in self.tasks:
            self.color_head = colorHead(self.backbone)
            
        if 'sr' in self.tasks:
            self.sr_head = srHead(self.backbone)
            
        if 'scene' in self.tasks:
            self.scene_head = sceneHead(self.backbone)
            
        if 'jigsaw' in self.tasks:
            self.jigsaw_head = jigsawHead(self.backbone)
        
        if 'anp' in self.tasks:
            self.anp_head = anpHead(self.backbone)
            
        if 'caption' in self.tasks:
            self.caption_head = captionHead(self.backbone, self.tasks_info['caption'])
           
            
            
    
    def forward_backbone(self, x, save_stage = False):
        if save_stage:
            x1 = self.stage1(x)
            x2 = self.stage2(x1)
            x3 = self.stage3(x2)
            x4 = self.stage4(x3)
            return x1, x2, x3, x4
        else:
            return self.stage4(self.stage3(self.stage2(self.stage1(x))))
    
    def forward_color(self, x, res):
        # s =  time.time()
        x = x.repeat(1,3,1,1) # notice: repeat the channel dimenstions from 1 (gray) to 3 (RGB)
        x1, x2, x3, x4 = self.forward_backbone(x, save_stage = True)
        catUp = self.color_catUp(x1, x2, x3, x4)
        out = self.color_head(catUp)
        res['color'] = out
        # print('fwd color: ', time.time() -s )
    
    def forward_sr(self, x, res):
        # s =  time.time()
        x1, x2, x3, x4 = self.forward_backbone(x, save_stage = True)
        catUp = self.sr_catUp(x1, x2, x3, x4)
        out = self.sr_head(catUp)
        res['sr'] = out
        # print('fwd sr: ', time.time() -s )
    
    def forward_jigsaw(self, x, res):
        # s =  time.time()
        B,T,C,H,W = x.size() ## T = 9
        x = x.view(B*T, C, H, W)
        x = self.forward_backbone(x, save_stage=False)
        C,new_H, new_W = x.size()[-3:] ## T = 9
        x = x.view(B,T,C,new_H,new_W)
        out = self.jigsaw_head(x)
        res['jigsaw'] = out
        # print('fwd scene: ', time.time() -s )
        
    def forward_scene(self, x, res):
        # s =  time.time()
        x = self.forward_backbone(x, save_stage=False)
        x = F.adaptive_avg_pool2d(x, (1,1)).squeeze()
        out = self.scene_head(x)
        res['scene'] = out
        # print('fwd scene: ', time.time() -s )
        
    def forward_anp(self,x,res):
        x = self.forward_backbone(x, save_stage=False)
        x = F.adaptive_avg_pool2d(x, (1,1)).squeeze()
        out = self.anp_head(x)
        res['anp'] = out
    
    def forward_caption(self,x,caps, res):
        x = self.forward_backbone(x, save_stage=False)
        x = x.permute((0,2,3,1))
        out = self.caption_head(x, caps)
        logits, caps_sorted, decode_lengths, alphas, sort_ind = out
        targets = caps_sorted[:, 1:]
        logits = pack_padded_sequence(logits, decode_lengths, batch_first=True)
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)
        res['caption'] = dict()
        res['caption']['logits'] = logits
        res['caption']['targets'] = targets
        res['caption']['decode_lengths'] = decode_lengths
        res['caption']['alphas'] = alphas
        
    # def forward_scene_anp_caption(self, x, res):
    #     if 'scene' in self.tasks:
    #         self.forward_scene(x, res)
            
    #     if 'anp' in self.tasks:
    #         self.forward_anp(x, res)
            
    #     if 'caption' in self.tasks:
    #         self.forward_caption(x, res)
    
        
        
    def forward(self,input):
        res = copy.deepcopy(self.fwd_template)
        if 'color' in self.tasks or 'sr' in self.tasks:
            self.forward_color(input['color']['gray'],res)
        
        if 'sr' in self.tasks:
            self.forward_sr(input['sr']['img'], res)
            
        if 'jigsaw' in self.tasks:
            self.forward_jigsaw(input['jigsaw']['img'], res)
        
        if 'scene' in self.tasks:
            self.forward_scene(input['gene']['img'],res)
        
        if 'anp' in self.tasks:
            self.forward_anp(input['anp']['img'],res)
        
        if 'caption' in self.tasks:
            self.forward_caption(input['caption']['image'],
                                 input['caption']['tokens'],
                                 res)
            
        return res


if __name__ == '__main__':
    from torch import randn as rd
    
    input = {'color' : rd(2,3,256,256), 'sr' : rd(2,3,128,128), 'gene' : rd(2,3,256,256), 'jigsaw' : rd(2,9, 3, 256, 256)}
    
    tasks = {'color':None, 'sr' : None, 'scene' : None, 'jigsaw' : None}
    
    model = multiHeadResNet(False, tasks)
    
    out = model(input)
    
    print()
    
        