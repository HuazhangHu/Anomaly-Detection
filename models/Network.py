""" masked cube + video swin """

from re import X
import sys



from mmcv import Config
from mmaction.models import build_model
from mmcv.runner import load_checkpoint
import torch
import torch.nn as nn
import os.path as osp

import math
from timm.models.vision_transformer import Block

from mask import mask_matrix

class VideoSwinTransformer(nn.Module):

    def __init__(self):
        super(VideoSwinTransformer, self).__init__()
        
         # video-swin model base on kinetics400 pre-trained on ImageNet22K
        self.config= './configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py' 
        self.checkpoint = './pre-trained/swin_tiny_patch244_window877_kinetics400_1k.pth'
        check_file_exist(self.checkpoint)
        check_file_exist(self.config)
        self.backbone = self.load_model()
        self.average_pooling= nn.AvgPool3d(kernel_size=(1,7,7))
        

    def load_model(self):
        ### load  pretrained model of video swin transformer using mmaction and mmcv API
        cfg = Config.fromfile(self.config)
        model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))

        # alternative
        ### load hyperparameters by mmcv api 
        # load_checkpoint(model, self.checkpoint, map_location='cpu')
        # backbone = model.backbone

        ### load hyperparameters by pytorch 
        loaded_ckpt = torch.load(self.checkpoint)
        backbone = model.backbone
        net_dict = backbone.state_dict()
        state_dict={k: v for k, v in loaded_ckpt.items() if k in net_dict.keys()}
        net_dict.update(state_dict)   
        backbone.load_state_dict(net_dict, strict=False)

        print('--------- backbone loaded ------------')

        return backbone

    def forward(self, x):
        # input.shape [batch_size, 3, 2, 224,224]
        batch_size, c, length, h, w = x.shape
        x = self.backbone(x) #  ->[batch_size, 768, c/2,7,7]


        return x
    


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.swin=VideoSwinTransformer()

    def masking(self,x,mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, C,T,W,H], sequence
        """
        mask_mat=mask_matrix(x,mask_ratio=mask_ratio)
        mask=torch.tensor(mask_mat,device=x.device)
        if mask.shape!=x.shape:
            print('shape error')
        else:
            x_masked=x * mask

            return x_masked, mask
        

    def forward(self,x):
         # tensor [b,3, 16, 224, 224] for video-swin
        b,c,t,w,h=x.shape
        x_masked,mask=self.masking(x,mask_ratio=0.6)
        slices = torch.chunk(x_masked, 4, dim=2)
        featureSet=[]
        for slice in slices:
            feature=self.swin(slice)  # ->[batch_size, 768, 2]
            feature=feature.transpose(1,2)  # ->[batch_size, 2, 768]
            featureSet.append(feature)
        x=torch.cat(featureSet,dim=1)# ->[batch_size,8,1024]


        return x



def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


model=Model()
x=torch.rand(1,3,16,224,224)
x=model(x)
print(x.shape)
