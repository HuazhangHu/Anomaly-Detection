""" video swin transfomer feature extractor """

from re import X
import sys
sys.path.append("..")

from mmcv import Config
from mmaction.models import build_model
from mmcv.runner import load_checkpoint
import torch
import torch.nn as nn
import os.path as osp


class VideoSwinTransformer(nn.Module):

    def __init__(self):
        super(VideoSwinTransformer, self).__init__()
        
         # video-swin model base on kinetics400 pre-trained on ImageNet22K
        self.config= '../configs/recognition/swin/swin_base_patch244_window877_kinetics400_22k.py' 
        self.checkpoint = '../pre-trained/swin_base_patch244_window877_kinetics400_22k.pth'
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
        # input.shape [batch_size, 3, 16, 224,224]
        batch_size, c, length, h, w = x.shape
        slices = torch.chunk(x, 8, dim=2)
        featureSet=[]
        for slice in slices:
            feature=self.backbone(slice)  # ->[batch_size, 1024,1,7,7]
            feature=self.average_pooling(feature)  #  [batch_size, channel, length,1,1]
            feature=feature.squeeze(-1).squeeze(-1)  # ->[batch_size,1024,1]
            feature=feature.transpose(1,2)  # ->[batch_size,1,1024]
            featureSet.append(feature)
        output=torch.cat(featureSet,dim=1)  # ->[batch_size,8,1024]

        return output
    
def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))

# model=VideoSwinTransformer()
# # print(model)
# x=torch.rand(1,3,16,224,224)
# x=model(x) # [1,1024,32,7,7]  temporal 64->32  channel 1024
# print("videoswin output: ", x.shape)



