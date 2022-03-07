""" video swin transfomer feature extractor """

from mmcv import Config
from mmaction.models import build_model
from mmcv.runner import load_checkpoint
import torch
import torch.nn as nn
import os.path as osp


class VideoSwinTransformer(nn.Module):

    def __init__(self):
        super(VideoSwinTransformer, self).__init__()
        
        self.config= 'configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py'
        self.checkpoint = 'pre-trained/swin_tiny_patch244_window877_kinetics400_1k.pth'
        check_file_exist(self.checkpoint)
        check_file_exist(self.config)
        self.backbone = self.load_model()
        

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

    def forward(self, input):
        # input.shape [batch_size, channel, length, h, w]
        batch_size, c, length, h, w = input.shape
        output = self.backbone(input)

        return output
    
def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))

# model=VideoSwinTransformer()
# # print(model)
# x=torch.rand(1,3,64,224,224)
# x=model(x)
# print("videoswin output: ", x.shape)



