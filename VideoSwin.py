""" video swin transfomer feature extractor """

from mmcv import Config
from mmaction.models import build_model
from mmcv.runner import load_checkpoint
import torch
import torch.nn as nn

class VideoSwinTransformer(nn.Module):

    def __init__(self, config, checkpoint, ):
        super(VideoSwinTransformer, self).__init__()
    
        self.backbone = self.load_model()


    def load_model(self):
        ### load  pretrained model of video swin transformer using mmaction and mmcv API
        cfg = Config.fromfile(self.config)
        model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))

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
        # print("videoswin input: ", input.shape)
        output = self.backbone(input)
        # print("videoswin output: ", output.shape)

        return output
    



