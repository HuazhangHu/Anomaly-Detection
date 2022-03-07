import sys
sys.path.append("..")
from mmcv import Config
from mmaction.models import build_model
from mmcv.runner import load_checkpoint
import torch 
import torch.nn as nn
from torch.cuda.amp import autocast

class X3D(nn.Module): 
    def __init__(self,OPEN=False):
        super().__init__()
        self.OPEN=OPEN
        self.config='../configs/recognition/x3d/x3d_m_16x5x1_facebook_kinetics400_rgb.py'
        self.checkpoint='../pre-trained/x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth'
        self.X3D = self.load_model()

    def load_model(self):
        """加载预训练模型"""
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

    def forward(self,input):
        # input = input.transpose(1,2)# -> [1, 64, 3, 224, 224]
        with autocast():
            if not self.OPEN: # 不打开网络
                with torch.no_grad():                   
                    x= self.X3D(input)

            else: # 打开网络                    
                x= self.X3D(input)
            
            return x


# model=X3D()
# # print(model)
# x=torch.rand(1,3,64,224,224)
# x=model(x)
# #[1, 432, 64, 7, 7] temporal 64->64  channel 432
# print("output: ", x.shape)