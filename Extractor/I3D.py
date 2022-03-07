import sys
sys.path.append("..")
from mmcv import Config
from mmaction.models import build_model
from mmcv.runner import load_checkpoint
import torch 
import torch.nn as nn
from torch.cuda.amp import autocast


class I3D(nn.Module): 
    def __init__(self,OPEN=False):
        super(I3D, self).__init__()
        self.OPEN=OPEN
        self.config='../configs/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb.py'
        self.checkpoint='../pre-trained/i3d_r50_32x2x1_100e_kinetics400_rgb_20200614-c25ef9a4.pth'
        self.I3D = self.load_model()
        # self.SpatialPooling = nn.MaxPool3d(kernel_size=(1, 7, 7))

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
        with autocast():
            if not self.OPEN: # 不打开网络
                with torch.no_grad(): 
                    # print('input',input.shape)                  
                    x= self.I3D(input)
            else: # 打开网络                    
                x= self.I3D(input)
            print("I3D output: ", x.shape)  # [1, 2048, 8, 7, 7]
            # output=self.SpatialPooling(x).squeeze(-1).squeeze(-1)
            #[b, 2048,f]
            return x


# model=I3D()
# # print(model)
# x=torch.rand(1,3,64,224,224)
# x=model(x)
# #[1, 2048, 8, 7, 7] temporal 64->8  channel 2048
# print("output: ", x.shape)

