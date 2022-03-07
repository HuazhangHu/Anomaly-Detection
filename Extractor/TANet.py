import sys
sys.path.append("..")
from mmcv import Config
from mmaction.models import build_model
from mmcv.runner import load_checkpoint
import torch 
import torch.nn as nn
from torch.cuda.amp import autocast

class TANet(nn.Module): 
    def __init__(self,OPEN=False):
        super().__init__()
        self.OPEN=OPEN
        self.config='../configs/recognition/tanet/tanet_r50_dense_1x1x8_100e_kinetics400_rgb.py'
        self.checkpoint='../pre-trained/tanet_r50_dense_1x1x8_100e_kinetics400_rgb_20210219-032c8e94.pth'
        self.TANet = self.load_model()

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
        input = input.transpose(1,2)# -> [B, Temporal, channel, h, w]
        with autocast():
            if not self.OPEN: 
                with torch.no_grad():
                    slice=[]
                    for index in range(input.shape[0]):
                        x=input[index]
                        x= self.TANet(x)
                        slice.append(x)
                    x=torch.stack(slice,dim=0)# ->[b, 64, 2048,  7, 7]
                    
            else:    
                slice=[]
                for index in range(input.shape[0]):
                    x=input[index]
                    x= self.TANet(x)
                    slice.append(x)
                x=torch.stack(slice,dim=0)# ->[b, 64, 2048,  7, 7]
            
            return x


# model=TANet()
# # print(model)
# x=torch.rand(1,3,64,224,224)
# x=model(x)
# #[1, 64, 2048, 7, 7] temporal 64->64  channel 2048
# print("output: ", x.shape)