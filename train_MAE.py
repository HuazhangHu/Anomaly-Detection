''' train '''

import os

import torch
from torch.utils.tensorboard import SummaryWriter




# def train_looping(model, args):

#     if args.log_dir is not None:
#         os.makedirs(args.log_dir, exist_ok=True)
#         log_writer = SummaryWriter(log_dir=args.log_dir)

def read_feat(path):
    """take care of dimensions"""
    feat=torch.load(path)
    # print('feature shape', feat.shape) # torch.Size([1, 1024, 8, 7, 7])
    if feat.shape[0] == 1:
        feat = feat.squeeze(0) # -> torch.Size([1024, 8, 7, 7])
        # return feat
    print(feat[2][2]) 


path = '/public/home/huhzh/ShanghaiTech/training/feature_videoswin_16/01_038_8.pt'
read_feat(path)