"""dataloader for masked autoencoder"""

import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class FeatData(Dataset):
    def __init__(self, feat_path,view):
        self.feat_path = feat_path
        self.scene=[]
        self.feat_list = os.listdir(feat_path)
        self.select_view(view)
        


    def __getitem__(self, index):
        file_name = self.scene[index]
        feat_file_path = os.path.join(self.feat_path,file_name)
        feat = get_feat(feat_file_path)  # torch.Size([1024, 8])
        feat = feat.transpose(0, 1)  # torch.Size([1024, 8]) -> [8, 1024]

        return feat 

    def __len__(self):
        return len(self.scene)

    def select_view(self, view):
        for file in self.feat_list:
            if file.split('_')[0]==view:
                self.scene.append(file)
            
    

def get_feat(file_path):
    feat=torch.load(file_path)
    feat=torch.autograd.Variable(feat,requires_grad = False)   #  这个tensor是由别的模型提取的，要把梯度关掉
    # print('feature shape', feat.shape) # torch.Size([1024, 8)
    return feat
    



