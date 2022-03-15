"""dataloader for masked autoencoder"""

from operator import ne
import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class FeatData(Dataset):
    def __init__(self, feat_path):
        self.feat_path = feat_path
        self.feat_list = os.listdir(feat_path)


    def __getitem__(self, index):
        anchor=index
        if anchor ==len(self.feat_list)-1:
            negative=0
        else:
            negative=anchor+1
        anchor_feat=self.__getfeat(anchor)
        negative_feat=self.__getfeat(negative)
        data= torch.stack((anchor_feat, negative_feat),dim=0) # [2, 8, 1024]

        return data


    def __len__(self):
        return len(self.feat_list)

    def __getfeat(self,index):
        file_name = self.feat_list[index]
        feat_file_path = os.path.join(self.feat_path,file_name)
        feat = load_feat(feat_file_path)  # torch.Size([1024, 8])
        feat = feat.transpose(0, 1)  # torch.Size([1024, 8]) -> [8, 1024]
        return feat

    
def load_feat(file_path):
    feat=torch.load(file_path)
    feat=torch.autograd.Variable(feat,requires_grad = False)   #  这个tensor是由别的模型提取的，要把梯度关掉
    # print('feature shape', feat.shape) # torch.Size([1024, 8])
    return feat



# train_path = '/storage/data/huhzh/ShanghaiTech/training/feature_videoswin_16'
# batch_size=4
# dataset = FeatData(train_path)
# train_set= torch.utils.data.Subset(dataset, range(0, int(0.9 * len(dataset))))
# valid_set = torch.utils.data.Subset(dataset, range(int(0.9*len(dataset)), len(dataset)))
# trainloader = DataLoader(train_set, batch_size=batch_size, pin_memory=False, shuffle=True, num_workers=4)
# validloader = DataLoader(valid_set, batch_size=batch_size, pin_memory=False, shuffle=True, num_workers=4)
# print(' train ',len(trainloader))
# print(' valid ',len(validloader))

# for input in tqdm(trainloader, total=len(trainloader)):
#     print('input shape',input[:,0,:,:].shape) 
#     break



