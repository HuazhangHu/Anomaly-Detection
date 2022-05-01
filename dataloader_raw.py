"""dataloader for masked autoencoder from raw clip"""

import os
import numpy as np
from ordered_set import T
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class ClipData(Dataset):
    def __init__(self, clips_path):
        self.clips_path = clips_path
        self.clips_list = os.listdir(clips_path)


    def __getitem__(self, index):
        file_name = self.clips_list[index]
        file_path = os.path.join(self.clips_path,file_name)
        clip = get_data(file_path)  # numpy.ndarray [f, 224, 224, 3]
        frames = torch.tensor(clip)
        frames = frames.permute(3, 0, 1, 2)  # tensor [f, 224, 224, 3]->[3, f, 224, 224]

        return frames


    def __len__(self):
        return len(self.clips_list)


def get_data(file_path):
    with np.load(file_path, allow_pickle=True) as data:
        frames = data['frames']  # numpy.ndarray [f, 224, 224, 3]
        frames_length = frames.shape[0]
        frames-=127.5
        frames/=127.5

    return frames


# path ='/storage/data/huhzh/ShanghaiTech/training/clips_0317'
# batch_size=4
# dataset = ClipData(path)
# train_set= torch.utils.data.Subset(dataset, range(0, int(0.95 * len(dataset))))
# valid_set = torch.utils.data.Subset(dataset, range(int(0.95*len(dataset)), len(dataset)))
# trainloader = DataLoader(train_set, batch_size=batch_size, pin_memory=False, shuffle=True, num_workers=8)
# validloader = DataLoader(valid_set, batch_size=batch_size, pin_memory=False, shuffle=True, num_workers=8)
# print(' train: ',len(trainloader))
# print(' valid :',len(validloader))

# for input in tqdm(trainloader, total=len(trainloader)):

#     print('input shape:',input.shape) 

#     break

