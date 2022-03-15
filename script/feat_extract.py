""" extracting features through extractor video-swin transformer"""


import sys

from kornia import rotation_matrix_to_quaternion
sys.path.append("..")
import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from Extractor.VideoSwin import VideoSwinTransformer


class MyData(Dataset):
    def __init__(self,video_path):
        self.video_path = video_path
        self.video_list = os.listdir(video_path)

    def __getitem__(self, index):
        video_file_name = self.video_list[index]
        video_file_path = os.path.join(self.video_path,video_file_name)
        frames = get_frames(video_file_path)
        frames = frames.transpose(0,1)  # tensor [f, 3, 224, 224] -> [3, f, 224, 224] for video-swin

        return frames, video_file_name

    def __len__(self):
        """:return the number of video """
        return len(self.video_list)


def get_frames(file_path):
    with np.load(file_path, allow_pickle=True) as data:
        frames = data['frames']  # numpy.ndarray [f, 224, 224, 3]
        frames_length = frames.shape[0]
        #!!!!!!!!!! 加回来 TODO: 由于clip的时候已经归一化了一次，此时不需要归一化了
        frames = torch.FloatTensor(frames)
        frames = frames.permute(0, 3, 1, 2)  # tensor [f, 224, 224, 3]->[f, 3, 224, 224]
        
        # dtype(float64)
        return frames

def save_feat(feature: torch.Tensor, filename, save_path):
    feature=feature.cpu()
    torch.save(feature, os.path.join(save_path,filename.split('.')[0]+'.pt'))



N_GPU = 4
device_ids = [i for i in range(N_GPU)]
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device = torch.device("cuda:" + str(device_ids[0]) if torch.cuda.is_available() else "cpu")

train_path = '/storage/data/huhzh/ShanghaiTech/testing/clips_16'
save_path='/storage/data/huhzh/ShanghaiTech/testing/feature_videoswin_16'
batch_size=4
train_set = MyData(train_path)
trainloader = DataLoader(train_set, batch_size=batch_size, pin_memory=False, shuffle=True, num_workers=8)

model=VideoSwinTransformer()
model = nn.DataParallel(model.to(device), device_ids=device_ids)


for input,filenames in tqdm(trainloader, total=len(trainloader)):
    # print('input shape',input.shape) [3, f, 224, 224]
    input = input.to(device)
    features= model(input)
    # print('feature shape', features.shape)
    # print('filename ', filenames)
    features = torch.chunk(features, batch_size, dim=0)
    for i in range(len(features)):
        save_feat(features[i].squeeze(0), filenames[i], save_path)  # -> torch.Size([1024, 8])







