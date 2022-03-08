""" clip video to a mount of snips """

import numpy as np
import os
from tqdm import tqdm
import torch


def read_npz(file):
    with np.load(file, allow_pickle=True) as data:
        frames = data['frames']  # numpy.ndarray
        frames_length = frames.shape[0]
        frames = torch.FloatTensor(frames)

        frames -=127.5
        frames /= 127.5 
        # dtype(float64)
    return frames


def save_clip(clip,file_name,index):
    clip=clip.numpy()
    if isinstance(clip,np.ndarray):
        savename=os.path.join(save_path,file_name.split('.')[0]+'_'+str(index)+'.npz')
        np.savez(savename,frames=clip)
    else:
        print('type error')


def padding(frames):
    frames_num=frames.shape[0]
    # print("before padding :",frames_num)
    patch=torch.zeros(snip_len-(frames_num%snip_len),224,224,3)
    frames=torch.cat([frames,patch],dim=0)
    # print("after padding: ",frames.shape[0])

    return frames


def get_clips(frames):
    # [frames,  224, 224,3]
    frames_num=frames.shape[0]
    if frames_num%snip_len !=0:
        frames=padding(frames)
        frames_num=frames.shape[0]

    clips=torch.chunk(frames,frames_num//snip_len,dim=0)
    # clips tuple
    
    return clips

def video2clip():
    video_list=os.listdir(video_path)
    for video in tqdm(video_list):
        frames=read_npz(os.path.join(video_path,video))
        clips=get_clips(frames)
        for index,clip in enumerate(clips):
            save_clip(clip, video, index)
        print(video+' saved!')
        

snip_len=16
video_path='/public/home/huhzh/ShanghaiTech/training/videos_train_npz'
save_path='/public/home/huhzh/ShanghaiTech/training/clips'
video2clip()