""" clip video to a mount of snips """

import numpy as np
import os
from tqdm import tqdm
import torch


def read_npz(file):
    with np.load(file, allow_pickle=True) as data:
        print(file)
        frames = data['frames']  # numpy.ndarray
        # frames=process_image(frames)
        frames=torch.FloatTensor(frames) 
        # dtype(float64)
        return frames


def save_clip(clip,file_name,index):
    clip=clip.numpy()
    if isinstance(clip,np.ndarray):
        savename=os.path.join(save_path,file_name.split('.')[0]+'_'+str(index)+'.npz')
        np.savez(savename,frames=clip)
    else:
        print('type error')


# def process_image(frames):
#     # 归一化
#     for i in range(frames.shape[0]):
#         image=frames[i]
#         image = (image - np.min(image)) / (np.max(image) - np.min(image))
#         frames[i]=image
#     return frames

def padding(frames):
    frames_num=frames.shape[0]
    # print("before padding :",frames_num)
    patch=torch.zeros(snip_len-(frames_num%snip_len),224,224,3)
    # patch=np.zeros((snip_len-(frames_num%snip_len),224,224,3))
    # frames=np.array([frames,patch],dtype=object)
    frames=torch.cat([frames,patch],dim=0)
    # print("after padding: ",frames.shape[0])

    return frames


def get_clips(frames):
    # [frames,  224, 224,3]
    frames_num=frames.shape[0]
    if frames_num%snip_len !=0:
        frames=padding(frames)
        frames_num=frames.shape[0]

    # clips=[frames[i:i+snip_len] for i in range(0,frames_num,snip_len) ]
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
video_path='/storage/data/huhzh/ShanghaiTech/training/videos_train_npz'
save_path='/storage/data/huhzh/ShanghaiTech/training/new_clips_'+str(snip_len)
video2clip()