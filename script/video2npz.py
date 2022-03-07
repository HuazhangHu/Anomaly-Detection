""" video to npz file"""
import numpy as np
import cv2 
import os
import os.path as osp
from tqdm import tqdm

def get_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    # assert cap.isOpened()
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))  # [64, 3, 224, 224]
        frames.append(frame)
    cap.release()
    frames_len = len(frames)
    frames=np.array(frames) # to narray

    return frames

def save_npz(frames,file_name):
    if isinstance(frames,np.ndarray):
        savename=os.path.join(save_path,file_name.split('.')[0]+'.npz')
        np.savez(savename,frames=frames)


def video2npz():
    video_list=os.listdir(video_path)
    for video in tqdm(video_list):
        # print(video)
        # check_file_exist(video)
        frames=get_frames(os.path.join(video_path,video))
        print(frames.shape)
        save_npz(frames,video)
    print('over!')


# def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
#     if not osp.isfile(filename):
#         raise FileNotFoundError(msg_tmpl.format(filename))

def get_npz(npz_path):
    with np.load(npz_path,allow_pickle=True) as data:
        frames = data['frames']
        # print(frames)
    return frames


video_path='/public/home/huhzh/ShanghaiTech/training/videos'
save_path='/public/home/huhzh/ShanghaiTech/training/videos_train_npz'
video2npz()


# f=get_npz('/public/home/huhzh/ShanghaiTech/training/videos_train_npz/01_001.npz')
# print(type(f))
# print(f.shape)

# f=get_frames('/public/home/huhzh/ShanghaiTech/training/videos/01_001.avi')
# print(type(f))
# print(f.shape)

