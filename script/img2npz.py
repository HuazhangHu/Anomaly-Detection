""" merge image of test set into video """

import cv2 
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def merge(video_path):
    frames=[]
    img_list=os.listdir(video_path)
    for i in range(len(img_list)):
        img_name='0'*(3-len(str(i)))+str(i)+'.jpg'
        if not os.path.exists(os.path.join(video_path,img_name)):
            print(img_name+'not exists')
        img=Image.open(os.path.join(video_path,img_name)).resize((224,224),Image.BILINEAR)
        frame=np.array(img)
        frames.append(frame)

    frames=np.array(frames)
    # print(frames.shape)
    save_npz(frames,video_path.split('/')[-1])

def save_npz(frames,file_name):
    if isinstance(frames,np.ndarray):
        savename=os.path.join(save_path,file_name+'.npz')
        np.savez(savename,frames=frames) #data [frames,224,224,3]

    

def img2npz():
    frames_list=os.listdir(root)
    for frames in tqdm(frames_list):
        merge(os.path.join(root,frames))

root ='/public/home/huhzh/ShanghaiTech/testing/frames/'
save_path = '/public/home/huhzh/ShanghaiTech/testing/videos'
img2npz()