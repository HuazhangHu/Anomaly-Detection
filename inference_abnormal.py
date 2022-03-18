"""step3 inference abnormal snip的PSNR值,并计算score,划分candidate"""


import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import json
import os
import numpy as np
from tqdm import tqdm

from Triple_MAE import MaskedAutoencoder
from psnr_dataloader import FeatData
from utils.loss import l1_loss, psnr_error, Score

def inference(batch_size=1,lastckpt=None):
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda:" + str(device_ids[0]) if torch.cuda.is_available() else "cpu")

    model=MaskedAutoencoder()
    model = nn.DataParallel(model.cuda(), device_ids=device_ids)
    checkpoint = torch.load(lastckpt)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    for view in views:
        testset =FeatData(test_path,view)
        testloader = DataLoader(testset, batch_size=batch_size, pin_memory=False, num_workers=8)
        min_P=load_dict[view]["min"]
        max_P=load_dict[view]['max']
        pbar = tqdm(testloader, total=len(testloader))
        with torch.no_grad():
            for input in pbar:
                model.eval()
                input= input.to(device)
                pred, mask= model(input)    
                loss=l1_loss(pred,input,mask)
                psnr=psnr_error(pred,input,mask)
                score=Score(pred,input,mask,min_P,max_P)     

                print('l1 loss:{0}, psnr:{1}, score:{2}'.format(round(loss.item(),4),round(psnr.item(),4),round(score.item(),4)))


N_GPU =1
device_ids = [i for i in range(N_GPU)]
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
test_path='/storage/data/huhzh/ShanghaiTech/testing/new_feature_videoswin_16'
views=['01','02','03','04','05','06','07','08','09','10','11','12','13']
ckpt='/storage/data/huhzh/Anomaly-Detection/checkpoint/0316_step1/99_0.1771.pt'
batch_size=1
with open('normal_psnr.json','r') as file:
    load_dict = json.load(file)
    inference(batch_size=batch_size,lastckpt=ckpt)



