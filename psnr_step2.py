"""step2 计算normal snip各个视角下的PSNR值 """


import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import numpy as np
from tqdm import tqdm

from Triple_MAE import MaskedAutoencoder
from psnr_dataloader import FeatData
from utils.loss import l1_loss, psnr_error,product,cos_theta

def inference(batch_size=1,lastckpt=None):
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda:" + str(device_ids[0]) if torch.cuda.is_available() else "cpu")

    model=MaskedAutoencoder()
    model = nn.DataParallel(model.cuda(), device_ids=device_ids)
    checkpoint = torch.load(lastckpt)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    count=0
    for view in views:
        testset =FeatData(test_path,view)
        testloader = DataLoader(testset, batch_size=batch_size, pin_memory=False, num_workers=8)

        PSNR=[]
        pbar = tqdm(testloader, total=len(testloader))
        with torch.no_grad():
            for input in pbar:
                model.eval()
                input= input.to(device)
                pred, mask= model(input)    
                loss=l1_loss(pred,input,mask)
                psnr=psnr_error(pred,input,mask)
                # prod=product(pred,input,mask)
                # theta=cos_theta(pred,input)
                # print(theta)
                PSNR.append(psnr.item())   
                if psnr.item()>3:
                    count+=1
                

                print('l1 loss:{0}, psnr:{1}'.format(round(loss.item(),4),round(psnr.item(),4)))

        all_view[view]={'min':np.min(PSNR),'max':np.max(PSNR)}
    print(count)
N_GPU =1
device_ids = [i for i in range(N_GPU)]
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
test_path='/storage/data/huhzh/ShanghaiTech/testing/feature_0318'
views=['01','02','03','04','05','06','07','08','09','10','11','12']
all_view={}
ckpt='/storage/data/huhzh/Anomaly-Detection/checkpoint/0318_step1_mask0.5/50_0.0825.pt'
batch_size=1

inference(batch_size=batch_size,lastckpt=ckpt)

# print(all_view)

# import json
# obj=json.dumps(all_view)

# fileobj=open('abnormal_psnr3.json', 'w') 
# fileobj.write(obj)

