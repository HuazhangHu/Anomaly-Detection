''' train '''

import os
from tkinter import S
from tkinter.messagebox import NO
import numpy as np
from sympy import N
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

import timm
import timm.optim.optim_factory as optim_factory

from Triple_MAE import MaskedAutoencoder
from pre_dataloader import FeatData
from utils.loss import l1_loss, psnr_error
from utils.lr_sched import WarmUpLR, adjust_learning_rate



def train_step1(n_epochs, TTR=0.9, batch_size=4, lr=1e-6, weight_decay=0.05, valid=True, lastckpt=None, save=None, log_dir=None):

    # fix the seed for reproducibility
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    ## ------ gpu environment seeting ------
    device = torch.device("cuda:" + str(device_ids[0]) if torch.cuda.is_available() else "cpu")
    model=MaskedAutoencoder()
    model = nn.DataParallel(model.cuda(), device_ids=device_ids)

    ## ------ dataloader settings ------
    dataset =FeatData(data_path)
    train_set = torch.utils.data.Subset(dataset, range(0, int(TTR * len(dataset))))
    valid_set = torch.utils.data.Subset(dataset, range(int(TTR*len(dataset)), len(dataset)))
    trainloader = DataLoader(train_set, batch_size=batch_size, pin_memory=False, num_workers=8)
    validloader = DataLoader(valid_set, batch_size=batch_size, pin_memory=False, num_workers=8)
   
    ## ------ load model ------------
    currEpoch = 0
    if lastckpt is not None: 
        # print("loading checkpoint")
        checkpoint = torch.load(lastckpt)
        currEpoch = checkpoint['epoch']
        # # # if change model
        # net_dict=model.state_dict()
        # state_dict={k: v for k, v in checkpoint.items() if k in net_dict.keys()}
        # net_dict.update(state_dict)
        # model.load_state_dict(net_dict, strict=False)

        # # # or don't change model
        model.load_state_dict(checkpoint['state_dict'], strict=False)

        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        del checkpoint

    ## ------ optimizer settings  -------
    param_groups = optim_factory.add_weight_decay(model , weight_decay=weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    scaler = GradScaler()

    for epoch in tqdm(range(currEpoch, n_epochs + currEpoch)):

        pbar = tqdm(trainloader, total=len(trainloader))
        Trainlosses=[]
        PSNR=[]
        Step1=True
        data_iter_step=0
        for input in pbar:
            with autocast():
                model.train()
                optimizer.zero_grad()
                adjust_learning_rate(optimizer, data_iter_step / len(trainloader) + epoch,lr,n_epochs,warmup_epochs=5)
                anchor = input[:,0,:,:]
                negative = input[:,1,:,:]
                anchor = anchor.to(device)
                negative = negative.to(device)
                pred_anchor, mask_anchor = model(anchor)
                pred_negative, mask_negatve = model(negative)
                if Step1:
                    score=1        
                    psnr=psnr_error(pred_negative,negative,mask_negatve)
                    PSNR.append(psnr)
                loss=l1_loss(pred_anchor,anchor,mask_anchor)+ score *l1_loss(pred_negative,negative,mask_negatve)        

                loss_value=loss.item()
                Trainlosses.append(loss_value)
                pbar.set_postfix({'Epoch': epoch,'loss_train': loss_value,'lr':optimizer.state_dict()['param_groups'][0]['lr']})
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            data_iter_step+=1

        if  valid:
            Validlosses=[]
            pbar = tqdm(validloader, total=len(validloader))
            with torch.no_grad():
                for input in pbar:
                    model.eval()
                    anchor = input[:,0,:,:]
                    negative = input[:,1,:,:]
                    anchor = anchor.to(device)
                    negative = negative.to(device)
                    pred_anchor, mask_anchor = model(anchor)
                    pred_negative, mask_negatve = model(negative)
                    if Step1:
                        score=1        
                        psnr=psnr_error(pred_negative,negative,mask_negatve)
                        PSNR.append(psnr)
                    loss=l1_loss(pred_anchor,anchor,mask_anchor)+ score *l1_loss(pred_negative,negative,mask_negatve)        

                    loss_value=loss.item()
                    Validlosses.append(loss_value)
                    pbar.set_postfix({'Epoch': epoch,'loss_valid': loss_value,'lr':optimizer.state_dict()['param_groups'][0]['lr']})


        if log_dir is not None:
            os.makedirs(os.path.join('log', log_dir), exist_ok=True)
            # print('log dir : ',os.path.join('log', log_dir))
            log_writer = SummaryWriter(log_dir=os.path.join('log', log_dir))
            log_writer.add_scalars('epoch_loss', {"epoch_trainloss": np.mean(Trainlosses),"epoch_vlidloss": np.mean(Validlosses)}, epoch)
            log_writer.add_scalars('epoch_lr', {"lr": optimizer.state_dict()['param_groups'][0]['lr']}, epoch)

        if save:
            savepath='/storage/data/huhzh/Anomaly-Detection/checkpoint/'
            if not os.path.exists(savepath+save):
                os.makedirs(savepath+save)
            if (epoch < 10 and epoch % 5 == 0) or (epoch > 10 and epoch % 3 == 0):
                    checkpoint = {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'PSNR':PSNR
                    }
                    torch.save(checkpoint,os.path.join(savepath,save,str(epoch)+'_{0}.pt'.format(round(np.mean(Validlosses),4))))

N_GPU =4
device_ids = [i for i in range(N_GPU)]
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
data_path='/storage/data/huhzh/ShanghaiTech/training/feature_videoswin_16'
ckpt=None
LR=1e-5
batch_size=64

train_step1(100,lr=LR,batch_size=batch_size,lastckpt=ckpt,save='0316_step1',log_dir='0316_step1_warm_cos')