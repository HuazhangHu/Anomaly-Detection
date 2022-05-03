''' train '''

import os
from tkinter import S
import numpy as np
from sympy import N
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from utils.lr_sched import adjust_learning_rate
import timm
import timm.optim.optim_factory as optim_factory

from models.MAE import MaskedAutoencoderViT
from dataloader_raw import ClipData

def train_looping(n_epochs, model, dataset, TTR=0.95, batch_size=4, lr=1e-4, valid=True, lastckpt=None, save=None, log_dir=None):

    # fix the seed for reproducibility
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    if log_dir is not None:
        os.makedirs(os.path.join('log', log_dir), exist_ok=True)
        # print('log dir : ',os.path.join('log', log_dir))
        log_writer = SummaryWriter(log_dir=os.path.join('log', log_dir))
    else:
        log_writer = None

    ## ------ gpu environment seeting ------\
    device = torch.device("cuda:" + str(device_ids[0]) if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model.to(device), device_ids=device_ids)
    torch.cuda.empty_cache()
    ## ------ dataloader settings ------
    train_set= torch.utils.data.Subset(dataset, range(0, int(TTR * len(dataset))))
    valid_set = torch.utils.data.Subset(dataset, range(int(TTR*len(dataset)), len(dataset)))
    trainloader = DataLoader(train_set, batch_size=batch_size, pin_memory=False, shuffle=True, num_workers=8)
    validloader = DataLoader(valid_set, batch_size=batch_size, pin_memory=False, shuffle=True, num_workers=8)

    ## ------ optimizer settings  -------
    param_groups = optim_factory.add_weight_decay(model , weight_decay=0.005)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))

    # milestones = [i for i in range(0, n_epochs, 40)]
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.8)  # three step decay
    currEpoch = 0

    # # # load hyperparameters by pytorch
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
    
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    scaler = GradScaler()

    for epoch in tqdm(range(currEpoch, n_epochs + currEpoch)):
        pbar = tqdm(trainloader, total=len(trainloader))
        Trainlosses=[]
        data_iter_step=0
        for input in pbar:
            with autocast():
                model.train()
                optimizer.zero_grad()
                adjust_learning_rate(optimizer, data_iter_step / len(trainloader) + epoch,lr,n_epochs,warmup_epochs=5)
                input = input.to(device)
                loss, pred, mask = model(input)
                loss_value=loss.item()
                Trainlosses.append(loss_value)
                pbar.set_postfix({'Epoch': epoch,'loss_train': loss_value,'lr':optimizer.state_dict()['param_groups'][0]['lr']})
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            data_iter_step+=1

        if valid:
            Validlosses=[]
            pbar = tqdm(validloader, total=len(validloader))
            with torch.no_grad():
                for input in pbar:
                    model.eval()
                    input = input.to(device)
                    loss, pred, mask = model(input)
                    loss_value=loss.item()
                    Validlosses.append(loss_value)
                    pbar.set_postfix({'Epoch': epoch,'loss_valid': loss_value,'lr':optimizer.state_dict()['param_groups'][0]['lr']})

        if save:
            savepath='/storage/data/huhzh/Anomaly-Detection/checkpoint/'
            if not os.path.exists(savepath+save):
                os.makedirs(savepath+save)
            if (epoch < 30 and epoch % 5 == 0) or (epoch > 30 and epoch % 3 == 0):
                    checkpoint = {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }
                    torch.save(checkpoint,os.path.join(savepath,save,str(epoch)+'_{0}.pt'.format(round(np.mean(Validlosses),4))))

        if log_dir is not None:
            os.makedirs(os.path.join('log', log_dir), exist_ok=True)
            # print('log dir : ',os.path.join('log', log_dir))
            log_writer = SummaryWriter(log_dir=os.path.join('log', log_dir))
            log_writer.add_scalars('epoch_loss', {"epoch_trainloss": np.mean(Trainlosses),
                                                    "epoch_vlidloss": np.mean(Validlosses)}, epoch)
            log_writer.add_scalars('epoch_lr', {"lr": optimizer.state_dict()['param_groups'][0]['lr']}, epoch)

            

N_GPU =1
device_ids = [i for i in range(N_GPU)]
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

train_path = '/storage/data/huhzh/ShanghaiTech/training/clips_0317'
EPOCHS=200
batch_size=8
lr=1e-4
lastckpt=None
dataset = ClipData(train_path)

model=MaskedAutoencoderViT()
train_looping(EPOCHS, model, dataset=dataset, batch_size=batch_size, lr=lr, save='VMAE0502',log_dir='VMAE0502', lastckpt=lastckpt)