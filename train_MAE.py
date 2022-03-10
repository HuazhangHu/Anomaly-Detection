''' train '''

import os
from tkinter import S
import numpy as np
from sympy import N
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

import timm
import timm.optim.optim_factory as optim_factory

from sklearn.model_selection import KFold
from Masked_AE import MaskedAutoencoder, Network
from dataloader import FeatData

def train_looping(n_epochs, model, dataset, TTR=0.9, batch_size=4, lr=1e-4, weight_decay=0.05, valid=True, lastckpt=None, save=None, log_dir=None):

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

    ## ------ gpu environment seeting ------
    torch.distributed.init_process_group(backend="nccl")
    num_tasks=torch.distributed.get_world_size()
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    model.to(device)
    model=nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],output_device=local_rank)

    ## ------ dataloader settings ------
    train_set = torch.utils.data.Subset(dataset, range(0, int(TTR * len(dataset))))
    valid_set = torch.utils.data.Subset(dataset, range(int(TTR*len(dataset)), len(dataset)))
    sampler_train=DistributedSampler(train_set,num_replicas=num_tasks,rank=local_rank,shuffle=True)
    sampler_valid=DistributedSampler(valid_set,num_replicas=num_tasks,rank=local_rank,shuffle=True)
    trainloader = DataLoader(train_set, batch_size=batch_size, pin_memory=False, num_workers=8, sampler=sampler_train)
    validloader = DataLoader(valid_set, batch_size=batch_size, pin_memory=False, num_workers=8, sampler=sampler_valid)

    ## ------ optimizer settings  -------
    param_groups = optim_factory.add_weight_decay(model , weight_decay=weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))

    milestones = [i for i in range(0, n_epochs, 40)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.8)  # three step decay
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
        sampler_train.set_epoch(epoch)
        pbar = tqdm(trainloader, total=len(trainloader))
        Trainlosses=[]
        for input in pbar:
            with autocast():
                model.train()
                optimizer.zero_grad()
                input = input.to(device)
                loss, pred, mask = model(input)
                dist.reduce(loss,0)
                if torch.distributed.get_rank()==0:
                    loss_value=loss.item()/num_tasks
                    Trainlosses.append(loss_value)
                    pbar.set_postfix({'Epoch': epoch,'loss_train': loss_value})
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


        if valid:
            sampler_valid.set_epoch(epoch)
            Validlosses=[]
            pbar = tqdm(validloader, total=len(validloader))
            with torch.no_grad():
                for input in pbar:
                    model.eval()
                    input = input.to(device)
                    loss, pred, mask = model(input)
                    dist.reduce(loss,0)
                    if torch.distributed.get_rank()==0:
                        loss_value=loss.item()/num_tasks
                        Validlosses.append(loss_value)
                        pbar.set_postfix({'Epoch': epoch,'loss_valid': loss_value})


        scheduler.step()
        if save:
            if not os.path.exists('checkpoint/{0}'.format(save)):
                os.makedirs('checkpoint/{0}'.format(save))
            if (epoch < 50 and epoch % 5 == 0) or (epoch > 50 and epoch % 3 == 0):
                    checkpoint = {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }
                    if torch.distributed.get_rank()==0:
                        torch.save(checkpoint,os.path.join('checkpoint',save,str(epoch)+'_{0}.pt'.format(round(np.mean(Validlosses),4))))
        if torch.distributed.get_rank()==0:
            log_writer.add_scalars('epoch_loss', {"epoch_trainloss": np.mean(Trainlosses),"epoch_vlidloss": np.mean(Validlosses)}, epoch)
            log_writer.add_scalars('epoch_lr', {"lr": optimizer.state_dict()['param_groups'][0]['lr']}, epoch)

            


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

train_path = '/public/home/huhzh/ShanghaiTech/training/feature_videoswin_16'
EPOCHS=200
batch_size=64
lr=1e-4
lastckpt='checkpoint/0309/267_0.0054.pt'
dataset = FeatData(train_path)

model=Network()
train_looping(EPOCHS, model, dataset=dataset, batch_size=batch_size, lr=lr, save='0309',log_dir='0309', lastckpt=lastckpt)