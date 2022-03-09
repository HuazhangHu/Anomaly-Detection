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

from torch.utils.data.distributed import DistributedSampler

import timm
import timm.optim.optim_factory as optim_factory

from Masked_AE import MaskedAutoencoder, Network
from dataloader import FeatData


def train_looping(n_epochs, model, train_set, batch_size=4, lr=1e-4, weight_decay=0.05, lastckpt=None, device_ids =[0], log_dir=None):

    # fix the seed for reproducibility
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    if log_dir is not None:
        os.makedirs(os.path.join('log', log_dir), exist_ok=True)
        print('log dir : ',os.path.join('log', log_dir))
        log_writer = SummaryWriter(log_dir=log_dir)
    else:
        log_writer = None

    # gpu environment seeting 
    torch.distributed.init_process_group(backend="nccl")

    num_tasks=torch.distributed.get_world_size()
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    sampler=DistributedSampler(train_set,num_replicas=num_tasks,rank=local_rank,shuffle=True)
    trainloader = DataLoader(train_set, batch_size=batch_size, pin_memory=False, num_workers=8, sampler=sampler)

    model.to(device)
    model=nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],output_device=local_rank)
    # model = nn.DataParallel(model.cuda())

    param_groups = optim_factory.add_weight_decay(model , weight_decay=weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))

    currEpoch = 0

    # # # load hyperparameters by pytorch
    if lastckpt is not None: 
        print("loading checkpoint")
        checkpoint = torch.load(lastckpt)
        currEpoch = checkpoint['epoch']

        # # # if change model
        # net_dict=model.state_dict()
        # state_dict={k: v for k, v in checkpoint.items() if k in net_dict.keys()}
        # net_dict.update(state_dict)
        # model.load_state_dict(net_dict, strict=False)

        # # # or don't change model
        model.load_state_dict(checkpoint['state_dict'], strict=False)

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        del checkpoint
    
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    scaler = GradScaler()

    for epoch in tqdm(range(currEpoch, n_epochs + currEpoch)):
        sampler.set_epoch(epoch)
        pbar = tqdm(trainloader, total=len(trainloader))
        for input in pbar:
            with autocast():
                model.train()
                optimizer.zero_grad()
                input = input.to(device)
                loss, pred, mask = model(input)
                loss_value = loss.item()
                pbar.set_postfix({'Epoch': epoch,'loss_train': loss_value})
                
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()




train_path = '/public/home/huhzh/ShanghaiTech/training/feature_videoswin_16'

EPOCHS=4
batch_size=16
lr=1e-6
train_set = FeatData(train_path)
model=Network()
train_looping(EPOCHS, model, train_set, batch_size, lr)