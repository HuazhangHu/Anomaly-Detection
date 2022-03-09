''' train '''

import os
import numpy as np
from sympy import N
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler


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


    trainloader = DataLoader(train_set, batch_size=batch_size, pin_memory=False, shuffle=True, num_workers=8)
    model = nn.DataParallel(model.cuda())
    param_groups = optim_factory.add_weight_decay(model , weight_decay=weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))

    currEpoch = 0

    if lastckpt is not None: 
        print("loading checkpoint")
        checkpoint = torch.load(lastckpt)
        currEpoch = checkpoint['epoch']
        # # # load hyperparameters by pytorch
        # # # if change model
        # net_dict=model.state_dict()
        # state_dict={k: v for k, v in checkpoint.items() if k in net_dict.keys()}
        # net_dict.update(state_dict)
        # model.load_state_dict(net_dict, strict=False)

        # # # or don't change model
        model.load_state_dict(checkpoint['state_dict'], strict=False)

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        del checkpoint
    print("actual lr: %.2e" % lr)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    scaler = GradScaler()

    for epoch in tqdm(range(currEpoch, n_epochs + currEpoch)):
        pbar = tqdm(trainloader, total=len(trainloader))
        for input in pbar:
            with autocast():
                model.train()
                model.cuda()
                optimizer.zero_grad()
                input = input.cuda()
                print(input.device)
                loss, pred, mask = model(input)
                loss=torch.mean(loss,dim=0)
                loss_value = loss.item()
                pbar.set_postfix({'Epoch': epoch,'loss_train': loss_value})
                
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


N_GPU = 4
device_ids = [i for i in range(N_GPU)]
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


train_path = '/public/home/huhzh/ShanghaiTech/training/feature_videoswin_16'

EPOCHS=4

batch_size=16
lr=1e-6
train_set = FeatData(train_path)
model=Network()
train_looping(EPOCHS, model, train_set, batch_size, lr, device_ids=device_ids)