''' train '''

import os

import torch
from torch.utils.tensorboard import SummaryWriter




def train_looping(model, args):

    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)