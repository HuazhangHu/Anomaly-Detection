# # import json
# # from Extractor.VideoSwin import VideoSwinTransformer
# # import os
# # import torch
# # import torch.nn as nn
# # from tqdm import tqdm
# # from utils.compute_paramter import compute_paramter
# # import time

# # N_GPU =1
# # device_ids = [i for i in range(N_GPU)]
# # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


# # model=VideoSwinTransformer()
# # compute_paramter(model)

# # device = torch.device("cuda:" + str(device_ids[0]) if torch.cuda.is_available() else "cpu")
# # model = nn.DataParallel(model.cuda(), device_ids=device_ids)
# # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-6)
# # groundtruth=torch.rand(4,8,768)
# # x=torch.rand(4,3,16,224,224)
# # lossMSE = nn.MSELoss()

# # t1=time.time()

# # for i in tqdm(range(10000)):
# #     model.train()
# #     optimizer.zero_grad()
# #     x=x.to(device)
# #     groundtruth=groundtruth.to(device)
# #     output=model(x)
# #     # print(output.shape)
# #     loss=lossMSE(output,groundtruth)
# #     loss.backward()
# #     optimizer.step()

# # import numpy as np
# # import random
# # x=np.zeros((1,3,4,224,224))
# # N, C,T,W,H = x.shape
# # w=h=random.randint(4,8)
# # t=random.randint(1,T)
# # mask=np.ones((N,C,T*2,W+8,H+8))
# # while mask[:,:,:T,:W,:H].sum()>N*C*T*W*H*0.4:
    

# #     # print(W//w-1)
# #     x=random.randint(0,W//w-1)
# #     # print(x)
# #     y=random.randint(0,H//h-1)
# #     z=random.randint(0,T-1)
# #     if mask[:,:,z:z+t,x:x+w,y:y+h].sum()!=0:
# #         mask[:,:,z:z+t,x:x+w,y:y+h]=np.zeros((N,C,t,w,h))


# # mask_mat=mask[:,:,:T,:W,:H]
# # print(mask_mat)

    
# import numpy as np
# import random
# x=np.zeros((4,224,224))
# T,W,H = x.shape
# w=h=14
# t=2
# block_x=W/w
# block_y=H/h
# block_z=T/t
# block=int(block_x*block_y*block_z)
# # print(block)
# mask=np.ones((T,W,H))

# # print(mask_patch)
# list=[i for i in range(0,block)]
# # print(len(list))
# sampler=random.sample(list,int(0.4*block))
# # print(len(sampler))
# for index in sampler:
#     import copy
#     # print(mask.sum())
#     z=int(index//(block_x*block_y))
#     y=int((index-z*(block_x*block_y))%block_y)
#     x=int((index-z*(block_x*block_y))//block_y)
#     i=x*h
#     j=y*w
#     k=z*t
#     mask[k:k+t,i:i+h,j:j+w]=0
#     # print(mask_patch)
#     # print('shape',mask[z:z+t,x:x+w,y:y+h].shape)
#     # mask[np.where(mask[z:z+t,x:x+w,y:y+h]!=np.zeros((t,w,h)))]=np.zeros((t,w,h))
#     # mask[mask_patch]=0
    

# #     # print(W//w-1)
# #     x=random.randint(0,W//w-1)
# #     # print(x)
# #     y=random.randint(0,H//h-1)
# #     z=random.randint(0,T-1)
# #     if mask[:,:,z:z+t,x:x+w,y:y+h].sum()!=0:
# #         mask[:,:,z:z+t,x:x+w,y:y+h]=np.zeros((N,C,t,w,h))


# mask_mat=mask
# # print(mask_mat)
# print(mask_mat.sum()/(T*W*H))

import torch 

model=torch.load('pre-trained/maepretrain.pth',map_location=torch.device('cpu'))
print(model['epoch'])
