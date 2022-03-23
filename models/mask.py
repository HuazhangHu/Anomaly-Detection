import numpy as np
import random


def mask_matrix(x,patch_size=14,patch_t=2,mask_ratio=0.4):
    """
    x: [N,C,T,W,H]
    """
    # mask标为0,保留标为1
    N,C,T,W,H = x.shape
    mask=np.ones((N,C,T,W,H))
    w=h=patch_size
    t=patch_t
    ## 绘制3Dblock坐标图
    block_x=W/w
    block_y=H/h
    block_z=T/t
    block=int(block_x*block_y*block_z)
    # 随机选取masked block
    list=[i for i in range(0,block)]
    sampler=random.sample(list, int(mask_ratio*block))

    for index in sampler:
        # 求出在3D block中的坐标
        z=int(index//(block_x*block_y))
        y=int((index-z*(block_x*block_y))%block_y)
        x=int((index-z*(block_x*block_y))//block_y)
        # 求出在mask矩阵中的坐标
        i=x*h
        j=y*w
        k=z*t
        mask[:,:,k:k+t,i:i+h,j:j+w]=0
    
    return mask