

from einops import reduce
import torch

def l1_loss(pred_feature: torch.Tensor, target:torch.Tensor,mask):
    loss = torch.abs((pred_feature - target))
    loss = loss.mean(dim=-1)  # [N, L], mean loss per frame
    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches 只计算masked loss
    return loss


def log10(x):
    # 用换底公式换成log10
    return torch.log(x)/torch.log(torch.tensor(10,dtype=torch.float32))


def psnr_error(pred_feature: torch.Tensor, target:torch.Tensor,mask):
    # input_feature: [batchsize, length, channel]
    # pred_feature: [batchsize, length, channel]
    # mask: [b, length]
    # return [b]
    mses=((pred_feature - target)**2).mean(dim=-1)  # Size([b, 8]) 
    masked_mse=(mses * mask).sum(dim=-1)/mask.sum(dim=-1) # [b,1]
    # print('l2 loss: ',masked_mse.item())
    psnr=10 * log10(1 / masked_mse)

    return psnr

def Score(pred_feature: torch.Tensor, target:torch.Tensor, mask, min_P,max_P):
    score=(psnr_error(pred_feature,target,mask)-min_P)/(max_P-min_P)

    return score

def product(pred_feature: torch.Tensor, target:torch.Tensor, mask):
    # input_feature: [b,length, channel]
    # pred_feature: [b,length, channel]
    inner=torch.matmul(pred_feature,target.transpose(-2,-1))
    # print(inner.shape) #[b,length,length]
    return inner

def cos_theta(pred_feature: torch.Tensor, target:torch.Tensor):
    theta=torch.matmul(pred_feature,target.transpose(-2,-1))/(torch.norm(pred_feature)*torch.norm(target))
    # print(theta.shape)
    return theta

def triple_loss(f_a,f_n,f_p,S_n,S_p,sigma):
    ## f_a, f_n, f_p都是masked后的 [b,l,c]
    S_gap=S_n-S_p
    loss=S_gap*(torch.max(0,(torch.norm(f_a-f_n)**2)-(torch.norm(f_a-f_p)**2)+sigma))

    return loss

    

def masked(feature,mask):
    # feature:[b,l,c]
    # mask: [b,l]
    b,l,c=feature.shape
    for i in range(mask.shape[0]):
        row=mask[i]# [0,0,0,0,1,1,1,1]
        for j in range(row.shape[1]):
            if row[j]==0:
                feature[i,j,:]=torch.zeros(1,1,c)   
    return feature
