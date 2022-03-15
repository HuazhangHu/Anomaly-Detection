

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
    mses=((pred_feature - target)**2).mean(dim=-1)  # Size([4, 8]) 
    masked_mse=(mses * mask).sum()/mask.sum() #
    psnr=10 * log10(1 / masked_mse)

    return psnr

def Score(pred_feature: torch.Tensor, target:torch.Tensor, mask, psnr_set):
    min_P=min(psnr_set)
    max_P=max(psnr_set)
    score=(psnr_error(pred_feature,target,mask)-min_P)/(max_P-min_P)

    return score

    


    