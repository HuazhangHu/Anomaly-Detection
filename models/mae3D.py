# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
from turtle import pos

import torch
import torch.nn as nn

from timm.models.vision_transformer import  Block
from PatchEmbed3D import PatchEmbed3D

from pos_embed import PositionalEncodingPermute3D,PositionalEncoding3D


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, temp_dim=16,patch_size=(4,16,16), in_chans=3,
                 embed_dim=1024, depth=2, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed3D(img_size,temp_dim, patch_size, in_chans, embed_dim)
        self.pos_encoding=PositionalEncodingPermute3D(embed_dim)
        num_patches = self.patch_embed.num_patches
 

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
 

        self.norm_pix_loss = norm_pix_loss



    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        # x [B,3,T,W,H]
        B,C,T,W,H=x.shape
        embedding = self.patch_embed(x) # -> [B self.embed_dim Dd Ww Hh]
        pe=self.pos_encoding(embedding.permute(0,3,4,2,1))# [B self.embed_dim Dd Ww Hh]-> [B,Ww, Hh, Dd, embed_dim]
        position_embeding=pe.permute(0,4,3,1,2) #-> [B self.embed_dim Dd Ww Hh]
        x=embedding+position_embeding
        x=x.flatten(2).transpose(1,2) # [B, self.embed_dim, Dd*Wh*Ww ]-> [B, N, self.embed_dim]
        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore

   
    def forward(self, imgs, mask_ratio=0.5):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)

        return  mask


model=MaskedAutoencoderViT()
x=torch.rand(1,3,16,224,224)
out=model(x)