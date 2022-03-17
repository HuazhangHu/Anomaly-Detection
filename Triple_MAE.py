'''' 基于triple loss 方法的mae
'''

from functools import partial
from re import T
from time import sleep
from turtle import Turtle

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from timm.models.vision_transformer import Block


class MaskedAutoencoder(nn.Module): 
    
    def __init__(self, tempotal_len=8,in_chans=1024, embed_dim=1024, encoder_depth=4, num_heads=8,
                 decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=8,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        # No class token
        self.encoder_embed= nn.Linear(in_chans, embed_dim, bias=True)
        self.pos_embed=PositionalEncoding(embed_dim)

        self.encoder_blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,norm_layer=norm_layer)
            for i in range(encoder_depth)]) # transformer encoder  
        self.norm = norm_layer(embed_dim)  
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed=PositionalEncoding(decoder_embed_dim)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.decoder_pred = nn.Linear(decoder_embed_dim, in_chans, bias=True)  # decoder to feature
        # --------------------------------------------------------------------------
        
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch_size, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample  
        # ascend: small is keep, large is remove 
        ids_shuffle = torch.argsort(noise, dim=1)  #  torch.argsort 返回排序后的索引
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # 元素按从小达到排序，
 
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)  # 收集输入的特定维度指定位置的数值

        return x_masked, mask, ids_restore

    def forward(self,x,mask_ratio=0.125):
        ''' 
        x : input feature [batch_size, length, dim]
        '''
        latent, mask, ids_restore=self.forward_encoder(x,mask_ratio)
        # print("encoder size:",latent.shape) # [batch_size, length, dim]
        # print('mask shape',mask.shape)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, f_size*f_size*channel]
        # print("decoder size:",x.shape)
        return pred, mask


    def forward_encoder(self, x, mask_ratio):
        x=self.encoder_embed(x)

        x=x+self.pos_embed(x)

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        for blk in self.encoder_blocks:
            x = blk(x)

        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        # x.shape [batch_size, length, dim]
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        # print('mask_tokens shape: ',mask_tokens.shape)
        x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # add pos embed
        x = x_ + self.decoder_pos_embed(x_)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        return x

