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
    def __init__(self, img_size=224, temp_dim=16,patch_size=(2,16,16), in_chans=3,
                 embed_dim=768, depth=6, num_heads=16,
                 decoder_embed_dim=384, decoder_depth=2, decoder_num_heads=16,
                 mlp_ratio=4.,dim_proj=1536, norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.embed_dim = embed_dim
        self.decoder_embed_dim=decoder_embed_dim
        self.patch_embed = PatchEmbed3D(img_size,temp_dim, patch_size, in_chans, embed_dim)
        self.encoder_pos_encoding=PositionalEncodingPermute3D(embed_dim)
        self.encoder_pos_embed=None
        num_patches = self.patch_embed.num_patches
 

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_encoding=PositionalEncodingPermute3D(decoder_embed_dim)

        self.decoder_pos_embed=None

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.decoder_pred = nn.Linear(decoder_embed_dim, dim_proj, bias=True)  # decoder to feature

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
    
    def pos_encoding(self,patch_embed,decoder_dim):
        # patch_embed [B self.embed_dim Dd Ww Hh]
        B,encoder_dim,Dd,Ww,Hh = patch_embed.shape 
        pe_encoder=self.encoder_pos_encoding(patch_embed.permute(0,3,4,2,1))
        pe_decoder=self.decoder_pos_encoding(torch.rand(B,decoder_dim,Dd,Ww,Hh).permute(0,3,4,2,1))

        encoder_pos_embed=pe_encoder.permute(0,4,3,1,2)  # [B self.embed_dim Dd Ww Hh]
        encoder_pos_embed=encoder_pos_embed.flatten(2).transpose(1,2)

        decoder_pos_embed=pe_decoder.permute(0,4,3,1,2) # [B self.decoder_dim Dd Ww Hh]
        decoder_pos_embed=decoder_pos_embed.flatten(2).transpose(1,2)# [B, self.embed_dim, Dd*Wh*Ww ]-> [B, N, self.embed_dim]

        return encoder_pos_embed,decoder_pos_embed
        
    def forward_encoder(self, x, mask_ratio):
        # embed patches
        # x [B,3,T,W,H]
        # B,C,T,W,H=x.shape
        patch_embedding = self.patch_embed(x) # -> [B self.embed_dim Dd Ww Hh]
        self.encoder_pos_embed,self.decoder_pos_embed=self.pos_encoding(patch_embedding,self.decoder_embed_dim)
        x=patch_embedding.flatten(2).transpose(1,2) # ->[B,N,D]
        x=x +self.encoder_pos_embed
        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # apply Transformer blocks
        for blk in self.blocks:
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
        x = x_ + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x

    def patchify(self, imgs):
        """
        imgs: (B, 3,16, H, W)
        x: (B, L, t*patch_size**2 *3)
        """
        print(imgs.shape)
        p = self.patch_embed.patch_size[1]
        t = self.patch_embed.patch_size[0]
        assert imgs.shape[3] == imgs.shape[4] and imgs.shape[3] % p == 0

        h = w = imgs.shape[3] // p
        n = imgs.shape[2] // t
        x = imgs.reshape(shape=(imgs.shape[0], 3, n, t, h, p, w, p))
        x = torch.einsum('bcnthpwq->bnhwpqtc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w * n , t * p**2 * 3))
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [B, 3, T, H, W]
        pred: [B, L, p*p*3]
        mask: [B, N], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
         
    def forward(self, imgs, mask_ratio=0.5):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, 1536]
        loss = self.forward_loss(imgs, pred, mask)
        return  loss, pred, mask


model=MaskedAutoencoderViT()
x=torch.rand(1,3,16,224,224)
loss, pred, mask=model(x)
print(loss.shape)
print(pred.shape)
print(mask.shape)