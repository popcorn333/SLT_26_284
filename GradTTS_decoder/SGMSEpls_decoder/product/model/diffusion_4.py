# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import math
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from einops import rearrange

from model.base import BaseModule
from model.utils import ReverseLayerF
from model.utils import Cutout
from model.utils import cutout_along_dimension

class Mish(BaseModule):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


class Upsample(BaseModule):
    def __init__(self, dim):
        super(Upsample, self).__init__()
        self.conv = torch.nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        x = torch.clamp(x, min=-1e3, max=1e3)
        return self.conv(x)


class Downsample(BaseModule):
    def __init__(self, dim):
        super(Downsample, self).__init__()
        self.conv = torch.nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        x = torch.clamp(x, min=-1e3, max=1e3)
        return self.conv(x)


class Rezero(BaseModule):
    def __init__(self, fn):
        super(Rezero, self).__init__()
        self.fn = fn
        self.g = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g


class Block(BaseModule):
    def __init__(self, dim, dim_out, groups=8):
        super(Block, self).__init__()
        self.block = torch.nn.Sequential(torch.nn.Conv2d(dim, dim_out, 3, 
                                         padding=1), torch.nn.GroupNorm(
                                         groups, dim_out), Mish())

    def forward(self, x, mask):
        x = torch.clamp(x, min=-1e3, max=1e3)
        output = self.block(x * mask)
        return output * mask


class ResnetBlock(BaseModule):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super(ResnetBlock, self).__init__()
        self.mlp = torch.nn.Sequential(Mish(), torch.nn.Linear(time_emb_dim, 
                                                               dim_out))

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        if dim != dim_out:
            self.res_conv = torch.nn.Conv2d(dim, dim_out, 1)
        else:
            self.res_conv = torch.nn.Identity()

    def forward(self, x, mask, time_emb):
        x = torch.clamp(x, min=-1e3, max=1e3)
        h = self.block1(x, mask)
        h += self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output


class LinearAttention(BaseModule):
    def __init__(self, dim, heads=4, dim_head=32):
        super(LinearAttention, self).__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = torch.nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = torch.nn.Conv2d(hidden_dim, dim, 1)            

    def forward(self, x):
        x = torch.clamp(x, min=-1e3, max=1e3)
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', 
                            heads = self.heads, qkv=3)            


        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', 
                        heads=self.heads, h=h, w=w)
        return self.to_out(out)


class Residual(BaseModule):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        x = torch.clamp(x, min=-1e3, max=1e3)
        output = self.fn(x, *args, **kwargs) + x
        return output


class SinusoidalPosEmb(BaseModule):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class GradLogPEstimator2d(BaseModule):
    def __init__(self, dim, dim_mults=(1, 2, 4), groups=8,
                 n_spks=None, spk_emb_dim=64, n_feats=80, pe_scale=1000, dropout_rate=0.1):
        super(GradLogPEstimator2d, self).__init__()
        self.dim = dim
        self.dim_mults = dim_mults
        self.groups = groups
        self.n_spks = n_spks if not isinstance(n_spks, type(None)) else 1
        self.spk_emb_dim = spk_emb_dim
        self.pe_scale = pe_scale
        if n_spks > 1:
            self.spk_mlp = torch.nn.Sequential(torch.nn.Linear(spk_emb_dim, spk_emb_dim * 4), Mish(),
                                               torch.nn.Linear(spk_emb_dim * 4, n_feats))
        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim, dim * 4), Mish(),
                                       torch.nn.Linear(dim * 4, dim))

        dims = [2 + (1 if n_spks > 1 else 0), *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(torch.nn.ModuleList([
                       ResnetBlock(dim_in, dim_out, time_emb_dim=dim),
                       torch.nn.Dropout(dropout_rate),
                       ResnetBlock(dim_out, dim_out, time_emb_dim=dim),
                       torch.nn.Dropout(dropout_rate),
                       Residual(Rezero(LinearAttention(dim_out))),
                       Downsample(dim_out) if not is_last else torch.nn.Identity()]))

        mid_dim = dims[-1]

        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)
        self.mid_block1_dropout = torch.nn.Dropout(dropout_rate)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)
        self.mid_block2_dropout = torch.nn.Dropout(dropout_rate)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(torch.nn.ModuleList([
                     ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim),
                     torch.nn.Dropout(dropout_rate),
                     ResnetBlock(dim_in, dim_in, time_emb_dim=dim),
                     torch.nn.Dropout(dropout_rate),
                     Residual(Rezero(LinearAttention(dim_in))),
                     Upsample(dim_in)
                     ]))
        self.final_block = Block(dim, dim)
        self.final_block_dropout = torch.nn.Dropout(dropout_rate)
        

        self.final_conv = torch.nn.Conv2d(dim, 1, 1)

        self.Xn=None

    def forward(self, x, mask, mu, t, spk=None):
        if not isinstance(spk, type(None)):
            s = self.spk_mlp(spk)
        t = self.time_pos_emb(t, scale=self.pe_scale)
        t = self.mlp(t)
        if self.n_spks < 2:
            x = torch.stack([mu, x], 1)
        else:
            s = s.unsqueeze(-1).repeat(1, 1, x.shape[-1])
            x = torch.stack([mu, x, s], 1)
        mask = mask.unsqueeze(1)
        hiddens = []
        masks = [mask]
        for resnet1, dropout1, resnet2, dropout2, attn, downsample in self.downs:
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t)
            x = dropout1(x)
            x = resnet2(x, mask_down, t)
            x = dropout2(x)
            x = attn(x)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, :, ::2])
        masks = masks[:-1]
        mask_mid = masks[-1]
        
        x = self.mid_block1(x, mask_mid, t)
        x = self.mid_block1_dropout(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x, mask_mid, t)
        x = self.mid_block2_dropout(x)

        for resnet1, dropout1, resnet2, dropout2, attn, upsample in self.ups:
            mask_up = masks.pop()
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, mask_up, t)
            x = dropout1(x)
            x = resnet2(x, mask_up, t)
            x = dropout2(x)
            x = attn(x)
            x = upsample(x * mask_up)
        x = self.final_block(x, mask)
        x= self.final_block_dropout(x)
        output = self.final_conv(x * mask)
        return (output * mask).squeeze(1)


class Forward_Diffusion(BaseModule):
    def __init__(self, cfg):
        super(Forward_Diffusion, self).__init__()
        
        mean = torch.ones(10).unsqueeze(1).expand(10, 80)
        full_std = torch.load(cfg.model.Masking.No1_std).view(10, 1, 1) * torch.eye(80)#The full std matrix (diagonal)
        lower_indices = torch.tril_indices(80, 80)  # Get lower triangular indices
        lower_tri_std = full_std[:, lower_indices[0], lower_indices[1]] #get the convert lower triangular part to a 1D matrix in each step
        zero_mask = lower_tri_std == 0
        lower_tri_std[zero_mask] = 0.001

        mean = torch.log(mean)
        std = torch.log(lower_tri_std)
        self.mean = torch.nn.Parameter(mean)
        self.std = torch.nn.Parameter(std)
        
        self.mean_init = torch.zeros(80)
        self.cov_init = torch.eye(80)
        
    
    def forward(self, X0, mask, mu, n):
        device = X0.device
        X_shape = X0[0].shape
        Xn = torch.ones(X0.shape).to(device) 
        terms = torch.ones(X0.shape).to(device)  # Creating the terms tensor on GPU
        std = torch.log( torch.ones(10, 80, 80, device=X0.device)*0.001 )
        lower_indices = torch.tril_indices(80, 80)
        std[:, lower_indices[0], lower_indices[1]] = self.std
        mvn = MultivariateNormal(self.mean_init.to(device), self.cov_init.to(device))
                
        for i, num in enumerate(n):
            X_itm = X0[i]
            for j in range(num - 1):
                eps = mvn.sample((X_shape[1],)).to(device).T
                noise = torch.exp(std[j])@eps + torch.exp(self.mean[j]).unsqueeze(1) #sample_from_a_distribution_with_covarianc_Cov)= \mu + M*(sample_from_a_standard distribution
                X_itm = noise * X_itm
            eps = mvn.sample((X_shape[1],)).T
            term = torch.exp(std[num-1])@eps   + torch.exp(self.mean[num-1]).unsqueeze(1)
            X_itm = term * X_itm
            terms[i] = term
            Xn[i] = X_itm
        
        return Xn* mask, terms

class ReversedForward_Diffusion(nn.Module):
    def __init__(self, cfg):
        super(ReversedForward_Diffusion, self).__init__()
        self.Forward_Diffusion = Forward_Diffusion(cfg)  # Assuming UNet is your model
        self.alpha = cfg.model.Masking.alpha  # Scaling factor for gradient reversal

    def forward(self, x, mask, mu, n):
        # Get the output of Forward_Diffusion
        Xn, terms = self.Forward_Diffusion(x, mask, mu, n)
        
        alpha = torch.tensor([self.alpha]).cuda() 
        # Apply gradient reversal
        Xn = ReverseLayerF.apply(Xn, alpha)
        terms = ReverseLayerF.apply(terms, alpha)
        return Xn, terms


class Diffusion(BaseModule):
    def __init__(self, cfg):
        super(Diffusion, self).__init__()
        self.n_spks = cfg.data.n_spks
        self.spk_emb_dim = cfg.model.spk_emb_dim
        self.n_feats = cfg.data.n_feats
        
        self.dim = cfg.model.decoder.dim
        self.pe_scale = cfg.model.decoder.pe_scale
        
        self.n_timesteps = cfg.training.n_timesteps
        self.forward_pass = ReversedForward_Diffusion(cfg)
        cfg = cfg.model.Masking
        self.c = cfg.c
        self.d = cfg.d
        self.l = cfg.l
        self.p = cfg.p
        self.No1_std = cfg.No1_std       
        self.estimator = GradLogPEstimator2d(self.dim, n_spks=self.n_spks,
                                             spk_emb_dim=self.spk_emb_dim,
                                             pe_scale=self.pe_scale, dropout_rate = self.d)
        
        mean = torch.ones(10).unsqueeze(1).expand(10, 80)

        full_std = torch.load(cfg.No1_std).view(10, 1, 1) * torch.eye(80)#The full std matrix (diagonal)
        lower_indices = torch.tril_indices(80, 80)  # Get lower triangular indices
        lower_tri_std = full_std[:, lower_indices[0], lower_indices[1]] #get the lower triangular part fo the full std matrix
        zero_mask = lower_tri_std == 0
        lower_tri_std[zero_mask] = 0.001
        

        self.mean_i = mean
        self.std_i = lower_tri_std
    
    @torch.no_grad()
    def reverse_diffusion(self, z, mask, mu, n_timesteps, stoc=False, spk=None):
        X0 = z * mask
        Xn = z * mask
        for n in range(n_timesteps):
            n = n_timesteps - n 
            n = n * torch.ones(z.shape[0], dtype=z.dtype, device=z.device)
            n = n/n_timesteps
            O_o_Mask  = self.estimator(Xn, mask, mu, n, spk)  #input N, Unet actually predicts M(N-1) 
            if torch.isnan(O_o_Mask).any():
                print("Warning: The Mask contains NaN")
            Xn = Xn*O_o_Mask
        return Xn

    @torch.no_grad()
    def forward(self, z, mask, mu, n_timesteps, stoc=False, spk=None):
        return self.reverse_diffusion(z, mask, mu, n_timesteps, stoc, spk)

    def loss_t(self, X0, mask, mu, n, spk=None):
        device = X0.device
        dropout_rate=self.d
        
        Mask = Cutout(n_holes=int(X0.shape[2]/80), length=self.c)
        Mask = Mask(X0.shape).to(device)
        X0 = X0*Mask
        X0 = cutout_along_dimension(X0, l=self.l, cutout_percentage=self.p)
        
        Xn, Masks = self.forward_pass(X0, mask, mu, n)

        dropout = torch.nn.Dropout(p=dropout_rate)
        mu_dropout = dropout(mu)
        
        n = (n/self.n_timesteps).float()
        noise_estimation = self.estimator(Xn, mask, mu_dropout, n, spk)   #Despite input N, it actually predicts M(n_steps*(N-1))
        loss = torch.sum((noise_estimation - 1/Masks)**2) / (torch.sum(mask)*self.n_feats)
        
        return loss, Xn

    def compute_loss(self, x0, mask, mu, spk=None, offset=1e-5):
        n_timesteps = self.n_timesteps
        n = torch.randint(1, int(n_timesteps+1), size = (x0.shape[0],), dtype=int, device=x0.device,
                       requires_grad=False)
        return self.loss_t(x0, mask, mu, n, spk)

