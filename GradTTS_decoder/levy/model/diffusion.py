# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import math
import torch
import numpy as np
from scipy.fftpack import dct, idct
from torch.distributions.multivariate_normal import MultivariateNormal
from einops import rearrange
import matplotlib.pyplot as plot
import os
from model.base import BaseModule
from model.Unet_wote import GradLogPEstimator2d
from model.levy_forward import *

def pt_to_pdf(pt, pdf, vmin=-12.5, vmax=0.0):
    spec = pt
    fig = plot.figure(figsize=(20, 4), tight_layout=True)
    subfig = fig.add_subplot()
    image = subfig.imshow(
        spec,
        cmap="viridis",   # matches the screenshot
        origin="lower",
        aspect="equal",
        interpolation="none",
        vmax=vmax,
        vmin=vmin
    )
    fig.colorbar(mappable=image, orientation='vertical', ax=subfig, shrink=0.5)
    plot.savefig(pdf, format="pdf")
    plot.close()


class Diffusion(BaseModule):
    def __init__(self, cfg):
        super(Diffusion, self).__init__()
        self.n_spks = cfg.data.n_spks
        self.spk_emb_dim = cfg.model.spk_emb_dim
        self.n_feats = cfg.data.n_feats
        
        self.dim = cfg.model.decoder.dim
        self.pe_scale = cfg.model.decoder.pe_scale
        
        self.n_timesteps = cfg.training.n_timesteps
        cfg = cfg.model.masking
        self.a = cfg.a
        self.b = cfg.b
        self.c = cfg.c
        self.d = cfg.d
        self.alpha = cfg.alpha
        self.estimator = GradLogPEstimator2d(self.dim, n_spks=self.n_spks,
                                             spk_emb_dim=self.spk_emb_dim,
                                             pe_scale = self.pe_scale)

    def forward_diffusion(self, X0, mask, mu, n, n_steps):
        device = X0.device
        Kernel = levy_forward_mel(X0, mask, n/n_steps, self.alpha, 0.008)
        Xn = Kernel
        n = n.view(-1, 1, 1)
        Xn = torch.where(n == 0, X0, Xn)
        return Xn * mask, n_steps

    @torch.no_grad()
    def reverse_diffusion(self, z, mask, mu, n_timesteps, stoc=False, spk=None):
        xt, _ = self.forward_diffusion(mu, mask, mu, torch.ones(z.shape[0], device=z.device) * n_timesteps, n_timesteps)
        xt = xt * mask
        sol = []
        sol.append(xt)
        for n in range(n_timesteps, 0, -1):  # n_timesteps, ..., 1
            n = torch.full((z.shape[0],), n, dtype=z.dtype, device=z.device)        
            x0_est = self.estimator(xt, mask, mu, n/n_timesteps)  #input N, Unet actually predicts M(N-1)
            xt, _ = self.forward_diffusion(x0_est, mask, mu, n-1, n_timesteps)
            sol.append(xt)
        return sol[-1]

    @torch.no_grad()
    def forward(self, z, mask, mu, n_timesteps, stoc=False, spk=None):
        return self.reverse_diffusion(z, mask, mu, n_timesteps, stoc, spk)

    def loss_t(self, x0, mask, mu, n, spk=None):
        """
        Args:
            n: (bs,) range from 1 to n_timesteps
        """

        xt, z = self.forward_diffusion(x0, mask, mu, n, self.n_timesteps)
        
        x0_est = self.estimator(xt, mask, mu, n/self.n_timesteps)
        
        loss = torch.sum((x0_est - x0)**2) / (torch.sum(mask)*self.n_feats)
        return loss, xt

    def compute_loss(self, x0, mask, mu, spk=None, offset=1e-5):
        """
        Args:
            n: (bs,) range from 1 to n_timesteps
        """

        n = torch.randint(1, self.n_timesteps + 1, (x0.shape[0],), device=x0.device).to(x0.dtype)
        return self.loss_t(x0, mask, mu, n, spk)

