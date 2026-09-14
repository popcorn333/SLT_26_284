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
        self.estimator = GradLogPEstimator2d(self.dim, n_spks=self.n_spks,
                                             spk_emb_dim=self.spk_emb_dim,
                                             pe_scale = self.pe_scale)
    def log_scale_t(self, n, a, b, n_timesteps):
        term = ((n_timesteps - n) / (n_timesteps - 1)) * math.log(a) + ((n - 1) / (n_timesteps - 1)) * math.log(b)
    
        n_log = 0.5 * torch.exp(2 * term)
        return n_log  

    def forward_diffusion(self, X0, mask, mu, n, n_steps):
        device = X0.device
        H = mu.shape[2]
        W = mu.shape[1]

        n = n.view(-1, 1, 1)

        n_log = self.log_scale_t(n, self.a, self.b, n_steps)

        freqs_x = np.pi * np.linspace(0, W - 1, W) / W
        freqs_y = np.pi * np.linspace(0, H - 1, H) / H
        Λ = -(freqs_x[:, None]**2 + freqs_y[None, :]**2)
        X0_np = X0.detach().cpu().numpy().astype(np.float64)
        X0_proj = dct(X0_np, axis=1, norm='ortho')
        X0_proj = dct(X0_proj, axis=2, norm='ortho')
        X0_proj = np.exp(Λ * n_log.detach().cpu().numpy()) * X0_proj

        X0_recons = idct(X0_proj, axis=1, norm='ortho')
        X0_recons = idct(X0_recons, axis=2, norm='ortho')

        X0_recons = torch.from_numpy(X0_recons).to(X0.device).float()

        Xn = X0_recons
        Xn = torch.where(n == 0, X0, Xn)

        return Xn * mask, n_steps


    @torch.no_grad()
    def reverse_diffusion(self, z, mask, mu, n_timesteps, stoc=False, spk=None):

        xt, _ = self.forward_diffusion(mu, mask, mu, n_timesteps * torch.ones(z.shape[0], dtype=z.dtype, device=z.device), n_timesteps)
        xt = xt * mask
        sol = []
        sol.append(xt)

        for t in range(n_timesteps):
            t = n_timesteps - t #[n_timesteps, n_timesteps-1, ... ,1]
            t = t * torch.ones(z.shape[0], dtype=z.dtype, device=z.device)

            x0_est = self.estimator(xt, mask, mu, t/n_timesteps)  #input N, Unet actually predicts M(N-1)
            Dt, _ = self.forward_diffusion(x0_est, mask, mu, t, n_timesteps)
            Dtm1, _ = self.forward_diffusion(x0_est, mask, mu, t-1, n_timesteps)
            xt = xt - Dt + Dtm1
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

