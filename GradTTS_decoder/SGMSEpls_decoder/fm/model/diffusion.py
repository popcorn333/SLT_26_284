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
from sgmse.backbones.ncsnpp_v2_te import NCSNpp_v2_te as GradLogPEstimator2d 
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
        self.estimator = GradLogPEstimator2d()


    def xt_compute(self, x0, mask, noise, t):
        xt = (1 - (1 - 1e-4) * t) * noise + t * x0
        return xt * mask

    def solve_euler(self, xt, t_span, mu, mask, spks, cond):
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []
        sol.append(xt)
        t = t.to(dtype=mu.dtype, device=mu.device).expand(mu.shape[0])
        for step in range(1, len(t_span)):
            dphi_dt = self.estimator(xt.unsqueeze(1), mu.unsqueeze(1), t)

            xt = xt + dt * dphi_dt
            t = t + dt
            sol.append(xt)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1]

    @torch.no_grad()
    def reverse_diffusion(self, z, mask, mu, n_timesteps, stoc=False, spk=None):
        z = torch.randn_like(mu) * self.a
        t_span = torch.linspace(1e-4, 1, n_timesteps + 1, device=mu.device)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=None, cond=None)

    @torch.no_grad()
    def forward(self, z, mask, mu, n_timesteps, stoc=False, spk=None):
        return self.reverse_diffusion(z, mask, mu, n_timesteps, stoc, spk)

    def loss_t(self, X0, mask, mu, t, spk=None):
        device = X0.device
        dropout_rate=self.d
        n_timesteps = self.n_timesteps
        noise = torch.randn_like(X0) * self.a
        xt = self.xt_compute(X0, mask, noise, t.unsqueeze(-1).unsqueeze(-1))

        vf_est = self.estimator(xt.unsqueeze(1), mu.unsqueeze(1), t)    #Despite input N, it actually predicts M(n_steps*(N-1))
        
        vf = X0 - (1 - 1e-4) * noise
        loss = torch.sum((vf - vf_est)**2) / (torch.sum(mask)*self.n_feats)
        return loss, vf

    def compute_loss(self, x0, mask, mu, spk=None, offset=1e-5):
        """
        Args:
            t: (bs,) range from 0 to 1
        """

        t = torch.rand([mu.shape[0],], device=mu.device, dtype=mu.dtype)

        return self.loss_t(x0, mask, mu, t, spk)

