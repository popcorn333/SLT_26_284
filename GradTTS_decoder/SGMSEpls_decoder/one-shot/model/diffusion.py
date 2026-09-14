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
#from sgmse.backbones.ncsnpp_v2 import NCSNpp_v2 as GradLogPEstimator2d
from sgmse.backbones.ncsnpp_v2_wote import NCSNpp_v2_wote as GradLogPEstimator2d

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
        self.estimator = GradLogPEstimator2d()


    @torch.no_grad()
    def reverse_diffusion(self, z, mask, mu, n_timesteps, stoc=False, spk=None):
        '''
        Args:
            N: shape (bs,),
        '''

        N = torch.ones(mu.shape[0], device=mu.device, dtype=mu.dtype)
      
        x0_est = self.estimator(mu.unsqueeze(1), mu.unsqueeze(1), N)
        return x0_est


    @torch.no_grad()
    def forward(self, z, mask, mu, n_timesteps, stoc=False, spk=None):
        return self.reverse_diffusion(z, mask, mu, n_timesteps, stoc, spk)

    def loss_t(self, x0, mask, mu, n, spk=None):
        """
        Args:
            n: (bs,) range from 1 to n_timesteps
        """
        x0_est = self.estimator(mu.unsqueeze(1), mu.unsqueeze(1), n/self.n_timesteps)
        
        loss = torch.sum((x0_est - x0)**2) / (torch.sum(mask)*self.n_feats)
        return loss, x0_est

    def compute_loss(self, x0, mask, mu, spk=None, offset=1e-5):
        """
        Args:
            n: (bs,) range from 1 to n_timesteps
        """

        n = torch.randint(1, self.n_timesteps + 1, (x0.shape[0],), device=x0.device).to(x0.dtype)
        return self.loss_t(x0, mask, mu, n, spk)

