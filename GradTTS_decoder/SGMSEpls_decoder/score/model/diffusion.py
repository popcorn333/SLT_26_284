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
#from sgmse.backbones.ncsnpp_v2_wote import NCSNpp_v2_wote as GradLogPEstimator2d
from sgmse.backbones.ncsnpp_v2_te import NCSNpp_v2_te as GradLogPEstimator2d

def get_noise(t, beta_init, beta_term, cumulative=False):
    if cumulative:
        noise = beta_init*t + 0.5*(beta_term - beta_init)*(t**2)
    else:
        noise = beta_init + (beta_term - beta_init)*t
    return noise

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

        self.beta_min = 0.05
        self.beta_max = 20

    def forward_diffusion(self, x0, mask, mu, t):
        prior_std = self.a
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        mean = x0*torch.exp(-0.5*cum_noise) + mu*(1.0 - torch.exp(-0.5*cum_noise))
        base_variance = 1.0 - torch.exp(-cum_noise)
        variance = prior_std ** 2 * base_variance

        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device,
                        requires_grad=False)
        xt = mean + z * torch.sqrt(variance)
        return xt * mask, z * mask

    @torch.no_grad()
    def reverse_diffusion(self, z, mask, mu, n_timesteps, stoc=False, spk=None):
        prior_std = self.a
        z = mu + torch.randn_like(mu, device=mu.device) * prior_std
        h = 1.0 / n_timesteps
        xt = z * mask
        for i in range(n_timesteps):
            t = (1.0 - (i)*h) * torch.ones(z.shape[0], dtype=z.dtype,
                                                 device=z.device)
            time = t.unsqueeze(-1).unsqueeze(-1)
            noise_t = get_noise(time, self.beta_min, self.beta_max,
                                cumulative=False)

            dxt = 0.5 * (mu - xt) - 0.5 * self.a**2 * self.estimator(xt.unsqueeze(1), mu.unsqueeze(1), t)
            dxt = dxt * noise_t * h

            xt = (xt - dxt) * mask
        return xt

    @torch.no_grad()
    def forward(self, z, mask, mu, n_timesteps, stoc=False, spk=None):
        return self.reverse_diffusion(z, mask, mu, n_timesteps, stoc, spk)


    def loss_t(self, x0, mask, mu, t, spk=None):
        """
        Args:
            t: (bs,) range from 0 to 1
        """
        prior_std = self.a
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        xt, z = self.forward_diffusion(x0, mask, mu, t)
        
        
        variance = prior_std ** 2 * (1.0 - torch.exp(-cum_noise))
        noise_estimation = self.estimator(xt.unsqueeze(1), mu.unsqueeze(1), t)
        noise_estimation *= torch.sqrt(variance)
        loss = torch.sum((noise_estimation + z)**2 * mask) / (torch.sum(mask)*self.n_feats)

        return loss, xt

    def compute_loss(self, x0, mask, mu, spk=None, offset=1e-5):
        """
        Args:
            t: (bs,) range from 0 to 1
        """

        t = torch.rand(mu.shape[0], device=mu.device, dtype=mu.dtype)
        return self.loss_t(x0, mask, mu, t, spk)


