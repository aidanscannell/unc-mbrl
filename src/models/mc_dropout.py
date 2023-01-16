#!/usr/bin/env python3
from typing import List, Optional

import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType

from src.models.networks import GaussianMLP
from src.types import Prediction


# def MC_dropout(act_vec, p=0.5, mask=True):
#     return F.dropout(act_vec, p=p, training=mask, inplace=True)


class MCDropout(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        features: List[int],
        activations: Optional[List] = None,
        dropout_probs: List[int] = None,
    ):
        super(MCDropout, self).__init__()

        self.out_size = out_size
        self.dropout_probs = dropout_probs
        self.network = GaussianMLP(
            in_size=in_size,
            out_size=out_size,
            features=features,
            activations=activations,
            dropout_probs=dropout_probs,
        )

    def forward(
        self, x: TensorType["N", "in_size"], num_samples: int = 5
    ) -> Prediction:
        ys = []
        for _ in range(num_samples):
            ys.append(self.forward(x))
        ys = torch.stack(ys, 0)  # [num_samples, num_data, 2*out_size]
        print("ys.shape")
        print(ys.shape)

        f_samples = ys[..., 0 : self.out_size]  # [num_samples, num_data, out_size]
        noise_var_samples = ys[
            ..., self.out_size :
        ]  # [num_samples, num_data, out_size]

        f_mean = torch.mean(f_samples, 0)  # [num_data, out_size]
        f_var = torch.var(f_samples, 0)  # [num_data, out_size]
        f_dist = td.Normal(loc=f_mean, scale=torch.sqrt(f_var))

        noise_var_mean = torch.mean(noise_var_samples, 0)  # variance over MC samples
        noise_var_var = torch.var(
            noise_var_samples, 0
        )  # epistemic uncertainty over noise_var
        noise_dist = td.Normal(loc=f_mean, scale=torch.sqrt(noise_var_mean))
        noise_var_dist = td.Normal(loc=noise_var_mean, scale=torch.sqrt(noise_var_var))
        # TODO should I use epistemic unc over noise var?
        y_dist = td.Normal(loc=f_mean, scale=torch.sqrt(noise_var_mean + f_var))

        return Prediction(latent=f_dist, noise=noise_dist, output=y_dist)
