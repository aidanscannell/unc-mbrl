#!/usr/bin/env python3
from typing import List, Optional

import torch
import torch.distributions as td
import torch.nn as nn
from torchtyping import TensorType


class GaussianMLP(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        features: List[int],
        activations: Optional[List] = None,
    ):
        super(GaussianMLP, self).__init__()
        assert len(features) == len(activations)
        self.out_size = out_size
        if activations is None:
            activations = [nn.ReLU()] * len(features) - 1

        hidden_layers = [nn.Linear(in_size, features[0]), activations[0]]  # first layer
        for layer in range(1, len(features)):
            hidden_layers.append(nn.Linear(features[layer - 1], features[layer]))
            hidden_layers.append(activations[layer])

        hidden_layers.append(nn.Linear(features[-1], out_size * 2))  # final layer
        self.model = nn.Sequential(*hidden_layers)

    def forward(self, x: TensorType["N", "in_size"]) -> TensorType["N", "out_size*2"]:
        pred = self.model(x)  # [..., N, out_size*2]
        mean = pred[..., : self.out_size]
        log_var = pred[..., self.out_size :]
        var = torch.exp(log_var)
        return torch.concat([mean, var], -1)

    def dist(self, x: TensorType["N", "in_size"]) -> td.Normal:
        pred = self.forward(x)
        print("pred.shape")
        print(pred.shape)
        mean = pred[..., : self.out_size]
        var = pred[..., self.out_size :]
        print("mean.shape")
        print(mean.shape)
        print(var.shape)
        dist = td.Normal(loc=mean, scale=torch.sqrt(var))  # batch_shape = [N, out_size]
        return dist
