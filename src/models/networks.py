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
        dropout_probs: List[int] = None,
    ):
        super(GaussianMLP, self).__init__()
        assert len(features) == len(activations)
        self.out_size = out_size
        if activations is None:
            activations = [nn.ReLU()] * len(features) - 1

        if dropout_probs is not None:
            hidden_layers = [
                nn.Dropout(p=dropout_probs[0]),
                nn.Linear(in_size, features[0]),
                activations[0],
            ]
        else:
            hidden_layers = [
                nn.Linear(in_size, features[0]),
                activations[0],
            ]
        for layer in range(1, len(features)):
            if dropout_probs is not None:
                print("layer={}".format(layer))
                # hidden_layers.append(nn.Dropout(p=dropout_probs[layer]))
                hidden_layers.append(nn.Dropout(p=dropout_probs[layer], inplace=True))
            hidden_layers.append(nn.Linear(features[layer - 1], features[layer]))
            hidden_layers.append(activations[layer])

        hidden_layers.append(nn.Linear(features[-1], out_size * 2))  # final layer
        self.model = nn.Sequential(*hidden_layers)

    def forward(self, x: TensorType["N", "in_size"]) -> TensorType["N", "out_size*2"]:
        pred = self.model(x)  # [..., N, out_size*2]
        print("pred.shape")
        print(pred.shape)
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


if __name__ == "__main__":
    features = [2, 3, 4, 5]
    activations = [nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU()]
    model = GaussianMLP(
        in_size=5,
        out_size=4,
        features=features,
        activations=activations,
        dropout_probs=[0, 0, 0, 0],
        # dropout_probs=[0.5, 0.5, 0.5, 0.5],
    )

    x = torch.ones(2, 5)
    print("x.shape")
    print(x.shape)

    y = model.forward(x)
    print("y.shape")
    print(y.shape)

    y_mean, y_var = model.forward_(x)
    print("y_mean.shape")
    print(y_mean.shape)
    print(y_var.shape)
    print(y_mean)
    print(y_var)
