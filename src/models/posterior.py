#!/usr/bin/env python3
from typing import Callable, NamedTuple

import torch.distributions as td
from torchtyping import Array, Float
from src.types import Data, InputData, Prediction


class Posterior(pl.LightningModule):
    def __init__(self, prior: Prior, likelihood: Likelihood):
        super().__init__()
        self.prior = prior
        self.likelihood = likelihood

    def predict(x: InputData) -> Prediction:
        raise NotImplementedError

    def loss_fn(batch: Data) -> Float[Array, "1"]:
        raise NotImplementedError


class Likelihood(nn.Module):
    def __call__(x: InputData, f) -> td.Distribution:
        raise NotImplementedError

    def log_prob(x: InputData) -> Float[Array, "1"]:
        raise NotImplementedError


class Gaussian(Likelihood):
    def __init__(self, noise_network: nn.Module):
        super().__init__()
        self.noise_network = noise_network

    def __call__(self, x: InputData, f) -> td.Distribution:
        y = self.noise_network(x)
        mean = y[:, 0 : self.dim]
        var = y[:, self.dim :]
        return td.Normal(loc=mean, scale=torch.sqrt(var))

    def forward(self, x: InputData) -> td.Distribution:
        return self(x)

    def log_prob(self, x: InputData, y: OutputData) -> Float[Array, "1"]:
        return self(x).log_prob(y)
