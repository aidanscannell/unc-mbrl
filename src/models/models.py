#!/usr/bin/env python3
from typing import Callable, NamedTuple, Optional
import torch.optim as optim
import torch
import pytorch_lightning as pl
from src.types import ReplayBuffer


class Prediction(NamedTuple):
    latent: torch.distributions.Distribution  # p(f_{\theta}(x) \mid x, \mathcal{D})
    noise: torch.distributions.Distribution  # p(y \mid f_{\theta}(x))
    output: torch.distributions.Distribution  # p(y \mid x, \mathcal{D})


class DynamicsModel(pl.LightningModule):
    def forward(self) -> Prediction:
        raise NotImplementedError

    def train(self, replay_buffer: ReplayBuffer):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
