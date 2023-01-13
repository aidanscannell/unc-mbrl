#!/usr/bin/env python3
import laplace
from typing import List
import torch
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn as nn
import laplace
from .models import DynamicsModel


class LaplaceDynamicsModel(DynamicsModel):
    # def __init__(self, laplace_approx: laplace.Laplace, optimizer: optim.Optimizer):
    def __init__(
        self,
        model: nn.Module,
        sigma_noise=torch.Tensor([1]),
        subset_of_weights="all",
        hessian_structure="diag",
    ):
        super().__init__()
        self.laplace_approx = laplace.Laplace(
            model,
            "regression",
            sigma_noise=sigma_noise,
            subset_of_weights=subset_of_weights,
            hessian_structure=hessian_structure,
        )

    def __call__(self, observation: Observation, action: Action) -> Prediction:
        x = torch.concat(observation, action)
        y = self.la(x, pred_type="glm")
        f_var = self.la.functional_variance(x)
        latent = torch.distributions.Normal(loc=f_mean, scale=torch.sqrt(f_var))
        return f_mean, f_var
        # prediction = Prediction(latent=latent, noise=noise, output=output)
        # return self.forward(x)

    def forward(self, observation: Observation, action: Action) -> Prediction:
        f_mean, f_var = self.la(obs_act_input, link_approx="probit")
        latent = torch.distributions.Normal(loc=f_mean, scale=torch.sqrt(f_var))
        prediction = Prediction(latent=latent, noise=noise, output=output)

    def training_step(self, train_batch, batch_idx: int):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        y_hat = self(x)
        # self.la.fit(train_loader)
        loss = nn.functional.mse_loss(y_hat, y)
        return loss

    def validation_step(self, val_batch, batch_idx):
        return

    def validation_end(self, outputs):
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class LaplaceDynamicsModel(DynamicsModel):
    def __init__(self, laplace_approx: laplace.Laplace, optimizer: optim.Optimizer):
        super().__init__()
        self.laplace_approx = laplace_approx
        self.optimizer = optimizer

    def __call__(self, observation: Observation, action: Action) -> Prediction:
        x = torch.concat(observation, action)
        return self.forward(x)

    def forward(self, observation: Observation, action: Action) -> Prediction:
        f_mean, f_var = self.la(obs_act_input, link_approx="probit")
        latent = torch.distributions.Normal(loc=f_mean, scale=torch.sqrt(f_var))
        prediction = Prediction(latent=latent, noise=noise, output=output)

    def training_step(self, train_batch, batch_idx: int):
        self.la.fit(train_loader)

    def validation_step(self, val_batch, batch_idx):
        return

    def validation_end(self, outputs):
        return
