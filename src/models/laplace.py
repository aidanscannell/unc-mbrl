#!/usr/bin/env python3
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from src.types import Action, Observation, Prediction
from torch import Tensor

import laplace

# from laplace.curvature import AsdlGGN
from laplace.curvature.backpack import BackPackGGN

from .models import DynamicsModel


class LaplaceTrainer:
    def __init__(
        self,
        likelihood="regression",
        hessian_structure="full",
        # backend=laplace.curvature.asdl.AsdlGGN,
        backend=BackPackGGN,
        # backend=curvature.asdl.AsdlGGN,
        # backend=AsdlGGN,
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs={},
        scheduler_cls=None,
        scheduler_kwargs={},
        n_epochs=300,
        lr_hyp=0.1,
        prior_structure="layerwise",
        n_epochs_burnin=0,
        n_hypersteps=10,
        marglik_frequency=1,
        prior_prec_init=1.0,
        sigma_noise_init=1.0,
        temperature=1.0,
    ):
        self.likelihood = likelihood
        self.hessian_structure = hessian_structure
        self.backend = backend
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_kwargs
        self.n_epochs = n_epochs
        self.lr_hyp = lr_hyp
        self.prior_structure = prior_structure
        self.n_epochs_burnin = n_epochs_burnin
        self.n_hypersteps = n_hypersteps
        self.marglik_frequency = marglik_frequency
        self.prior_prec_init = prior_prec_init
        self.sigma_noise_init = sigma_noise_init
        self.temperature = temperature

    def fit(self, model, train_loader):
        la, laplace_approx, margliks, losses = laplace.marglik_training(
            model.laplace_approx.model,
            train_loader,
            likelihood=self.likelihood,
            hessian_structure=self.hessian_structure,
            backend=self.backend,
            optimizer_cls=self.optimizer_cls,
            optimizer_kwargs=self.optimizer_kwargs,
            scheduler_cls=self.scheduler_cls,
            scheduler_kwargs=self.scheduler_kwargs,
            n_epochs=self.n_epochs,
            lr_hyp=self.lr_hyp,
            prior_structure=self.prior_structure,
            n_epochs_burnin=self.n_epochs_burnin,
            n_hypersteps=self.n_hypersteps,
            marglik_frequency=self.marglik_frequency,
            prior_prec_init=self.prior_prec_init,
            sigma_noise_init=self.sigma_noise_init,
            temperature=self.temperature,
        )
        model.laplace_approx = laplace_approx
        return model


# class LaplaceDynamicsModel(DynamicsModel):
class LaplaceDynamicsModel:
    # def __init__(self, laplace_approx: laplace.Laplace, optimizer: optim.Optimizer):
    def __init__(
        self,
        model: nn.Module,
        sigma_noise=torch.Tensor([1]),
        subset_of_weights="all",
        hessian_structure="full",
        # learning_rate: Optional[float] = 1e-3,
    ):
        # super().__init__()
        self.model = model
        self.sigma_noise = sigma_noise
        self.laplace_approx = laplace.Laplace(
            model,
            "regression",
            sigma_noise=sigma_noise,
            subset_of_weights=subset_of_weights,
            hessian_structure=hessian_structure,
        )

    # def __call__(self, observation: Observation, action: Action) -> Prediction:
    #     x = torch.concat(observation, action)
    #     y = self.laplace_approx.la(x, pred_type="glm")
    #     print("y")
    #     print(y)
    #     # f_var = self.laplace_approx.la.functional_variance(x)
    #     latent = torch.distributions.Normal(loc=f_mean, scale=torch.sqrt(f_var))
    #     return f_mean, f_var
    #     # prediction = Prediction(latent=latent, noise=noise, output=output)
    #     # return self.forward(x)

    # def forward(self, x: Tensor) -> Prediction:
    def forward(self, x: Tensor) -> Tensor:
        # y = self.la(x, pred_type="glm")
        # return y
        f_mean, f_var = self.laplace_approx(x)
        return f_mean, f_var
        # latent = torch.distributions.Normal(loc=f_mean, scale=torch.sqrt(f_var))
        # prediction = Prediction(latent=latent, noise=noise, output=output)

    # def training_step(self, train_batch, batch_idx: int):
    #     x, y = train_batch
    #     x = x.view(x.size(0), -1)
    #     y_hat = self(x)
    #     # self.la.fit(train_loader)
    #     loss = nn.functional.mse_loss(y_hat, y)
    #     return loss

    # def validation_step(self, val_batch, batch_idx):
    #     return

    # def validation_end(self, outputs):
    #     return

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    #     return optimizer


# class LaplaceDynamicsModel(DynamicsModel):
#     def __init__(self, laplace_approx: laplace.Laplace, optimizer: optim.Optimizer):
#         super().__init__()
#         self.laplace_approx = laplace_approx
#         self.optimizer = optimizer

#     def __call__(self, observation: Observation, action: Action) -> Prediction:
#         x = torch.concat(observation, action)
#         return self.forward(x)

#     def forward(self, observation: Observation, action: Action) -> Prediction:
#         f_mean, f_var = self.la(obs_act_input, link_approx="probit")
#         latent = torch.distributions.Normal(loc=f_mean, scale=torch.sqrt(f_var))
#         prediction = Prediction(latent=latent, noise=noise, output=output)

#     def training_step(self, train_batch, batch_idx: int):
#         self.la.fit(train_loader)

#     def validation_step(self, val_batch, batch_idx):
#         return

#     def validation_end(self, outputs):
#         return
