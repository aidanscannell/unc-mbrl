#!/usr/bin/env python3
from typing import List, Optional

# import pytorch_lightning as pl
import torch
import torch.distributions as td
import torch.nn as nn
from functorch import vmap

from .networks import GaussianMLP
from pytorch_lightning import LightningModule


# from .models import DynamicsModel


# class ProbabilisticEnsemble(DynamicsModel):
class ProbabilisticEnsemble(LightningModule):
    def __init__(
        self,
        ensemble_size: int,
        in_size: int,
        out_size: int,
        features: List[int],
        activations: Optional[List] = None,
        learning_rate: Optional[float] = 1e-3,
    ):
        super(ProbabilisticEnsemble, self).__init__()
        self.learning_rate = learning_rate
        self.models = nn.ModuleList()
        for _ in range(ensemble_size):
            self.models.append(
                GaussianMLP(
                    in_size=in_size,
                    out_size=out_size,
                    features=features,
                    activations=activations,
                )
            )

    # # def forward(self, observation: Observation, action: Action) -> Prediction:
    def forward(self, x) -> td.MixtureSameFamily:
        def single_forward(model):
            y = model(x)
            out_size = int(y.shape[-1] / 2)
            mean = y[:, 0:out_size]
            var = y[:, out_size:]
            return mean, var

        means, vars = [], []
        # TODO make this run in parallel
        for model in self.models:
            mean, var = single_forward(model)
            means.append(mean)
            vars.append(var)
        means = torch.stack(means, -1)
        vars = torch.stack(vars, -1)
        # means, stddevs = vmap(single_forward)(self.models)

        mix = td.Categorical(torch.ones(self.ensemble_size) / self.ensemble_size)
        comp = td.Normal(loc=means, scale=torch.sqrt(vars))
        gmm = td.MixtureSameFamily(mix, comp)
        return gmm  # [N, output_dim, ensemble_size]

    def single_forward(self, x, ensemble_idx: int) -> td.Normal:
        dist = self.models[ensemble_idx](x)
        return dist

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self._nll_loss(x, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        val_loss = self._nll_loss(x, y), {}
        self.log("val_loss", val_loss)

    def _nll_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.ndim == y.ndim
        assert x.ndim >= 2

        def single_forward(model):
            y = model(x)
            out_size = int(y.shape[-1] / 2)
            mean = y[:, 0:out_size]
            var = y[:, out_size:]
            return mean, var

        nll = 0
        # TODO make this run in parallel
        for model in self.models:
            mean, var = single_forward(model)
            dist = td.Normal(loc=mean, scale=torch.sqrt(var))
            nll -= dist.log_prob(y)

        # if x.ndim == 2:  # add ensemble dimension
        #     x = x.unsqueeze(0)
        #     y = y.unsqueeze(0)
        # pred_mean, pred_logvar = self.forward(model_in, use_propagation=False)
        # if target.shape[0] != self.ensemble_size:
        #     target = target.repeat(self.ensemble_size, 1, 1)
        # pred_y_dist = td.Normal(loc=pred_mean, scale=pred_logvar)
        # nll = pred_y_dist.log_prob(target)

        # y_dist = self.forward(x)
        # print("y_dist")
        # print(y_dist.shape)
        # nll = -y_dist.log_prob(y)
        # TODO sum or mean?
        nll_sum = torch.sum(nll)
        return nll_sum

    @property
    def ensemble_size(self):
        return len(self.models)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


# class EnsembleDynamicsModel(DynamicsModel):
#     def __init__(self, models: List[nn.Module]):
#         super().__init__()
#         self.models = models

#     # def forward(self, observation: Observation, action: Action) -> Prediction:
#     def forward(self, x) -> Prediction:
#         dists = []
#         for model in self.models:
#             dists.append(model(x))
#         f_mean = torch.mean(fs)
#         f_var = torch.variance(fs)
#         # prediction = Prediction(latent=latent, noise=noise, output=output)

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         # if self.deterministic:
#         #     loss = self._mse_loss(x, y), {}
#         # else:
#         loss = self._nll_loss(x, y), {}
#         self.log("train_loss", loss)
#         return loss

#     def _mse_loss(self, model_in: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         assert model_in.ndim == target.ndim
#         if model_in.ndim == 2:  # add model dimension
#             model_in = model_in.unsqueeze(0)
#             target = target.unsqueeze(0)
#         pred_mean, _ = self.forward(model_in, use_propagation=False)
#         return F.mse_loss(pred_mean, target, reduction="none").sum((1, 2)).sum()

#     def _nll_loss(self, model_in: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         assert model_in.ndim == target.ndim
#         if model_in.ndim == 2:  # add ensemble dimension
#             model_in = model_in.unsqueeze(0)
#             target = target.unsqueeze(0)
#         pred_mean, pred_logvar = self.forward(model_in, use_propagation=False)
#         if target.shape[0] != self.num_members:
#             target = target.repeat(self.num_members, 1, 1)
#         pred_y_dist = td.Normal(loc=pred_mean, scale=pred_logvar)
#         nll = pred_y_dist.log_prob(target)
#         print("nll")
#         print(nll)
#         # nll = (
#         #     mbrl.util.math.gaussian_nll(pred_mean, pred_logvar, target, reduce=False)
#         #     .mean((1, 2))  # average over batch and target dimension
#         #     .sum()
#         # )  # sum over ensemble dimension
#         # nll += 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())
#         return nll

# def training_step(self, batch, batch_idx):
#     # training_step defines the train loop.
#     x, y = batch
#     x = x.view(x.size(0), -1)
#     fs = []
#     for model in self.models:
#         fs.append(model.forward(x))
#     f_mean = torch.mean(fs)
#     f_var = torch.variance(fs)
#     loss = nn.functional.mse_loss(y_hat, y)
#     # Logging to TensorBoard by default
#     self.log("train_loss", loss)
#     return loss


# class EnsemblePrediction:
#     dists: List[td.Distribution]
#     mean: torch.Tensor
#     var: torch.Tensor

#     def __init__(self, dists: List[td.Distribution]):
#         self.dists = dists
#         means, vars = [], []
#         for dist in self.dists:
#             means = dist.mean()
#             vars = dist.variance()
#         self.mean = torch.mean(means)
#         self.var = torch.variance(vars)
#         self.output = o
#         num_members = len(dists)
#         mix = D.Categorical(torch.ones(num_members))
#         # comp = D.Normal(torch.randn(5,), torch.rand(5,))
#         gmm = td.MixtureSameFamily(mix, comp)
