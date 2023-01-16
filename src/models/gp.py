#!/usr/bin/env python3
from typing import Callable

import gpytorch
import torch
import torch.distributions as td
from src.models.models import DynamicsModel
from src.types import (
    Action,
    Observation,
    Prediction,
    ReplayBuffer,
    ReplayBuffer_to_dynamics_TensorDataset,
)


class GPDynamicsModel(DynamicsModel):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        learning_rate: float = 0.1,
        num_iterations: int = 1000,
        delta_state: bool = True,
    ):
        super(GPDynamicsModel, self).__init__()
        self.num_iterations = num_iterations
        self.delta_state = delta_state

        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([out_size])
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                batch_shape=torch.Size([out_size]), ard_num_dims=in_size
            ),
            batch_shape=torch.Size([out_size]),
        )
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=out_size
        )
        self.model = ExactGPModel(
            train_x=torch.ones(10, in_size),  # dummy inputs
            train_y=torch.ones(10, out_size),  # dummy outputs
            # train_y=None,
            likelihood=self.likelihood,
            mean_module=self.mean_module,
            covar_module=self.covar_module,
        )

        # Use the adam optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # "Loss" for GPs - the marginal log likelihood
        self._mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood, self.model
        )

    def forward(self, x) -> Prediction:
        self.model.eval()
        self.likelihood.eval()
        latent = self.model(x)
        print("latent")
        print(latent)
        output = self.likelihood(latent)
        print("output")
        print(output)
        f_dist = td.Normal(loc=latent.mean, scale=torch.sqrt(latent.variance))
        print("f_dist")
        print(f_dist)
        y_dist = td.Normal(loc=latent.mean, scale=torch.sqrt(output.variance))
        print("y_dist")
        print(y_dist)
        noise_var = output.variance - latent.variance
        print("noise_var")
        print(noise_var)
        # pred = Prediction(latent=f_dist, output=y_dist, noise_var=noise_var)
        pred = Prediction(latent=f_dist, output=y_dist, noise=noise_var)
        print("pred")
        print(pred)
        return pred

    def predict(self, observation: Observation, action: Action) -> Prediction:
        x = torch.concat([observation, action], -1)
        self.model.eval()
        self.likelihood.eval()
        return self.forward(x)

    def train(self, replay_buffer: ReplayBuffer):
        dataset = ReplayBuffer_to_dynamics_TensorDataset(
            replay_buffer, delta_state=self.delta_state
        )
        train_x = dataset.tensors[0]
        train_y = dataset.tensors[1]
        self.model.set_train_data(inputs=train_x, targets=train_y, strict=False)

        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()
        # print("self.model")
        # print(self.model)
        # print(self.likelihood)
        # print(self.model.parameters())
        # self.optimizer = torch.optim.LBFGS(
        #     self.model.parameters(),
        #     lr=0.1,
        #     # lr=1,
        #     max_iter=20,
        #     max_eval=None,
        #     tolerance_grad=1e-07,
        #     tolerance_change=1e-09,
        #     history_size=100,
        #     line_search_fn=None,
        # )

        # def loss_closure():
        #     output = self.model(train_x)
        #     loss = -self._mll(output, train_y)
        #     print("loss: {}".format(loss))
        #     # print(self.model.parameters())
        #     loss.backward()
        #     return loss

        for i in range(self.num_iterations):
            self.optimizer.zero_grad()
            output = self.model(train_x)
            loss = -self._mll(output, train_y)
            loss.backward()
            print("Iter %d/%d - Loss: %.3f" % (i + 1, self.num_iterations, loss.item()))
            self.optimizer.step()
            # self.optimizer.step(loss_closure)
            # loss = loss_closure()
            # print("Iter %d/%d - Loss: %.3f" % (i + 1, self.num_iterations, loss.item()))

        self.model.eval()
        self.likelihood.eval()


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, mean_module, covar_module):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        # return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )
