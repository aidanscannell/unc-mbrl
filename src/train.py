#!/usr/bin/env python3
import logging
import random
from typing import List, Optional

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

import gymnasium as gym
import hydra
import laplace
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributions as td
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from setuptools.dist import Optional
from torch.utils.data import DataLoader

from src.models.laplace import LaplaceDynamicsModel, LaplaceTrainer
from src.models.models import DynamicsModel
from src.planning.agents import Agent, RandomAgent
from src.types import ReplayBuffer, ReplayBuffer_to_dynamics_DataLoader

# def plot_latent_function_variance(model, test_inputs):
#     # test_inputs = create_test_inputs(num_test=40000)
#     # test_states = test_inputs[:, 0:2]
#     fig = plt.figure()
#     gs = fig.add_gridspec(1, 1)
#     ax = gs.subplots()


# class Plotter(Callback):
#     def on_train_epoch_end(self, *args, **kwargs):

#         # if self.epoch_plot_freq %
#         if self.what == "epochs":
#             self.state["epochs"] += 1


def rollout_agent_and_populate_replay_buffer(
    env: gym.Env,
    agent: Agent,
    replay_buffer: ReplayBuffer,
    num_episodes: int,
    # rollout_horizon: Optional[int] = 1,
    rollout_horizon: Optional[int] = None,
) -> ReplayBuffer:
    logger.info(f"Collecting {num_episodes} episodes from env")

    observation, info = env.reset()
    for episode in range(num_episodes):
        terminated, truncated = False, False
        timestep = 0
        while not terminated or truncated:
            if rollout_horizon is not None:
                if timestep >= rollout_horizon:
                    break
            action = agent(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            replay_buffer.push(
                observation=observation,
                action=action,
                next_observation=next_observation,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
            )
            observation = next_observation

            timestep += 1

        observation, info = env.reset()

    return replay_buffer


def train(
    env: gym.Env,
    replay_buffer: ReplayBuffer,
    # agent: Agent,
    model: DynamicsModel,
    trainer,
    num_episodes: int,
):
    observation, info = env.reset()

    for episode in num_episodes:
        action = agent.act(observation)
        observation, reward, terminated, truncated, info = env.step(action)

        # Train the model ⚡
        trainer.fit(mnist_model, train_loader)

        if terminated or truncated:
            observation, info = env.reset()

        agent.train(replay_buffer)


# @hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
# def main(cfg: DictConfig) -> Optional[float]:
@hydra.main(version_base="1.3", config_path="../configs", config_name="main")
def run_experiment(cfg: DictConfig):
    # Make experiment reproducible
    torch.manual_seed(cfg.experiment.random_seed)
    # torch.cuda.manual_seed(cfg.random_seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(cfg.experiment.random_seed)
    random.seed(cfg.experiment.random_seed)
    pl.seed_everything(cfg.experiment.random_seed)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    # Fetching the device that will be used throughout this notebook
    device = (
        torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    )

    # Initialise WandB run
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        tags=cfg.wandb.tags,
        name=cfg.wandb.run_name,
    )
    # wandb_logger = WandbLogger(name="Adam-32-0.001", project="pytorchlightning")
    log_dir = run.dir

    # Configure environment
    env = hydra.utils.instantiate(cfg.env)
    env.reset(seed=cfg.experiment.random_seed)

    model = hydra.utils.instantiate(cfg.model)
    # network = torch.nn.Sequential(
    #     torch.nn.Linear(1, 50), torch.nn.Tanh(), torch.nn.Linear(50, 1)
    # )
    # network = hydra.utils.instantiate(cfg.model.model)
    # network.output_size = cfg.output_dim
    # model = LaplaceDynamicsModel(model=network)
    print("model")
    print(model)
    print(type(model))
    # Configure agent
    # agent = hydra.utils.instantiate(cfg.agent, env=env)
    # agent = hydra.utils.instantiate(cfg.agent)

    # Collect initial data set
    replay_buffer = rollout_agent_and_populate_replay_buffer(
        env=env,
        agent=RandomAgent(
            action_space=env.action_space, random_seed=cfg.experiment.random_seed
        ),
        replay_buffer=ReplayBuffer(capacity=cfg.dataset.replay_buffer_capacity),
        num_episodes=cfg.dataset.num_episodes,
    )
    # print("replay_buffer")
    # print(replay_buffer.memory)
    # print(len(replay_buffer.memory))

    model.train(replay_buffer)

    model.model.eval()
    model.likelihood.eval()

    x = torch.ones((500, cfg.input_dim))
    print("x.shape")
    print(x.shape)
    dist = model.forward(x)
    print("dist")
    print(dist[0].shape)
    print(dist[1].shape)
    # print(dist.batch_shape)
    # print(dist.mean.shape)
    # print(dist.stddev.shape)

    # Initialize a trainer
    early_stopping = EarlyStopping(
        monitor="train_loss",
        # monitor="val_loss",
        mode="min",
        min_delta=0.00,
        patience=50,
        # patience=3,
        # min_delta=0.0,
        # patience=500,
        verbose=False,
    )
    # wandb_logger = WandbLogger(project=cfg.wandb.project, log_model="all")
    # wandb_logger.watch(model)  # log gradients and model topology
    # trainer = Trainer(
    #     logger=wandb_logger,
    #     callbacks=[early_stopping],
    #     max_epochs=cfg.training.max_epochs,
    # )
    trainer = LaplaceTrainer()

    # trainer = Trainer(
    #     accelerator="auto",
    #     devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    #     max_epochs=3,
    #     callbacks=[TQDMProgressBar(refresh_rate=20)],
    # )

    # Train the model ⚡
    # train_loader = RLDataset(replay_buffer, sample_size=cfg.training.batch_size)
    train_loader = ReplayBuffer_to_dynamics_DataLoader(
        replay_buffer=replay_buffer,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        delta_state=True,
        num_workers=cfg.training.num_workers,
    )
    # train_loader = DynamicsModelDataset(
    #     replay_buffer, batch_size=cfg.training.batch_size, delta_state=True
    # )
    # dataset = torch.utils.data.TensorDataset(*train_loader.dataset)
    # train_loader=DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True)
    print("train_loader.dataset")
    # print(train_loader.dataset)
    # print(train_loader.dataset[0].shape)
    # print(train_loader.dataset[1].shape)
    trainer.fit(model, train_loader)
    # trainer.fit(model, train_loader, valid_loader)

    # la, model, margliks, losses = laplace.marglik_training(model, train_loader, likelihood='regression', hessian_structure='kron', backend=laplace.curvature.asdl.AsdlGGN, optimizer_cls=torch.optim.adam.Adam, optimizer_kwargs=None, scheduler_cls=None, scheduler_kwargs=None, n_epochs=300, lr_hyp=0.1, prior_structure='layerwise', n_epochs_burnin=0, n_hypersteps=10, marglik_frequency=1, prior_prec_init=1.0, sigma_noise_init=1.0, temperature=1.0)

    # Run the RL training loop
    # train(
    #     env=env, replay_buffer=replay_buffer, agent=agent, num_episodes=cfg.num_episodes
    # )


if __name__ == "__main__":
    run_experiment()  # pyright: ignore
