#!/usr/bin/env python3
import logging
import random
from typing import List, Optional


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar

import gymnasium as gym
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributions as td
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from setuptools.dist import Optional

from src.models.models import DynamicsModel
from src.planning.agents import Agent, RandomAgent
from src.types import ReplayBuffer, RLDataset, DynamicsModelDataset


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
@hydra.main(config_path="../configs", config_name="main")
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
    wandb_logger = WandbLogger(name="Adam-32-0.001", project="pytorchlightning")
    log_dir = run.dir

    wandb_logger = WandbLogger(project=cfg.wandb.project, log_model="all")
    trainer = Trainer(logger=wandb_logger)
    # log gradients and model topology
    # wandb_logger.watch(model)

    # Configure environment
    env = hydra.utils.instantiate(cfg.env)
    env.reset(seed=cfg.experiment.random_seed)
    print("env")
    print(env.action_space)
    print(env.action_space.shape)
    print(env.observation_space.shape)

    print("cfg.input_dim")
    print(cfg.input_dim)
    print(cfg.output_dim)
    model = hydra.utils.instantiate(cfg.model)
    print("model")
    print(model)
    # Configure agent
    # agent = hydra.utils.instantiate(cfg.agent, env=env)
    # agent = hydra.utils.instantiate(cfg.agent)

    x = torch.ones((500, cfg.input_dim))
    print("x.shape")
    print(x.shape)
    dist = model.forward(x)
    print("dist")
    print(dist)
    print(dist.batch_shape)
    print(dist.mean.shape)
    print(dist.stddev.shape)

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

    # Initialize a trainer
    early_stopping = EarlyStopping(
        monitor="val_loss", mode="min", min_delta=0.00, patience=3, verbose=False
    )
    trainer = Trainer(callbacks=[early_stopping], max_epochs=cfg.training.max_epochs)
    # trainer = Trainer(
    #     accelerator="auto",
    #     devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    #     max_epochs=3,
    #     callbacks=[TQDMProgressBar(refresh_rate=20)],
    # )

    # Train the model ⚡
    # train_loader = RLDataset(replay_buffer, sample_size=cfg.training.batch_size)
    train_loader = DynamicsModelDataset(
        replay_buffer, batch_size=cfg.training.batch_size, delta_state=True
    )
    print("train_loader.dataset")
    print(train_loader.dataset)
    print(train_loader.dataset[0].shape)
    print(train_loader.dataset[1].shape)
    trainer.fit(model, train_loader)
    # trainer.fit(model, train_loader, valid_loader)

    # Run the RL training loop
    # train(
    #     env=env, replay_buffer=replay_buffer, agent=agent, num_episodes=cfg.num_episodes
    # )


if __name__ == "__main__":
    run_experiment()  # pyright: ignore
