#!/usr/bin/env python3
from typing import Optional

import gymnasium as gym
import numpy as np
from src.types import Observation


class Agent:
    def __call__(self, observation: Observation):
        return self.act(observation=observation)

    def act(self, observation: Observation):
        raise NotImplementedError

    def reset(self):
        """Resets any internal state of the agent."""
        pass


class RandomAgent(Agent):
    def __init__(self, action_space: gym.spaces.Space, random_seed: Optional[int] = 42):
        action_space.seed(random_seed)
        self.action_space = action_space

    def act(self, observation: Optional[Observation] = None) -> np.ndarray:
        return self.action_space.sample()
