#!/usr/bin/env python3
import random
from collections import deque, namedtuple
from typing import Any, Iterator, Optional, Tuple

import numpy as np
import torch
from torch.utils.data.dataset import IterableDataset
from torchtyping import TensorType

Tensor = Any
Observation = Tensor
Action = Tensor


Transition = namedtuple(
    "Transition",
    ("observation", "action", "next_observation", "reward", "terminated", "truncated"),
)
TransitionBatch = namedtuple(
    "Transition",
    (
        "observations",
        "actions",
        "next_observations",
        "rewards",
        "terminateds",
        "truncateds",
    ),
)


class ReplayBuffer(object):
    def __init__(self, capacity: int):
        self.buffer = deque([], maxlen=capacity)

    def push(
        self, observation, action, next_observation, reward, terminated, truncated
    ):
        """Save a transition"""
        self.buffer.append(
            Transition(
                observation=torch.as_tensor(observation, dtype=torch.float32),
                action=torch.as_tensor(action, dtype=torch.float32),
                next_observation=torch.as_tensor(next_observation, dtype=torch.float32),
                reward=torch.as_tensor(reward, dtype=torch.float32),
                # TODO: are terminated and truncated always 1 or 0?
                terminated=torch.as_tensor(terminated, dtype=torch.bool),
                truncated=torch.as_tensor(truncated, dtype=torch.bool),
            )
        )
        # self.memory.append(Transition(*args))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        (
            observations,
            actions,
            next_observations,
            rewards,
            terminateds,
            truncateds,
        ) = zip(*(self.buffer[idx] for idx in indices))

        return TransitionBatch(
            # np.array(observations, dtype=np.float32),
            # np.array(actions, dtype=np.float32),
            # np.array(next_observations, dtype=np.float32),
            # np.array(rewards, dtype=np.float32),
            # np.array(terminateds, dtype=bool),
            # np.array(truncateds, dtype=bool),
            observations=torch.stack(observations, 0),
            actions=torch.stack(actions, 0),
            next_observations=torch.stack(next_observations, 0),
            rewards=torch.stack(rewards, 0),
            terminateds=torch.stack(terminateds, 0),
            truncateds=torch.stack(truncateds, 0),
            # np.array(states),
            # np.array(actions),
            # np.array(rewards, dtype=np.float32),
            # np.array(dones, dtype=bool),
            # np.array(next_states),
        )
        # return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.buffer)


class RLDataset(IterableDataset):
    """Iterable Dataset containing the ExperienceBuffer which will be updated with new experiences during training.

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, buffer: ReplayBuffer, sample_size: int = 200) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Iterator[Tuple]:
        # samples = self.buffer.sample(self.sample_size)
        (
            observations,
            actions,
            next_observations,
            rewards,
            terminateds,
            truncateds,
        ) = self.buffer.sample(self.sample_size)
        for i in range(len(terminateds)):
            # yield states[i], actions[i], rewards[i], dones[i], new_states[i]
            yield observations[i], actions[i], next_observations[i], rewards[
                i
            ], terminateds[i], truncateds[i],

        # yield samples
        # for sample in samples:
        #     yield sample


class DynamicsModelDataset(IterableDataset):
    """Iterable Dataset containing the ExperienceBuffer which will be updated with new experiences during training.

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(
        self,
        buffer: ReplayBuffer,
        batch_size: int = 64,
        delta_state: Optional[bool] = True,
    ) -> None:
        self.buffer = buffer
        self.batch_size = batch_size
        self.delta_state = delta_state

    @property
    def dataset(self) -> Tuple[TensorType["N, D"], TensorType["N, P"]]:
        # print("insiode dataset")
        # print(len(self.buffer))
        transitions = self.buffer.sample(len(self.buffer))
        # print("transitions.obs")
        # print(transitions.observations.shape)
        # print("transitions.next_obs")
        # print(transitions.next_observations.shape)
        observation_action_inputs = torch.concatenate(
            [transitions.observations, transitions.actions], -1
        )
        # print("observation_action_inputs")
        # print(observation_action_inputs.shape)
        if self.delta_state:
            delta_observations = (
                transitions.next_observations - transitions.observations
            )
            return (observation_action_inputs, delta_observations)
        else:
            return (observation_action_inputs, transitions.next_observations)

    def __iter__(self) -> Iterator[Tuple]:
        dataset = self.dataset
        dataset_size = len(self.buffer)
        while True:
            perm = torch.randperm(dataset_size)
            start = 0
            end = self.batch_size
            while end < dataset_size:
                batch_perm = perm[start:end]
                yield tuple(array[batch_perm] for array in dataset)
                start = end
                end = start + self.batch_size
