# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
import abc
import gym


class BaseBatchedEnv(gym.Env, abc.ABC):
    # thought about using `@property @abc.abstractmethod` here but we don't need explicit `@property` function here.
    n_envs: int

    @abc.abstractmethod
    def step(self, actions):
        pass

    def reset(self):
        return self.partial_reset(range(self.n_envs))

    @abc.abstractmethod
    def partial_reset(self, indices):
        pass

    def set_state(self, state):
        pass


class BaseModelBasedEnv(gym.Env, abc.ABC):
    @abc.abstractmethod
    def mb_step(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray):
        raise NotImplementedError

    def verify(self, n=2000, eps=1e-4):
        pass

    def seed(self, seed: int = None):
        pass

