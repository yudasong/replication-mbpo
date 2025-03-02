from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from envs import BaseModelBasedEnv


class Walker2dEnv(mujoco_env.MujocoEnv, utils.EzPickle, BaseModelBasedEnv):

    def __init__(self, frame_skip=4):
        self.prev_qpos = None
        dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        mujoco_env.MujocoEnv.__init__(
            self, '%s/assets/walker2d.xml' % dir_path, frame_skip=frame_skip
        )
        utils.EzPickle.__init__(self)

    def step(self, action):
        old_ob = self.get_obs()
        self.do_simulation(action, self.frame_skip)
        ob = self.get_obs()

        if getattr(self, 'action_space', None):
            action = np.clip(action, self.action_space.low,
                             self.action_space.high)

        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = old_ob[8]
        reward_height = -3.0 * np.square(old_ob[0] - 1.3)

        height, ang = ob[0], ob[1]
        done = (height >= 2.0) or (height <= 0.8) or (abs(ang) >= 1.0)
        alive_reward = float(not done)

        reward = reward_run + reward_ctrl + reward_height + alive_reward
        return ob, reward, done, {}

    def get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        self.prev_qpos = np.copy(self.sim.data.qpos.flat)
        return self.get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20

    def cost_np_vec(self, obs, acts, next_obs):
        reward_ctrl = -0.1 * np.sum(np.square(acts), axis=1)
        reward_run = obs[:, 8]
        reward_height = -3.0 * np.square(next_obs[:, 0] - 1.3)
        height, ang = next_obs[:, 0], next_obs[:, 1]
        done = np.logical_or(
            np.logical_or(height >= 2.0, height <= 0.8),
            np.abs(ang) >= 1.0
        )
        alive_reward = 1.0 - np.array(done, dtype=np.float)
        reward = reward_run + reward_ctrl + reward_height + alive_reward
        return -reward

    def cost_tf_vec(self, obs, acts, next_obs):
        """
        reward_ctrl = -0.1 * tf.reduce_sum(tf.square(acts), axis=1)
        reward_run = next_obs[:, 0]
        # reward_height = -3.0 * tf.square(next_obs[:, 1] - 1.3)
        reward = reward_run + reward_ctrl
        return -reward
        """
        raise NotImplementedError

    def verify(self):
        pass

    def mb_step(self, states, actions, next_states):
        if getattr(self, 'action_space', None):
            actions = np.clip(actions, self.action_space.low,
                              self.action_space.high)
        rewards = - self.cost_np_vec(states, actions, next_states)
        height, ang = next_states[:, 0], next_states[:, 1]
        done = np.logical_or(
            np.logical_or(height >= 2.0, height <= 0.8),
            np.abs(ang) >= 1.0
        )
        return rewards, done
