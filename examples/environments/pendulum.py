import numpy as np


# NEEDS a different initialisation than the one in gym (change the reset() method),
# to (m_init, S_init), modifying the gym env

# Introduces subsampling with the parameter SUBS and modified rollout function
# Introduces priors for better conditioning of the GP model
# Uses restarts

class PendulumEnv:
    def __init__(self, random_init=True):
        import gym
        # we could avoid the following import as it is just there to check the availability of the
        # pygame library but gym raises a custom error, which is inconvenient as we also want to avoid a hard
        # dependency on gym. So this is the cleanest solution we could come up with to have the basic python error.
        import pygame
        from pygame import gfxdraw

        self.env = gym.make('Pendulum-v1').env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.random_init = random_init
        self._state = None

    @property
    def state(self):
        self._state = self.env._get_obs()
        return self._state

    @state.setter
    def state(self, s):
        self._state = s
        phi = np.arctan2(s[1:2], s[0:1])[0]
        phi = phi + 2 * np.pi if phi < 0 else phi  # in [0, 2 pi)
        self.env.state = np.array([phi, s[2]])

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        if self.random_init:
            high = 0.5 * np.array([np.pi, 2.])
            self.env.state = np.random.uniform(low=0, high=high)
            self.env.state[0] += -np.pi
        else:
            self.env.state = np.array([0., 0.])
            self.env.state[0] += -np.pi
        self.env.last_u = None
        return self.env._get_obs()

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
