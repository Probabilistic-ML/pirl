import numpy as np


class InvPendulumEnv:
    def __init__(self):
        import gym
        # we could avoid the following import as it is just there to check the availability of the
        # pybulletgym library but gym raises a custom error, which is inconvenient as we also want to avoid a hard
        # dependency on gym. So this is the cleanest solution we could come up with to have the basic python error.
        import pybulletgym

        self.env = gym.make('InvertedPendulumSwingupPyBulletEnv-v0').env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self._state = None
        self.mean_init = np.array([0., 0., -1., 0., 0.])
        self.target = np.array([0., 0., 1., 0., 0.])
        self.robot = self.env.robot

    def _get_state(self):
        phi, dphi = self.env.robot.j1.current_position()
        x, dx = self.env.robot.slider.current_position()
        return np.array([phi, dphi, x, dx])

    @property
    def state(self):
        self._state = self._get_state()
        return np.array(
            [self._state[2], self._state[3], np.cos(self._state[0]), np.sin(self._state[0]), self._state[1]])

    @state.setter
    def state(self, s):
        phi = np.arctan2(s[3:4], s[2:3])[0]  # in (-pi, pi]
        phi = phi + 2 * np.pi if phi < 0 else phi  # in [0, 2 pi)
        self.env.robot.j1.set_state(phi, s[4])
        self.env.robot.j1.set_motor_torque(0)
        self.env.robot.slider.set_state(s[0], s[1])
        self._state = self._get_state()

    # obs = (x, vx, np.cos(theta), np.sin(theta), theta_dot)
    # -> target = (0, 0, 1, 0, 0)
    def step(self, action):
        ob, r, done, _ = self.env.step(action)
        self._state = self._get_state()
        done = False
        return ob, r, done, _

    def reset(self):
        ob = self.env.reset()
        self._state = self._get_state()
        return ob

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        self.env.close()

# https://github.com/openai/gym/blob/master/gym/envs/mujoco/inverted_pendulum.py

# The action space is a continuous `(action)` in `[-3, 3]`, where `action` represents
# the numerical force applied to the cart (with magnitude representing the amount of
# force and sign representing the direction)
# | Num | Action                    | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit      |
# |-----|---------------------------|-------------|-------------|----------------------------------|-------|-----------|
# | 0   | Force applied on the cart | -3          | 3           | slider                           | slide | Force (N) |

# The observation is a `ndarray` with shape `(4,)` where the elements correspond to the following:
# | Num | Observation           | Min                  | Max                | Name (in corresponding XML file) | Joint| Unit |
# |-----|-----------------------|----------------------|--------------------|----------------------|--------------------|--------------------|
# | 0   | position of the cart along the linear surface | -Inf                 | Inf                | slider | slide | position (m) |
# | 1   | vertical angle of the pole on the cart        | -Inf                 | Inf                | hinge | hinge | angle (rad) |
# | 2   | linear velocity of the cart                   | -Inf                 | Inf                | slider | slide | velocity (m/s) |
# | 3   | angular velocity of the pole on the cart      | -Inf                 | Inf                | hinge | hinge | anglular velocity (rad/s) |
# mean initial state for swing up (0, -np.pi, 0, 0)

# import gym
# import numpy as np

# DEFAULT_SIZE = 500

# class InvPendulumEnv():
# def __init__(self, weights):
# self.env = gym.make('InvertedPendulum-v2').env
# self.action_space = self.env.action_space
# self.observation_space = self.env.observation_space
# self._state = None
# self.mean_init = np.array([0., -np.pi, 0., 0.])
# self.weights = weights

# @property
# def state(self):
# return self._state

# @state.setter
# def state(self, s):
# self._state = s
# self.env.sim.data.qpos[:]  = s[:2]
# self.env.sim.data.qvel[:]  = s[2:]

# def step(self, action):
# ob, _, _, _ = self.env.step(action)
# done = False
# self.state = ob

# ob = ob.reshape(1, -1)
# W = np.diag(self.weights)
# r = np.exp(-ob @ W @ np.transpose(ob)/2) - 1
# r = r[0, 0]

# return self.state, r, done, {}

# def reset(self):
# self.state = np.random.normal(loc=0., scale=0.03, size=(4,)) + self.mean_init
# return self.state

# def render(self, mode="human",
# width=DEFAULT_SIZE,
# height=DEFAULT_SIZE,
# camera_id=None,
# camera_name=None):
# self.env.render(mode=mode,
# width=width,
# height=height,
# camera_id=camera_id,
# camera_name=camera_name)

# def close(self):
# self.env.close()
