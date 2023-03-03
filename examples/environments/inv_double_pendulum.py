import numpy as np

# InvertedDoublePendulum
# https://github.com/openai/gym/blob/master/gym/envs/mujoco/inverted_double_pendulum.py

# The observation is a `ndarray` with shape `(11,)` where the elements correspond to the following:
# | Num | Observation           | Min                  | Max                | Name (in corresponding XML file) | Joint| Unit |
# |-----|-----------------------|----------------------|--------------------|----------------------|--------------------|--------------------|
# | 0   | position of the cart along the linear surface                        | -Inf                 | Inf                | slider | slide | position (m) |
# | 1   | sine of the angle between the cart and the first pole                | -Inf                 | Inf                | sin(hinge) | hinge | unitless |
# | 2   | sine of the angle between the two poles                              | -Inf                 | Inf                | sin(hinge2) | hinge | unitless |
# | 3   | cosine of the angle between the cart and the first pole              | -Inf                 | Inf                | cos(hinge) | hinge | unitless |
# | 4   | cosine of the angle between the two poles                            | -Inf                 | Inf                | cos(hinge2) | hinge | unitless |
# | 5   | velocity of the cart                                                 | -Inf                 | Inf                | slider | slide | velocity (m/s) |
# | 6   | angular velocity of the angle between the cart and the first pole    | -Inf                 | Inf                | hinge | hinge | angular velocity (rad/s) |
# | 7   | angular velocity of the angle between the two poles                  | -Inf                 | Inf                | hinge2 | hinge | angular velocity (rad/s) |
# | 8   | constraint force - 1                                                 | -Inf                 | Inf                |  |  | Force (N) |
# | 9   | constraint force - 2                                                 | -Inf                 | Inf                |  |  | Force (N) |
# | 10  | constraint force - 3                                                 | -Inf                 | Inf                |  |  | Force (N) |


# ### Rewards
# The reward consists of two parts:
# - *alive_bonus*: The goal is to make the second inverted pendulum stand upright
# (within a certain angle limit) as long as possible - as such a reward of +10 is awarded
# for each timestep that the second pole is upright.
# - *distance_penalty*: This reward is a measure of how far the *tip* of the second pendulum
# (the only free end) moves, and it is calculated as
# *0.01 * x<sup>2</sup> + (y - 2)<sup>2</sup>*, where *x* is the x-coordinate of the tip
# and *y* is the y-coordinate of the tip of the second pole.
# - *velocity_penalty*: A negative reward for penalising the agent if it moves too
# fast *0.001 *  v<sub>1</sub><sup>2</sup> + 0.005 * v<sub>2</sub> <sup>2</sup>*
# The total reward returned is ***reward*** *=* *alive_bonus - distance_penalty - velocity_penalty*
# ### Starting State
# All observations start in state
# (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) with a uniform noise in the range
# of [-0.1, 0.1] added to the positional values (cart position and pole angles) and standard
# normal force with a standard deviation of 0.1 added to the velocity values for stochasticity.
# ### Episode Termination
# The episode terminates when any of the following happens:
# 1. The episode duration reaches 1000 timesteps.
# 2. Any of the state space values is no longer finite.
# 3. The y_coordinate of the tip of the second pole *is less than or equal* to 1. The maximum standing height of the system is 1.196 m when all the parts are perpendicularly vertical on top of each other.

DEFAULT_SIZE = 500


class InvDoublePendulumEnv:

    def __init__(self):
        import gym
        # we could avoid the following import as it is just there to check the availability of the
        # mujoco_py library but gym raises a custom error, which is inconvenient as we also want to avoid a hard
        # dependency on gym. So this is the cleanest solution we could come up with to have the basic python error.
        import mujoco_py

        self.env = gym.make('InvertedDoublePendulum-v2')
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self._state = None
        self.gym_state = None

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, s):
        self._state = s
        self.env.sim.data.qpos[:] = s[:3]
        self.env.sim.data.qvel[:] = s[3:]

    @staticmethod
    def state_trans(ob):
        a1 = np.arctan2(ob[1], ob[3])
        a2 = np.arctan2(ob[2], ob[4])
        # Array of angles in radians, in the range [-pi, pi]
        s_new = np.hstack([ob[0], a1, a2, ob[5:-3]])
        return s_new

    # ---> state_trans(state) = (x, gamma, theta, vx, theta_dot, gamma_dot)

    @staticmethod
    def inv_state_trans(s):
        return np.array([s[0], np.sin(s[1]), np.sin(s[2]), np.cos(s[1]), np.cos(s[2]), s[3], s[4], s[5], 0., 0., 0.])
        # (x, sin(gamma), sin(theta), cos(gamma), cos(theta), vx, theta_dot, gamma_dot, 0., 0., 0.)

    def step(self, action):
        ob, r, _, _ = self.env.step(action)
        x, _, y = self.env.sim.data.site_xpos[0]
        done = bool(y <= 0.9)  # in gym y <= 1; decreased threshold to collect more data
        self.gym_state = ob
        self.state = self.state_trans(ob)
        return self.state, np.array([r]), done, {}

    def reset(self):
        ob = self.env.reset()
        self.gym_state = ob
        self.state = self.state_trans(ob)
        return self.state

    def render(self, mode="human",
               width=DEFAULT_SIZE,
               height=DEFAULT_SIZE,
               camera_id=None,
               camera_name=None):
        self.env.render(mode=mode,
                        width=width,
                        height=height,
                        camera_id=camera_id,
                        camera_name=camera_name)

    def close(self):
        self.env.close()
