import numpy as np
import tensorflow as tf
from operator import itemgetter
from .config import float_type


class DataStore:
    def __init__(self):
        self.state_data = []
        self.control_data = []
        self.reward_data = []
        self.controller_specs = {}
        self.control_params = []
        self.loss_history = []
        self.dims = {"state": None,
                     "control": None,
                     "reward": None}
        self.fit_durations = []
        self.optimize_durations = []

    def update(self, state, control, reward):
        if not self.dims["state"]:
            self.dims["state"] = state.shape[1]
        if not self.dims["control"]:
            self.dims["control"] = control.shape[1]
        if not self.dims["reward"]:
            self.dims["reward"] = reward.shape[1]
        self.state_data.append(state)
        self.control_data.append(control)
        if reward.shape[0] < state.shape[0]:
            reward = np.append(np.zeros((state.shape[0] - reward.shape[0], reward.shape[1])), reward, axis=0)
        self.reward_data.append(reward)

    def update_params(self, params, controller_specs,
                      start_ind=0, stop_ind=-1, substeps=1, reward=None):
        reward = self.reward_history()[-1].sum() if reward is None else reward
        n_experiments = len(self.state_data)
        start_ind = positive_ind(start_ind, n_experiments)
        stop_ind = positive_ind(stop_ind, n_experiments)
        self.control_params.append((reward, {"start_ind": start_ind, "stop_ind": stop_ind,
                                             "substeps": substeps, "params": params,
                                             "controller_specs": controller_specs}))

    def training_data(self, start_ind=0, stop_ind=-1, substeps=1, check_unique=False,
                      as_tensor=False):
        n_experiments = len(self.state_data)
        start_ind = positive_ind(start_ind, n_experiments)
        stop_ind = positive_ind(stop_ind, n_experiments)
        index = range(start_ind, stop_ind + 1)
        if not self.state_data:
            raise ValueError("No data to extract!")
        # noinspection PyUnresolvedReferences
        X_ = np.empty((0, self.dims["state"] + self.dims["control"]))
        Y_ = np.empty((0, self.dims["state"]))
        R_ = np.empty((0, 1))

        for ind in index:
            ctrl = self.control_data[ind]
            state = self.state_data[ind]
            reward = self.reward_data[ind][:, [0]]
            extra = len(self.control_data[ind]) % substeps
            if extra:
                ctrl = ctrl[:-extra]
                state = state[:ctrl.shape[0] + 1]
                reward = reward[:ctrl.shape[0] + 1]
            u = ctrl[::substeps]
            u_back = np.repeat(u, substeps, axis=0)
            if not np.all(u_back == ctrl):
                print(f"Invalid number of substeps regarding control, skipping experiment {ind}")
                continue
            xi = state[:-substeps:substeps]
            xo = state[substeps::substeps]

            if reward.ndim < 2:
                reward = reward.reshape((-1, 1))
            reward = reward[1:].reshape((-1, substeps, reward.shape[1])).sum(1)
            X_ = np.append(X_, np.c_[xi, u], axis=0)
            Y_ = np.append(Y_, xo - xi, axis=0)
            R_ = np.append(R_, reward, axis=0)
        if check_unique:
            X_, inds = np.unique(X_, return_index=True, axis=0)
            inds = inds.tolist()
            Y_ = Y_[inds, :]
            R_ = R_[inds, :]
        if as_tensor:
            return tuple(tf.convert_to_tensor(a, dtype=float_type) for a in [X_, Y_, R_])
        return X_, Y_, R_

    def get_init_states(self):
        states = np.array([s[0] for s in self.state_data])
        return states.mean(0), np.cov(states, rowvar=False)

    def reward_history(self):
        return np.array([r.sum(0) for r in self.reward_data])

    def best_experiment(self, reward_ind=None):
        if reward_ind is None:
            return np.argmax(self.reward_history().sum(1))
        return np.argmax(self.reward_history(), axis=0)[reward_ind]

    def best_param(self):
        if not self.control_params:
            raise ValueError("Control params not inited yet!")
        return max(reversed(self.control_params), key=itemgetter(0))


def positive_ind(ind, upper_limit):
    if ind < 0:
        return upper_limit + ind
    return ind
