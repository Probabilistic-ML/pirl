import os
import pickle
import dataclasses
import time

import numpy as np
import tensorflow as tf
from copy import deepcopy

from .settings import PirlSettings
from .uncertainty.probabilistic_controller import DensityController, SamplingController, BatchSamplingController
from .optimizers import TensorflowOptimizer, DifferentialEvolution
from .transformers import StandardScaler, PCA, Pipeline
from .rewards.base import TransitionReward, CombinedReward
from .datastore import DataStore
from .config import float_type, REWARD_IND
from .misc.rollout import rollout

from .models.dgcn import DGCN
from .models.bnn import BNN
from .uncertainty.noisy_predict import GPEKF, GPUKF, GPPF


class Agent:
    def __init__(self, settings):
        tf.config.run_functions_eagerly(False)

        if settings.approach == 'density':
            settings.population_size = 0
        elif settings.approach == 'sampling' and settings.population_size == 1:
            settings.population_size = 0
        self.batch_sampling = (settings.approach == 'sampling' and settings.population_size > 1)

        self.settings = settings

        self.dims = dict(state=settings.num_state,
                         ctrl=settings.num_control,
                         latent=settings.num_state,
                         pop_size=settings.population_size)

        self.transforms = self.get_transforms()
        self.datastore = DataStore()

        m_init, S_init = self.get_initial_distribution()
        self.m_init = tf.Variable(m_init, trainable=False, name='m_init', dtype=float_type)
        self.S_init = tf.Variable(S_init, trainable=False, name='S_init', dtype=float_type)

        self.horizon = settings.horizon

        if isinstance(settings.reward, CombinedReward):
            self.transition = 2 * int(isinstance(settings.reward.BaseRewards[0], TransitionReward))
        else:
            self.transition = int(isinstance(settings.reward, TransitionReward))
        self.reward_raw = settings.reward
        self.reward = deepcopy(settings.reward)

        self.controller = None  # for random actions in initial experiments
        self.dyn_model = None  # for action computation wo training

        if settings.optimizer is None:
            self.optimizer = TensorflowOptimizer(tf.keras.optimizers.Adam(0.1))
        elif settings.optimizer in ['DE', 'hybrid']:
            self.optimizer = DifferentialEvolution()
        else:
            self.optimizer = TensorflowOptimizer(deepcopy(settings.optimizer))

        self.reward_ind = REWARD_IND
        self.best_ind = -1
        self._update_init = set()

    def get_transforms(self):
        state_scaler = StandardScaler()
        if self.settings.pca_state:
            state_scaler = Pipeline([state_scaler, PCA(self.settings.pca_components,
                                                       0.9999)])
        transforms = {"state": state_scaler,
                      "ctrl": StandardScaler(),
                      "reward": StandardScaler()}
        return transforms

    def get_initial_distribution(self):
        self._update_init = set()
        if self.settings.m_init is None:
            self._update_init.add("m")
            m_init = tf.zeros((1, self.dims["state"]))
        else:
            m_init = self.settings.m_init

        if self.settings.S_init is None:
            self._update_init.add("S")
            S_init = tf.eye(self.dims["state"], dtype=float_type)
        else:
            S_init = self.settings.S_init

        return m_init, S_init

    @property
    def horizon(self):
        return self._horizon

    @horizon.setter
    def horizon(self, new_horizon):
        if new_horizon is None:
            self._horizon = None
            self.loss_factor = None
        else:
            self._horizon = tf.constant(new_horizon, dtype=tf.int32)
            self.loss_factor = tf.constant(-1 / new_horizon,
                                           dtype=float_type)

    @property
    def update_m_init(self):
        return "m" in self._update_init

    @update_m_init.setter
    def update_m_init(self, value):
        if value:
            self._update_init.add("m")
        else:
            self._update_init.discard("m")

    @property
    def update_S_init(self):
        return "S" in self._update_init

    @update_S_init.setter
    def update_S_init(self, value):
        if value:
            self._update_init.add("S")
        else:
            self._update_init.discard("S")

    def experiment(self, env=None, num=1, timesteps=None, substeps=None, controller=None,
                   render=False, verbose=1, set_init=False, deplete_env=False,
                   render_kwargs=None, request_deterministic_start=False):
        env = env if env else self.settings.env
        timesteps = timesteps if timesteps else self.horizon
        if verbose > 1:
            print(f"Experimenting with {timesteps} steps and {self.settings.substeps} substeps")
        R_max = -np.inf

        try:
            random_init_buffer = env.random_init
        except AttributeError:
            random_init_buffer = True

        if request_deterministic_start:
            env.random_init = False

        for i_exp in tf.range(num):
            X, U, R = rollout(env,
                              timesteps=timesteps,
                              substeps=substeps if substeps else self.settings.substeps,
                              controller=controller if controller else self.controller,
                              render=render, verbose=verbose,
                              x_init=self.m_init if set_init else None,
                              deplete_env=deplete_env,
                              close=self.settings.close_env,
                              render_kwargs=render_kwargs)
            rewards = self.update(X, U, R, fit_transforms=i_exp == num - 1)
            R_sum = rewards.sum() if self.reward_ind is None else rewards.sum(0)[self.reward_ind]
            if R_sum > R_max:
                R_max = R_sum
                if self.controller is not None:
                    self.datastore.update_params(self.controller.params,
                                                 self.controller.controller.specs,
                                                 stop_ind=-2,
                                                 substeps=self.settings.substeps,
                                                 reward=R_max)
        env.random_init = random_init_buffer

    def _rewards(self, X, U, R):
        times = tf.range(1, R.shape[0] + 1) / self.horizon
        X = tf.cast(X, dtype=float_type)  # otherwise possibly wrong dtype
        U = tf.cast(U, dtype=float_type)  # for reward computation

        if self.transition == 1:
            self.reward_raw.prev_states = tf.reshape(X[0:1], (1, -1))
        elif self.transition == 2:
            self.reward_raw.BaseRewards[0].prev_states = tf.reshape(X[0:1], (1, -1))

        if self.settings.approach == 'density':
            S_x = 0 * tf.eye(X.shape[1], dtype=float_type)  # 1e-6
            S_u = 0 * tf.eye(U.shape[1], dtype=float_type)
            R_ = [self.reward_raw.compute_reward(tf.reshape(x, (1, -1)), S_x,
                                                 tf.reshape(u, (1, -1)), S_u,
                                                 tf.reshape(t, (1,)))
                  for x, u, t in zip(X[1:], U, times)]
        elif self.settings.approach == 'sampling':
            R_ = [self.reward_raw.compute_reward(tf.reshape(x, (1, -1)),
                                                 tf.reshape(u, (1, -1)),
                                                 tf.reshape(t, (1,)))
                  for x, u, t in zip(X[1:], U, times)]
        else:
            raise ValueError("settings.approach must be either sampling or density!")
        return np.c_[R, np.array(R_).reshape((-1, 1))]

    def update(self, states, controls, rewards, fit_transforms=True):
        self.best_ind = -1  # force to take whole data
        rewards = self._rewards(states, controls, rewards)
        self.datastore.update(states, controls, rewards)
        if fit_transforms:
            self.fit_transforms()
        if self._update_init:
            m_, S_ = self.datastore.get_init_states()
            if self.update_m_init:
                self.m_init.assign(tf.convert_to_tensor(m_))
            if self.update_S_init:
                self.S_init.assign(tf.convert_to_tensor(S_))
        return rewards

    def training_data(self):
        X, Y, R = self.datastore.training_data(stop_ind=self.best_ind,
                                               substeps=self.settings.substeps,
                                               check_unique=True, as_tensor=True)
        X_state = self.transforms["state"].transform(X[:, :self.dims["state"]])
        Y = self.transforms["state"].transform_std(Y)
        X_ctrl = self.transforms["ctrl"].transform(X[:, self.dims["state"]:])
        R = self.transforms["reward"].transform(R)
        return tf.concat((X_state, X_ctrl), axis=1), Y, R

    def fit_transforms(self):
        X, Y, R = self.datastore.training_data(stop_ind=self.best_ind,
                                               substeps=self.settings.substeps,
                                               check_unique=True, as_tensor=True)
        self.transforms["state"].fit(X[:, :self.dims["state"]])
        Y_ = self.transforms["state"].transform_std(Y)
        self.dims["latent"] = Y_.shape[1]
        self.transforms["ctrl"].fit(X[:, self.dims["state"]:])
        self.transforms["reward"].fit(R)
        self.reward.state_transformer = self.transforms["state"]

    def fit(self, *args, **kwargs):
        start = time.time()
        print("Fitting state model...")
        X, Y, _ = self.training_data()
        if X.shape[0] <= 2:
            raise ValueError("Not enough data points (invalid number of substeps?)")
        self.dyn_model = self.settings.model_trainer(X, Y, *args, **kwargs)
        duration = time.time() - start
        self.datastore.fit_durations.append(duration)
        return self

    def init_controller(self, params=None, pop_size=None):
        pop_size = pop_size if pop_size is not None else self.settings.population_size
        pop_size = 0 if pop_size == 1 else pop_size
        base_controller = self.settings.controller_initer((self.dims["latent"],
                                                           self.dims["ctrl"]),
                                                          pop_size)
        if self.settings.approach == 'density':
            self.controller = DensityController(base_controller,
                                                self.dyn_model,
                                                self.reward,
                                                self.settings.reward_calc,
                                                self.transforms,
                                                self.m_init, self.S_init,
                                                self.horizon)
        elif pop_size == 0:
            self.controller = SamplingController(base_controller,
                                                 self.dyn_model,
                                                 self.reward,
                                                 self.settings.reward_calc,
                                                 self.transforms,
                                                 self.m_init, self.S_init,
                                                 self.horizon,
                                                 n_samples=self.settings.n_samples)
        elif pop_size > 1:
            self.controller = BatchSamplingController(base_controller,
                                                      self.dyn_model,
                                                      self.reward,
                                                      self.settings.reward_calc,
                                                      self.transforms,
                                                      self.m_init, self.S_init,
                                                      self.horizon,
                                                      n_samples=self.settings.n_samples)
        if params is not None:
            self.controller.params = params

    def optimize(self, restarts=1, iters=1000, patience_stop=50, patience_red=10, epsilon=0, discount=0.5, min_lr=0,
                 verbose=1, params=None, **kwargs):
        start = time.time()
        if not hasattr(self.datastore, 'loss_history'):
            self.datastore.loss_history = []
        tf.print("Fitting controller...")
        try:
            self.to_best_params(fit_transforms=False)
            lowest_cost = tf.convert_to_tensor(np.inf, dtype=float_type)
        except ValueError:
            self.init_controller()
            lowest_cost = tf.convert_to_tensor(np.inf, dtype=float_type)
        best_param = self.controller.params
        loss_history_tmp = []
        for r in tf.range(restarts):
            tf.print("Restart no. %i..." % (r + 1))
            if params is not None:
                tf.print("Using initial controller parameters...")
                self.controller.params = params
            tf.print("Optimizing controller...")
            cost, param = self.optimizer.optimize(self.controller, iters=iters,
                                                  patience_stop=patience_stop,
                                                  patience_red=patience_red,
                                                  epsilon=epsilon,
                                                  discount=discount,
                                                  min_lr=min_lr,
                                                  verbose=verbose)
            if hasattr(self.optimizer, 'loss_history'):
                loss_history_tmp.append(self.optimizer.loss_history)
            if cost < lowest_cost:
                best_param = param.copy()
                lowest_cost = cost
            if verbose:
                tf.print(f"Restart {r + 1} - Loss: {lowest_cost:.4e}")
            if params is None:
                self.controller.randomize()
        self.datastore.loss_history.append(loss_history_tmp)
        if self.batch_sampling:
            self.init_controller(best_param, pop_size=0)
        else:
            self.init_controller(best_param)
        duration = time.time() - start
        self.datastore.optimize_durations.append(duration)
        return -1. * lowest_cost, best_param

    def auxiliary_optimize(self, iters=200, patience_stop=50, patience_red=20, epsilon=5e-6, discount=0.4, min_lr=0,
                           verbose=1, **kwargs):
        optimizer = TensorflowOptimizer(tf.keras.optimizers.Adam(0.1))
        lowest_cost, best_param = optimizer.optimize(self.controller, iters=iters,
                                                     patience_stop=patience_stop,
                                                     patience_red=patience_red,
                                                     epsilon=epsilon,
                                                     discount=discount,
                                                     min_lr=min_lr,
                                                     verbose=verbose,
                                                     profile_tf=False)
        if self.batch_sampling:
            self.init_controller(best_param, pop_size=0)
        else:
            self.init_controller(best_param)
        return -1. * lowest_cost, best_param

    def compute_action(self, x):
        if self.controller is None:
            raise ValueError("No controller found!")
        return self.controller.compute_action(x)

    def best_experiment(self):
        return self.datastore.best_experiment(reward_ind=self.reward_ind)

    def best_params(self):
        return self.datastore.best_param()

    def to_best_params(self, fit_transforms=True, pop_size=None):
        reward, params_dict = self.best_params()
        self.best_ind = params_dict.get("stop_ind", -1)
        if fit_transforms:
            self.fit_transforms()
        self.init_controller(params=params_dict["params"], pop_size=pop_size)

    def save(self, save_name=None):
        save_name = save_name if save_name else self.settings.save_name
        datastore_name = to_datastore_name(save_name)
        os.makedirs(os.path.dirname(datastore_name), exist_ok=True)
        with open(datastore_name, "wb") as f:
            pickle.dump({"datastore": self.datastore,
                         "settings": dataclasses.asdict(self.settings)},
                        f)

    @classmethod
    def load(cls, save_name):
        with open(to_datastore_name(save_name), "rb") as f:
            old_data = pickle.load(f)
        settings = PirlSettings(**old_data["settings"])
        new = cls(settings)
        new.datastore = old_data["datastore"]
        new.fit_transforms()
        return new


def to_datastore_name(save_name):
    return save_name + ".datastore.pkl"
