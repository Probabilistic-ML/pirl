from copy import deepcopy
import dataclasses
from typing import Any, Literal, Optional, Union, Callable, List, Type

import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer
import numpy as np

from pirl.rewards.base import ExponentialReward, ExponentialMMReward
from pirl.controllers.base import RbfController, RbfMMController
from pirl.controllers.bounders import trapezoidal_squash
from pirl.models.model_trainer.trainer_MM import model_trainer as mm_model_trainer
from pirl.models.model_trainer.trainer_filter import get_filter_model_trainer
from pirl.models.model_trainer.trainer_sampling import get_sampling_model_trainer
from pirl.uncertainty.noisy_predict import GPEKF, GPUKF, GPPF


@dataclasses.dataclass
class PirlSettings:
    name: str  # Name of the environment/experiment
    env_cls: Type  # A gym like environment class

    propagation_method: Literal["sampling", "ekf", "ukf", "pf", "momentmatching"]  # Uncertainty propagation method
    model_name: Literal["dgcn", "bnn",
                        "squared_exponential", "exponential", "matern32", "matern52"]  # kernels are used with GP
    close_env: bool  # Whether to call `environment.close()` after rollout
    num_state: int  # Number of state dimensions
    num_control: int  # Number of control dimensions
    horizon: int  # Number of maximum steps of the environment
    substeps: int  # Number of substeps to use. The last control value is repeated this many steps after each prediction.
    m_init: np.ndarray  # Initial state mean vector
    S_init: np.ndarray  # Initial state covariance matrix
    weights: np.ndarray  # Weights for the target state
    target: np.ndarray  # target state vector
    control_ub: np.ndarray  # Upper bounds of the control variables
    control_lb: np.ndarray  # Lower bounds of the control variables
    num_basis_functions: int  # Number of basis functions to use in RBF-like policies
    pca_state: bool  # Whether to use PCA to reduce state dimensions
    pca_components: Optional[int] = None  # Number of maximum pca components. If None, only variance threshold is used
    n_samples: int = 200  # Number of samples for sampling based uncertainty propagation
    save_name: Optional[str] = None  # path to save the results
    optimizer_config: Union[Literal["DE", "hybrid"], dict, Optimizer, None] = None  # for the inner optimization
    population_size: int = 50  # population size for differential evolution
    reward_calc: Optional[Callable] = None  # Used to make a scalar reward from immediate rewards, None uses mean
    env_init_args: tuple = tuple()  # init args for the environment
    env_init_kwargs: dict = dataclasses.field(default_factory=dict)  # init kwargs for the environment

    def __post_init__(self):
        if self.m_init.ndim < 2:
            self.m_init = self.m_init.reshape((1, -1))
        if self.target.ndim < 2:
            self.target = self.target.reshape((1, -1))
        if self.weights.ndim < 2:
            self.weights = np.diag(self.weights)
        if not isinstance(self.control_ub, np.ndarray):
            self.control_ub = np.array(self.control_ub)
        if not isinstance(self.control_lb, np.ndarray):
            self.control_lb = np.array(self.control_lb)

        # noinspection PyTypeChecker
        self.propagation_method = self.propagation_method.lower()
        # noinspection PyTypeChecker
        self.model_name = self.model_name.lower()

        if not self.save_name:
            self.save_name = f"results/{self.name}/{self.name}_{self.propagation_method}/" \
                             f"{self.name}_{self.propagation_method}_{self.model_name}"
        self.check_sanity()

        if self.optimizer_config is None:
            self.optimizer_config = tf.keras.optimizers.Adam(0.2)

        if isinstance(self.optimizer_config, Optimizer):
            self.optimizer_config = self.optimizer_config.get_config()

        if self.optimizer_config not in ["DE", "hybrid"]:
            self.population_size = 0

    @property
    def approach(self) -> Literal["sampling", "density"]:
        if self.propagation_method == "sampling":
            return "sampling"
        return "density"

    @property
    def model_trainer(self):
        return get_model_trainer(self.propagation_method, self.model_name, samples=self.n_samples,
                                 fixed_lhs=True)

    @property
    def controller_initer(self):
        return get_controller_initer(self.propagation_method, self.num_basis_functions,
                                     self.control_ub, self.control_lb)

    @property
    def optimizer(self):
        if self.optimizer_config in ["DE", "hybrid"]:
            return self.optimizer_config
        optimizer = tf.keras.optimizers.get(self.optimizer_config["name"])
        return optimizer.from_config(self.optimizer_config)

    @property
    def reward(self):
        return get_reward_inst(self.propagation_method, self.num_state, self.weights, self.target)

    def check_sanity(self):
        if self.approach == 'momentmatching' and self.model_name != 'squared_exponential':
            raise ValueError("MomentMatching requires squared_exponential model")
        if self.m_init.size != self.num_state:
            raise ValueError(f"Mismatch between num_state ({self.num_state}) and m_init shape ({self.m_init.shape})")
        if self.S_init.shape[0] != self.num_state:
            raise ValueError(f"Mismatch between num_state ({self.num_state}) and S_init shape ({self.m_init.shape})")
        if self.S_init.shape[0] != self.S_init.shape[1]:
            raise ValueError(f"S_init has to be a square matrix, got shape ({self.m_init.shape})")
        if self.target.size != self.num_state:
            raise ValueError(f"Mismatch between num_state ({self.num_state}) and target shape ({self.m_init.shape})")
        if self.weights.shape[0] != self.num_state:
            raise ValueError(f"Mismatch between num_state ({self.num_state}) and weights shape ({self.m_init.shape})")
        if self.control_lb.size != self.num_control:
            raise ValueError(f"Mismatch between num_control ({self.num_control}) and control_lb shape "
                             f"({self.control_lb.shape})")
        if self.control_ub.size != self.num_control:
            raise ValueError(f"Mismatch between num_control ({self.num_control}) and control_ub shape "
                             f"({self.control_ub.shape})")

    @property
    def env(self):
        return self.env_cls(*self.env_init_args, **self.env_init_kwargs)

    # For backwards compatibility with dotmap
    def keys(self):
        return dataclasses.asdict(self).keys()

    def values(self):
        return dataclasses.asdict(self).values()

    def items(self):
        return dataclasses.asdict(self).items()


def get_model_trainer(approach, model_name, samples=200, fixed_lhs=True):
    approach = approach.lower()
    if approach == 'momentmatching':
        return mm_model_trainer
    elif approach == 'ekf':
        return get_filter_model_trainer(GPEKF, model_name)
    elif approach == 'ukf':
        return get_filter_model_trainer(GPUKF, model_name)
    elif approach == 'pf':
        return get_filter_model_trainer(GPPF, model_name, samples=samples, fixed_lhs=fixed_lhs)
    elif approach == 'sampling':
        return get_sampling_model_trainer(model_name)


def get_controller_initer(approach, num_basis_functions, control_ub, control_lb):
    if approach.lower() == 'sampling':
        return RbfController.to_initer(num_basis_functions=num_basis_functions,
                                       control_ub=control_ub, control_lb=control_lb,
                                       bound_func=trapezoidal_squash)
    else:
        return RbfMMController.to_initer(num_basis_functions=num_basis_functions,
                                         control_ub=control_ub, control_lb=control_lb,
                                         bound_func=trapezoidal_squash)


def get_reward_inst(approach, num_state, weights, target):
    if approach.lower() == 'sampling':
        return ExponentialReward(state_dim=num_state,
                                 weights=weights,
                                 target=target,
                                 transformable=True)
    else:
        return ExponentialMMReward(state_dim=num_state,
                                   weights=weights,
                                   target=target,
                                   transformable=True)
