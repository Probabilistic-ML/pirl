import os
import sys
import numpy as np


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments.continuous_mountain_car import Continuous_MountainCarEnv

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pirl.settings import PirlSettings


def get_settings(approach, model_name):
    bnd = 1.
    weights = [5., 1.]
    horizon = 50
    substeps = 4

    _ = Continuous_MountainCarEnv(weights=weights, random_init=True, max_steps=horizon * substeps, action_bnd=bnd)

    settings = PirlSettings(
        name="ContinuousMountainCar",
        env_cls=Continuous_MountainCarEnv,
        propagation_method=approach,
        model_name=model_name,
        close_env=True,  # Whether to call `environment.close()` after rollout
        num_state=2,  # Number of state dimensions
        num_control=1,  # Number of control dimensions
        horizon=horizon,  # Number of maximum steps of the environment
        substeps=substeps,  # Number of substeps to use.
        m_init=np.array([[-0.6, 0]]),  # Initial state mean vector
        S_init=1e-5 * np.eye(2),  # Initial state covariance matrix
        weights=np.diag(weights),  # Weights for the target state
        target=np.array([[0.5, 0]]),  # target state vector
        control_ub=np.array([bnd]),  # Upper bounds of the control variables
        control_lb=np.array([-bnd]),  # Lower bounds of the control variables
        num_basis_functions=35,  # Number of basis functions to use in RBF-like policies
        pca_state=False,  # Whether to use PCA to reduce state dimensions
        n_samples=100,  # Number of samples for sampling based uncertainty propagation
        env_init_kwargs=dict(weights=weights, random_init=True, max_steps=horizon * substeps, action_bnd=bnd)
    )

    return settings
