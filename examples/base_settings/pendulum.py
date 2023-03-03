import os
import sys
import numpy as np


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments.pendulum import PendulumEnv

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pirl.settings import PirlSettings


def get_settings(approach, model_name):
    _ = PendulumEnv()

    settings = PirlSettings(
        name="Pendulum",
        env_cls=PendulumEnv,
        propagation_method=approach,
        model_name=model_name,
        close_env=True,  # Whether to call `environment.close()` after rollout
        num_state=3,  # Number of state dimensions
        num_control=1,  # Number of control dimensions
        horizon=50,  # Number of maximum steps of the environment
        substeps=2, # Number of substeps to use.
        m_init=np.array([[-1, 0., 0.]]),  # Initial state mean vector
        S_init=1e-5 * np.eye(3),  # Initial state covariance matrix
        weights=np.diag([2., 2., 0.1]),  # Weights for the target state
        target=np.array([[1., 0., 0.]]),  # target state vector
        control_ub=np.array([2.]),  # Upper bounds of the control variables
        control_lb=np.array([-2.]),  # Lower bounds of the control variables
        num_basis_functions=35,  # Number of basis functions to use in RBF-like policies
        pca_state=False,  # Whether to use PCA to reduce state dimensions
        n_samples=100,  # Number of samples for sampling based uncertainty propagation
    )

    return settings
