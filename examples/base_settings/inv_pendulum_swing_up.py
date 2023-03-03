import os
import sys
import numpy as np


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments.inv_pendulum_swing_up import InvPendulumEnv

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pirl.settings import PirlSettings


def get_settings(approach, model_name):
    _ = InvPendulumEnv()

    S_init = 1e-8 * np.eye(5)
    S_init[2, 2] = 6e-2
    S_init[3, 3] = 6e-2

    settings = PirlSettings(
        name="InvPendulumSwingUp",
        env_cls=InvPendulumEnv,
        propagation_method=approach,
        model_name=model_name,
        close_env=True,  # Whether to call `environment.close()` after rollout
        num_state=5,  # Number of state dimensions
        num_control=1,  # Number of control dimensions
        horizon=80,  # Number of maximum steps of the environment
        substeps=2,  # Number of substeps to use.
        m_init=np.array([[0., 0., -1., 0., 0.]]),  # Initial state mean vector
        S_init=S_init,  # Initial state covariance matrix
        weights=np.diag([0.2, 0., 1., 0., 0.1]),  # Weights for the target state
        target=np.array([0., 0., 1., 0., 0.]),  # target state vector
        control_ub=np.array([1.]),  # Upper bounds of the control variables
        control_lb=np.array([-1.]),  # Lower bounds of the control variables
        num_basis_functions=100,  # Number of basis functions to use in RBF-like policies
        pca_state=False,  # Whether to use PCA to reduce state dimensions
        n_samples=50,  # Number of samples for sampling based uncertainty propagation
    )

    return settings
