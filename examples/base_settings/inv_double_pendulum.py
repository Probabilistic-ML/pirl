import os
import sys
import numpy as np


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments.inv_double_pendulum import InvDoublePendulumEnv

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pirl.settings import PirlSettings


def get_settings(approach, model_name):
    _ = InvDoublePendulumEnv()

    settings = PirlSettings(
        name="InvDoublePendulum",
        env_cls=InvDoublePendulumEnv,
        propagation_method=approach,
        model_name=model_name,
        close_env=True,  # Whether to call `environment.close()` after rollout
        num_state=6,  # Number of state dimensions
        num_control=1,  # Number of control dimensions
        horizon=50,  # Number of maximum steps of the environment
        substeps=1,  # Number of substeps to use.
        m_init=np.zeros((1, 6)),  # Initial state mean vector
        S_init=1e-3 * np.eye(6),  # Initial state covariance matrix
        weights=np.diag([0.1, 1., 1., 0.2, 0.2, 0.2]),  # Weights for the target state
        target=np.zeros((1, 6)),  # target state vector
        control_ub=np.array([1.]),  # Upper bounds of the control variables
        control_lb=np.array([-1.]),  # Lower bounds of the control variables
        num_basis_functions=40,  # Number of basis functions to use in RBF-like policies
        pca_state=False,  # Whether to use PCA to reduce state dimensions
        n_samples=100,  # Number of samples for sampling based uncertainty propagation
    )

    return settings
