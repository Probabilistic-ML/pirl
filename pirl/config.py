import tensorflow as tf
from gpflow.kernels import SquaredExponential, Exponential, Matern32, Matern52

float_type = tf.float64


def set_float_type(new_type):
    global float_type
    float_type = new_type


REWARD_IND = 1  # 0 uses environment reward to determine best results, 1 uses settings.reward. None uses the sum of both

DEFAULT_OPTIMIZE_KWARGS = dict(restarts=3, iters=1000, patience_stop=100,
                               patience_red=20, epsilon=1e-6, discount=0.5,
                               min_lr=0, verbose=3)

ABSTRACT_ERRMSG = "This is an abstract class. Please use the child classes."

DEFAULT_NOISE_VARIANCE_LOWER_BOUND = 5e-6
DEFAULT_LENGTHSCALE_LOWER_BOUND = 1e-5
DEFAULT_LENGTHSCALE_UPPER_BOUND = 1e5
KERNEL_MAP = {'squared_exponential': SquaredExponential,
              'exponential': Exponential,
              'matern32': Matern32,
              'matern52': Matern52}

BNN_CONFIG = {
    "widths": [500] * 3,
    "activations": ["swish"] * 3,
    "weight_decays": [0.0001] + [0.00025] * 2 + [0.0005],
    "batch_size": 64,
    "learning_rate": 1e-3,
    "epochs": 500,
}



