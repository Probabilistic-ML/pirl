import os
import sys
from operator import itemgetter
import tensorflow as tf
import numpy as np
from tensorflow_probability import bijectors as tfb
import gpflow

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(".."))))
from pirl.config import float_type, DEFAULT_NOISE_VARIANCE_LOWER_BOUND, DEFAULT_LENGTHSCALE_LOWER_BOUND, \
    DEFAULT_LENGTHSCALE_UPPER_BOUND


class StepCallback:
    def __init__(self, model):
        self.model = model
        self.history = []
        self.best_params = None
        self.best_vars = None

    def monitor(self, step, variables, values):
        self.history.append((self.model.training_loss().numpy(),
                             [v.numpy() for v in self.model.variables]))
        self.best_vars = min(reversed(self.history), key=itemgetter(0))


class GP:
    def __init__(self, X, Y, kernel_cls, return_iK=False, restarts=5):
        self.X = X
        self.Y = Y
        self.kernel_cls = kernel_cls
        self.return_iK = return_iK
        gpflow.config.set_default_float(float_type)
        self.n_sample, self.n_inps = X.shape
        self.n_outs = Y.shape[1]
        self.eye = tf.eye(self.n_sample, dtype=float_type)
        self.step_callbacks = []
        self.restarts = restarts

    def fit(self):
        Ls, iKs, lengths, sigma2s = [], [], [], []
        for i_out in range(self.n_outs):
            kern = self.kernel_cls(
                lengthscales=tf.ones((self.n_inps,), dtype=float_type)
            )
            kern.lengthscales = bounded_lengthscale(DEFAULT_LENGTHSCALE_LOWER_BOUND,
                                                    DEFAULT_LENGTHSCALE_UPPER_BOUND,
                                                    tf.ones((self.n_inps,), dtype=float_type))
            model = gpflow.models.GPR((self.X, self.Y[:, i_out:i_out + 1]),
                                      kernel=kern)
            model.likelihood = gpflow.likelihoods.Gaussian(variance_lower_bound=DEFAULT_NOISE_VARIANCE_LOWER_BOUND)
            step_callback = StepCallback(model)
            for _ in range(self.restarts):
                try:
                    optimizer = gpflow.optimizers.Scipy()
                    optimizer.minimize(model.training_loss, model.trainable_variables,
                                       step_callback=step_callback.monitor)
                except:
                    print("ERROR during gp training...")
                    print(f"{sys.exc_info()[0]}")
                    print(f"{sys.exc_info()[1]}")
                    print(f"{sys.exc_info()[2]}")
                randomize_kernel(model.kernel)
                randomize_noise(model.likelihood)
            self.step_callbacks.append(step_callback)
            for i, v in enumerate(model.variables):
                v.assign(step_callback.best_vars[1][i])
            lengths_tmp = model.kernel.lengthscales.numpy()
            sigma2_tmp = model.kernel.variance.numpy()
            noise_tmp = model.likelihood.variance.numpy()
            lengths.append(lengths_tmp)
            sigma2s.append(sigma2_tmp)
            Kxx = model.kernel.K(self.X) + noise_tmp * self.eye
            L = tf.linalg.cholesky(Kxx)
            Ls.append(L)
            if self.return_iK:
                iKs.append(tf.linalg.cholesky_solve(L, self.eye))
            del model, optimizer
        [Ls, iKs, lengths, sigma2s] = [tf.convert_to_tensor(arr, dtype=float_type)
                                       for arr in [Ls, iKs, lengths, sigma2s]]
        if self.return_iK:
            return iKs, lengths, sigma2s
        return Ls, lengths, sigma2s


def bounded_lengthscale(low, high, lengthscales):
    sigmoid = tfb.Sigmoid(tf.cast(low, dtype=float_type), tf.cast(high, dtype=float_type))
    parameter = gpflow.Parameter(lengthscales, transform=sigmoid, dtype=float_type)
    return parameter


# from https://github.com/nrontsis/PILCO/blob/master/pilco/models/mgpr.py
def randomize_kernel(kernel, mean=1, sigma=0.01):
    ls = mean + sigma * np.random.normal(size=kernel.lengthscales.shape)
    var = mean + sigma * np.random.normal(size=kernel.variance.shape)
    kernel.lengthscales.assign(ls)
    if kernel.variance.trainable:
        kernel.variance.assign(var)


def randomize_noise(likelihood, mean=1, sigma=0.01):
    if likelihood.variance.trainable:
        noise = mean + sigma * np.random.normal()
        likelihood.variance.assign(noise)
