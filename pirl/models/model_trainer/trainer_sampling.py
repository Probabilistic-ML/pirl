import os
import sys
import tensorflow as tf
from gpflow.utilities.ops import square_distance
from .dgcn_trainer_sampling import dgcn_model_trainer
from .bnn_trainer_sampling import bnn_model_trainer
from ..gp import GP

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(".."))))

from ...config import KERNEL_MAP


def get_sampling_model_trainer(model_name='squared_exponential'):
    if model_name == 'dgcn':
        return dgcn_model_trainer
    elif model_name == 'bnn':
        return bnn_model_trainer
    else:
        kernel_cls = KERNEL_MAP[model_name]

        def model_trainer(X, Y):
            gp = GP(X, Y, kernel_cls)
            Ls, lengths, sigma2s = gp.fit()
            predictor = Predictor(X, Y, Ls, lengths, sigma2s, model_name)
            return predictor

        return model_trainer


class Predictor:
    def __init__(self, X, Y, Ls, lengths, sigma2s, model_name='squared_exponential'):
        self.X = X
        self.Ls = Ls
        self.beta = tf.linalg.cholesky_solve(Ls, tf.transpose(Y)[:, :, None])
        self.ls = lengths
        self.kernel_var = sigma2s
        self.model_name = model_name

    def covariance(self, X1, X2):
        x1 = X1[None, :, :] / self.ls[:, None, :]
        x2 = X2[None, :, :] / self.ls[:, None, :]
        dist = square_distance(x1, x2)
        dist = tf.maximum(dist, 1e-17)
        cov = tf.linalg.diag_part(tf.transpose(dist, [1, 3, 0, 2]))
        cov = tf.transpose(cov, [2, 0, 1])
        if self.model_name == 'squared_exponential':
            cov = tf.exp(-0.5 * cov)
        elif self.model_name == 'exponential':
            cov = tf.exp(-tf.sqrt(cov))
        elif self.model_name == 'matern32':
            cov = tf.sqrt(3 * cov)
            cov = (1. + cov) * tf.exp(-cov)
        elif self.model_name == 'matern52':
            cov2 = tf.sqrt(5 * cov)
            cov = (1. + cov2 + 5 / 3 * cov) * tf.exp(-cov2)
        cov = self.kernel_var[:, None, None] * cov
        return cov

    def predict(self, X_test):
        cov = self.covariance(X_test, self.X)  # [2, 50, 100]
        y_pred = tf.matmul(cov, self.beta)[:, :, 0]
        var = self.covariance(X_test, X_test)  # [2, 50, 50]
        var = tf.add(var, - tf.matmul(cov,
                                      tf.linalg.cholesky_solve(self.Ls,
                                                               tf.transpose(cov,
                                                                            perm=[0, 2, 1]))))
        var = tf.linalg.diag_part(var)
        var = tf.linalg.diag(tf.transpose(var))
        return tf.transpose(y_pred), var
