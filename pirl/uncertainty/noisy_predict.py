import abc
import tensorflow as tf
from scipy.stats import norm

from ..misc import type_checked
from ..config import float_type
from ..misc import lhs


class NoisyBaseClass(abc.ABC):
    def __init__(self, X=None, Y=None, is_ctrl=False):
        self.X = X
        self.Y = Y
        self.is_ctrl = is_ctrl
        if X is not None:
            self.num_dims = X.shape[1]
            self.num_datapoints = X.shape[0]
        else:
            self.num_dims = None
            self.num_datapoints = None
        if Y is not None:
            self.num_outputs = Y.shape[1]
        else:
            self.num_outputs = None

    @abc.abstractmethod
    def noisy_predict(self, m_in, S_in):
        """
        Abstract recurrent prediction method to be used in PILCO
        Implements the noise propagation given input mean and covariance

        Parameters
        ----------
        m_in : array(shape=(1, n_input))
            input mean
        S_in : array(shape=(n_input, n_input))
            input covariance

        Returns
        -------
        m_out : array(shape=(1, n_output))
            prediction mean
        S_out : array(shape=(n_output, n_output))
            prediction covariance
        C : array(shape=(num_inputs, n_output))
            input output 'covariance' metric # Better name?
        """
        raise NotImplementedError("This is a parent class. You should use one of its children!")


class MomentMatching(NoisyBaseClass):
    """
    Moment matching for noise propagation. Only works with RBF kernel. A GP
    must be fit first to acquire inv_cov, lengthscales and kernel_var

    Parameters
    ----------
    X : array(shape=(n_sample, n_input))
        Training inputs
    Y : array(shape=(n_sample, n_output))
        Training outputs
    inv_cov : array(shape=(n_output, n_sample, n_sample)
        Inverse of the covariance matrix of the GP `inv_cov=(kernel_var kernel(X, X) + sigma_noise I)^-1`
    lengthscales : array(shape=(n_output, n_input)
        Lenghtscales i.e. kernel parameters of the GP
    kernel_var : array(shape=(n_output,))
        Kernel variances of the GP

    """

    @type_checked
    def __init__(self, X, Y, inv_cov, lengthscales, kernel_var=None, is_ctrl=False):
        super().__init__(X=X, Y=Y, is_ctrl=is_ctrl)
        self.beta = tf.einsum('ijk,ki->ij', inv_cov, self.Y)
        self.ls = lengthscales
        if kernel_var is None:
            kernel_var = tf.ones(self.Y.shape[1], dtype=float_type)
        self.kernel_var = kernel_var
        if self.is_ctrl:
            self.iK = 0.0 * inv_cov
        else:
            self.iK = inv_cov
        if float_type == tf.float32:
            self.noise = 1e-6 * tf.eye(self.num_outputs, dtype=float_type)
        else:
            self.noise = tf.constant(0, dtype=float_type)

    def noisy_predict(self, m_in, S_in):
        """
        Propagate noise as defined by m_in and S_in through prediction using
        moment matching

        Parameters
        ----------
        m_in : array(shape=(1, n_input))
            input mean
        S_in : array(shape=(n_input, n_input))
            input covariance

        Returns
        -------
        m_out : array(shape=(1, n_output))
            prediction mean [Eq. (2.43)]
        S_out : array(shape=(n_output, n_output))
            prediction covariance [Eq. (2.44)]
        C : array(shape=(num_inputs, n_output))
            input output 'covariance' metric [Eq. (2.70)]# Better name?

        Based on the MATLAB code in Deisenroth 2010, Appendix E.1
        See also Deisenroth & Rasmussen 2011
        """
        S = tf.tile(S_in[None, None, :, :], [self.num_outputs, self.num_outputs, 1, 1])
        centralized_input = self.X - m_in
        inp = tf.tile(centralized_input[None, :, :], [self.num_outputs, 1, 1])

        iL = tf.linalg.diag(1 / self.ls)
        iN = inp @ iL
        B = iL @ S[0, ...] @ iL + tf.eye(self.num_dims, dtype=float_type)
        t = tf.linalg.matrix_transpose(
            tf.linalg.solve(B, tf.linalg.matrix_transpose(iN), adjoint=True, name='predict_gf_t_calc')
        )

        lb = tf.exp(-tf.reduce_sum(iN * t, -1) / 2) * self.beta
        tiL = t @ iL
        c = self.kernel_var / tf.sqrt(tf.linalg.det(B, name='det_B'))

        M = (tf.reduce_sum(lb, -1) * c)[:, None]
        V = tf.matmul(tiL, lb[:, :, None], adjoint_a=True)[..., 0] * c[:, None]

        R = S @ tf.linalg.diag(
            1 / tf.square(self.ls[None, :, :]) +
            1 / tf.square(self.ls[:, None, :])
        ) + tf.eye(self.num_dims, dtype=float_type)

        X = inp[None, :, :, :] / tf.square(self.ls[:, None, None, :])
        X2 = -inp[:, None, :, :] / tf.square(self.ls[None, :, None, :])
        Q = tf.linalg.solve(R, S, name='Q_solve') / 2
        Xs = tf.reduce_sum(X @ Q * X, -1)
        X2s = tf.reduce_sum(X2 @ Q * X2, -1)
        maha = -2 * tf.matmul(X @ Q, X2, adjoint_b=True) + Xs[:, :, :, None] + X2s[:, :, None, :]

        k = tf.math.log(self.kernel_var)[:, None] - tf.reduce_sum(tf.square(iN), -1) / 2
        L = tf.exp(k[:, None, :, None] + k[None, :, None, :] + maha)
        S = (tf.tile(self.beta[:, None, None, :], [1, self.num_outputs, 1, 1])
             @ L @
             tf.tile(self.beta[None, :, :, None], [self.num_outputs, 1, 1, 1])
             )[:, :, 0, 0]

        diagL = tf.transpose(tf.linalg.diag_part(tf.transpose(L)))
        S = S - tf.linalg.diag(tf.reduce_sum(tf.multiply(self.iK, diagL), [1, 2]))
        S = S / tf.sqrt(tf.linalg.det(R, name='det_R'))
        S = S + tf.linalg.diag(self.kernel_var)
        S = S - M @ tf.transpose(M)

        S = 0.5 * tf.add(S, tf.transpose(S)) + self.noise

        return tf.transpose(M), S, tf.transpose(V)

    @staticmethod
    def _maha(a, b, Q):
        p1 = tf.reduce_sum(tf.matmul(a, Q) * a, axis=-1)
        p2 = tf.reduce_sum(tf.matmul(b, Q) * b, axis=-1)
        p3 = -2 * tf.matmul(tf.matmul(a, Q), b, adjoint_b=True)

        return p1[:, :, :, None] + p2[:, :, None, :] + p3


class GPUKF(NoisyBaseClass):
    @type_checked
    def __init__(self, predictor, input_dims):
        super().__init__(X=None, Y=None, is_ctrl=False)
        self.predictor = predictor
        weight_0 = tf.cast((3 - input_dims) / 3, dtype=float_type)  # lambda = 3 - input_dims
        weight_0 = tf.reshape(weight_0, shape=(1,))
        weight_tmp = tf.constant([1 / 6], dtype=float_type)
        weights_tmp = tf.repeat(weight_tmp, 2 * input_dims)
        self.weights = tf.concat((weight_0, weights_tmp), axis=0)
        self.noise = 1e-5

    def noisy_predict(self, m_in, S_in):
        sigma_points = self.get_sigma_points(m_in, S_in)

        m_sp, S_sp = self.predictor.predict(sigma_points)

        m_out = tf.reduce_sum(tf.multiply(self.weights[:, None],
                                          m_sp),
                              axis=0,
                              keepdims=True)
        m_tmp = tf.add(m_sp[:, None, :], - m_out)
        S_tmp = tf.add(S_sp, tf.matmul(m_tmp, m_tmp, transpose_a=True))
        S_out = tf.reduce_sum(tf.multiply(self.weights[:, None, None],
                                          S_tmp),
                              axis=0)
        C_tmp = tf.matmul(tf.add(sigma_points[:, None, :], -m_in),
                          m_tmp,
                          transpose_a=True)
        C = tf.reduce_sum(tf.multiply(self.weights[:, None, None],
                                      C_tmp),
                          axis=0)

        S_out = 0.5 * tf.add(S_out, tf.transpose(S_out))

        return m_out, S_out, C

    @staticmethod
    def get_sigma_points(m, S):
        e, V = tf.linalg.eigh(S)
        R = tf.matmul(V, tf.linalg.diag(tf.sqrt(e)))
        pref = tf.sqrt(tf.constant(3., dtype=float_type))
        sigma_points = tf.concat((m,
                                  tf.add(m, tf.multiply(pref, R)),
                                  tf.add(m, tf.multiply(-1. * pref, R))),
                                 axis=0)
        return sigma_points


class GPPF(NoisyBaseClass):
    @type_checked
    def __init__(self, predictor, input_dims, samples=None, fixed_lhs=True):
        super().__init__(X=None, Y=None, is_ctrl=False)
        self.predictor = predictor
        self.num_dims = input_dims
        if samples is None:
            self.samples = 10 * input_dims
        else:
            self.samples = samples
        lhsamples = lhs(input_dims, self.samples)
        self.lhsamples = norm(0, 1).ppf(lhsamples)
        self.fixed_lhs = fixed_lhs
        self.noise = 1e-5

    def noisy_predict(self, m_in, S_in):
        R = tf.linalg.cholesky(S_in, name='PF_choleksy')
        if self.fixed_lhs:
            samples = tf.transpose(tf.matmul(R, self.lhsamples, transpose_b=True)) + m_in
        else:
            samples = tf.random.normal(shape=(self.samples, self.num_dims), dtype=float_type)
            samples = tf.transpose(tf.matmul(R, samples, transpose_b=True)) + m_in

        m_sp, S_sp = self.predictor.predict(samples)

        m_out = tf.reduce_mean(m_sp,
                               axis=0,
                               keepdims=True)
        m_tmp = tf.add(m_sp[:, None, :], - m_out)
        S_tmp = tf.add(S_sp, tf.matmul(m_tmp, m_tmp, transpose_a=True))
        S_out = tf.reduce_mean(S_tmp, axis=0)
        C_tmp = tf.matmul(tf.add(samples[:, None, :], -m_in),
                          m_tmp,
                          transpose_a=True)
        C = tf.reduce_mean(C_tmp, axis=0)

        S_out = 0.5 * tf.add(S_out, tf.transpose(S_out))
        return m_out, S_out, C

class GPEKF(NoisyBaseClass):
    @type_checked
    def __init__(self, predictor, input_dims, use_predictor_grad=False):
        super().__init__(X=None, Y=None, is_ctrl=False)
        self.predictor = predictor
        self.noise = 1e-5
        self.use_predictor_grad = use_predictor_grad

    def noisy_predict(self, m_in, S_in):
        if self.use_predictor_grad:
            m_out, S = self.predictor.predict(m_in)
            J = self.predictor.predict_grad(m_in)[0, :, :]
        else:
            with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
                tape.watch(m_in)
                m_out, S = self.predictor.predict(m_in)
            # See https://github.com/tensorflow/tensorflow/issues/32460 for understanding the next line
            J = tape.batch_jacobian(m_out, m_in,
                                    experimental_use_pfor=False)[0, :, :]

        C = tf.matmul(S_in, J, transpose_b=True)
        S_out = tf.matmul(J, C) + S[0, :, :]

        S_out = 0.5 * tf.add(S_out, tf.transpose(S_out))

        return m_out, S_out, C
