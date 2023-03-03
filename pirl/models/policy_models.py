import abc
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import gpflow
from ..config import float_type, ABSTRACT_ERRMSG


class BasePolicy(tf.Module):
    def __init__(self, namespace, x_init, y_init):
        super(BasePolicy, self).__init__(name=namespace)
        self.x_train = tf.Variable(x_init, trainable=True, name='x_train', dtype=float_type)
        self.y_train = tf.Variable(y_init, trainable=True, name='y_train', dtype=float_type)
        num_dims = self.x_train.shape[-1]
        num_datapoints = self.x_train.shape[-2]
        num_outputs = self.y_train.shape[-1]
        self.dims = {"inp": num_dims,
                     "out": num_outputs,
                     "samp": num_datapoints}

    @abc.abstractmethod
    def update_cov(self):
        raise NotImplementedError(ABSTRACT_ERRMSG)


class KernelPolicy(BasePolicy):
    """ A GP, for which the data is also found through optimization besides the lenghtscales """

    def __init__(self, namespace, x_init, y_init, ls_init):
        super(KernelPolicy, self).__init__(namespace, x_init, y_init)
        self.lengthscales = tf.Variable(ls_init,
                                        name='lengthscales',
                                        trainable=True, dtype=float_type)
        self.noise = tf.Variable(1e-5, dtype=float_type, trainable=False, name='noise')  # 1e-4

    def covariance(self, x_test):
        x1 = tf.divide(x_test[None, :, :], self.lengthscales[:, None, :])
        x2 = tf.divide(self.x_train[None, :, :], self.lengthscales[:, None, :])
        return self.kernel(x1, x2)

    def _get_inv_cov(self):
        covar = self.covariance(self.x_train)
        covar = tf.add(covar, tf.multiply(self.noise, tf.eye(self.dims["samp"],
                                                             batch_shape=[self.dims["out"]],
                                                             dtype=float_type)
                                          )
                       )
        covar_cho = tf.linalg.cholesky(covar, name="belief_gp_get_covar_cho")
        inv_cov = tf.linalg.cholesky_solve(covar_cho,
                                           tf.eye(self.dims["samp"],
                                                  batch_shape=[self.dims["out"]],
                                                  dtype=float_type),
                                           name="belief_gp_get_inv_cov")
        return inv_cov

    def update_cov(self):
        self._inv_cov.assign(self._get_inv_cov())


class RBFPolicy(KernelPolicy):
    def __init__(self, x_init, y_init, ls_init):
        super(RBFPolicy, self).__init__("RBFPolicy", x_init, y_init, ls_init)
        self._inv_cov = tf.Variable(self._get_inv_cov(),
                                    dtype=float_type, trainable=False,
                                    name='_inv_cov')

    @staticmethod
    def kernel(x1, x2):
        return tf.exp(-0.5 * square_distance(x1, x2))

    def predict(self, x_test):
        covar_xtestx = self.covariance(x_test)
        beta = tf.matmul(self._inv_cov, tf.transpose(self.y_train)[:, :, None])
        return tf.transpose(tf.matmul(covar_xtestx, beta)[:, :, 0])


class GPflowPolicy(KernelPolicy):
    def __init__(self, x_init, y_init, ls_init,
                 kernel=gpflow.kernels.SquaredExponential, bounder=None):
        super(GPflowPolicy, self).__init__("GPPolicy", x_init, y_init, ls_init)
        gpflow.config.set_default_float(float_type)
        if float_type == tf.float32:
            self.var = tf.constant(self.noise * tf.eye(self.dims["out"], dtype=float_type))
        else:
            self.var = tf.constant(tf.zeros([self.dims["out"], self.dims["out"]], dtype=float_type))
        self.bounder = bounder

        try:
            self.kernel = kernel(lengthscales=tf.ones([self.dims["inp"], ], dtype=float_type),
                                 variance=1.0)
        except AttributeError:
            self.kernel = kernel(lengthscales=tf.ones([self.dims["inp"], ], dtype=float_type))
        else:
            self.kernel = kernel()
        gpflow.set_trainable(self.kernel, False)
        self._inv_cov = tf.Variable(self._get_inv_cov(),
                                    dtype=float_type, trainable=False,
                                    name='_inv_cov')

    def predict(self, x_test):
        covar_xtestx = self.covariance(x_test)
        beta = tf.matmul(self._inv_cov, tf.transpose(self.y_train)[:, :, None])
        m = tf.transpose(tf.matmul(covar_xtestx, beta)[:, :, 0])
        S = tf.tile(self.var[None, :, :], [tf.shape(x_test)[0], 1, 1])
        if self.bounder is None:
            return m, S
        return self.bounder(m), S


class BatchKernelPolicy(BasePolicy):
    def __init__(self, namespace, x_init, y_init, ls_init):
        super(BatchKernelPolicy, self).__init__(namespace, x_init, y_init)
        self.lengthscales = tf.Variable(ls_init,
                                        name='lengthscales',
                                        trainable=True, dtype=float_type)
        self.pop_size = self.x_train.shape[0]
        self.noise = tf.Variable(1e-5, dtype=float_type, trainable=False, name='noise')  # 1e-4

    def covariance(self, x_test):
        x1 = tf.divide(x_test[:, None, :, :], self.lengthscales[:, :, None, :])
        x2 = tf.divide(self.x_train[:, None, :, :], self.lengthscales[:, :, None, :])
        return self.kernel(x1, x2)

    def _get_inv_cov(self):
        covar = self.covariance(self.x_train)
        covar = tf.add(covar, tf.multiply(self.noise, tf.eye(self.dims["samp"],
                                                             batch_shape=[self.pop_size, self.dims["out"]],
                                                             dtype=float_type)
                                          )
                       )
        covar_cho = tf.linalg.cholesky(covar, name="belief_gp_get_covar_cho")
        inv_cov = tf.linalg.cholesky_solve(covar_cho,
                                           tf.eye(self.dims["samp"],
                                                  batch_shape=[self.pop_size, self.dims["out"]],
                                                  dtype=float_type),
                                           name="belief_gp_get_inv_cov")
        return inv_cov

    def update_cov(self):
        self._inv_cov.assign(self._get_inv_cov())


class BatchRBFPolicy(BatchKernelPolicy):
    def __init__(self, x_init, y_init, ls_init):
        super(BatchRBFPolicy, self).__init__("RBFPolicy", x_init, y_init, ls_init)
        self._inv_cov = tf.Variable(self._get_inv_cov(),
                                    dtype=float_type, trainable=False,
                                    name='_inv_cov')

    @staticmethod
    def kernel(x1, x2):
        return tf.exp(-0.5 * batch_square_distance(x1, x2))

    def predict(self, x_test):
        covar_xtestx = self.covariance(x_test)
        beta = tf.matmul(self._inv_cov,
                         tf.transpose(self.y_train,
                                      perm=(0, 2, 1))[:, :, :, None])
        return tf.transpose(tf.matmul(covar_xtestx, beta)[:, :, :, 0], perm=(0, 2, 1))


class NNPolicy:
    def __init__(self, dims, nn_specs, num_actions=None):
        self.state_dim = dims[0]
        self.control_dim = dims[1]
        reg1 = None
        reg2 = None
        if self.control_dim == 1 and num_actions is not None:  # discrete actions
            self.num_outputs = num_actions
        elif num_actions is None:  # continuous actions
            self.num_outputs = self.control_dim
            reg1 = regularizers.l2(5e-3)
            reg2 = regularizers.l2(5e-3)
        units = nn_specs["units"]
        acts = nn_specs["activations"]
        nn_layers = []
        for u, a in zip(units[:-1], acts[:-1]):
            nn_layers.append(layers.Dense(u, activation=a))
        nn_layers.append(layers.Dense(self.num_outputs,
                                      kernel_regularizer=reg1,
                                      bias_regularizer=reg2,
                                      activation=acts[-1]))
        self.nn = keras.Sequential(nn_layers)
        self.nn.build(input_shape=(None, self.state_dim))

    def predict(self, x_test):
        return self.nn(x_test)

    @property
    def trainable_variables(self):
        return self.nn.trainable_variables

    def update_cov(self):
        pass


class BatchNNPolicy:
    def __init__(self, dims, pop_size, nn_specs, num_actions=None):
        self.state_dim = dims[0]
        self.control_dim = dims[1]
        self.pop_size = pop_size
        if self.control_dim == 1 and num_actions is not None:  # discrete actions
            self.num_outputs = num_actions
        elif num_actions is None:  # continuous actions
            self.num_outputs = self.control_dim
        units = nn_specs["units"]
        acts = nn_specs["activations"]
        input_layer = layers.Input(shape=(self.pop_size, self.state_dim))
        outputs = []
        for i in range(self.pop_size):
            x = layers.Lambda(lambda y: y[:, 1, :])(input_layer)
            for u, a in zip(units[:-1], acts[:-1]):
                x = layers.Dense(u, activation=a)(x)
            outputs.append(layers.Dense(self.num_outputs)(x)[:, None, :])
        outputs = layers.concatenate(outputs, axis=1)
        self.nn = keras.Model(inputs=[input_layer], outputs=[outputs])

    def predict(self, x_test):
        x_test = tf.transpose(x_test, perm=(1, 0, 2))
        return tf.transpose(self.nn(x_test), perm=(1, 0, 2))

    @property
    def trainable_variables(self):
        return self.nn.trainable_variables

    def update_cov(self):
        pass


class NonParamPolicy(tf.Module):
    def __init__(self, u):
        super().__init__(name="NoneParamPolicy")
        self.u = tf.Variable(u, trainable=True, dtype=float_type)
        self.count = 0

    def predict(self, x_test):
        n_samples = tf.shape(x_test)[0]
        u_tmp = self.u[self.count:self.count + 1, :]
        u = tf.repeat(u_tmp, repeats=n_samples, axis=0)
        self.count = self.count + 1
        return u

    def update_cov(self):
        self.count = 0


def square_distance(X, X2=None):  # similarly to square_distance from GPflow
    if X2 is None:
        Xs = tf.reduce_sum(tf.square(X), axis=-1, keepdims=True)
        dist = tf.add(Xs, tf.linalg.adjoint(Xs))
        dist = tf.add(dist, - 2 * tf.matmul(X, tf.transpose(X, perm=[0, 2, 1])))
        return dist
    Xs = tf.reduce_sum(tf.square(X), axis=-1, keepdims=True)
    X2s = tf.reduce_sum(tf.square(X2), axis=-1, keepdims=True)
    dist = tf.add(Xs, tf.linalg.adjoint(X2s))
    dist = tf.add(dist, - 2 * tf.matmul(X, tf.transpose(X2, perm=[0, 2, 1])))
    return dist


def batch_square_distance(X, X2=None):
    if X2 is None:
        Xs = tf.reduce_sum(tf.square(X), axis=-1, keepdims=True)
        dist = tf.add(Xs, tf.linalg.adjoint(Xs))
        dist = tf.add(dist, - 2 * tf.matmul(X, tf.transpose(X, perm=[0, 1, 3, 2])))
        return dist
    Xs = tf.reduce_sum(tf.square(X), axis=-1, keepdims=True)
    X2s = tf.reduce_sum(tf.square(X2), axis=-1, keepdims=True)
    dist = tf.add(Xs, tf.linalg.adjoint(X2s))
    dist = tf.add(dist, - 2 * tf.matmul(X, tf.transpose(X2, perm=[0, 1, 3, 2])))
    return dist
