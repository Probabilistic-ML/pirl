import abc
import numpy as np

import tensorflow as tf

import gpflow

from ..models.policy_models import RBFPolicy, BatchRBFPolicy, GPflowPolicy, NNPolicy, BatchNNPolicy, NonParamPolicy
from ..uncertainty.noisy_predict import MomentMatching, GPEKF
from ..misc import type_checked
from ..config import float_type, ABSTRACT_ERRMSG

from .bounders import trapezoidal_squash, custom_bounder


class BasicController(abc.ABC):
    def __init__(self, dims, population_size,
                 control_lb=None, control_ub=None,
                 bound_func=trapezoidal_squash):
        self.state_dim = dims[0]
        self.control_dim = dims[1]
        self.pop_size = population_size
        if [control_lb, control_ub].count(None) == 1:
            raise ValueError("One sided bounds are not supported")
        self.control_lb = _safe_get_action_bound(control_lb, self.control_dim)
        self.control_ub = _safe_get_action_bound(control_ub, self.control_dim)
        if self.is_bounded and tf.reduce_min(self.control_ub - self.control_lb) <= 0:
            msg = "Lower bound must be strictly smaller that "
            msg += "the upper bound for all control dimensions!"
            raise ValueError(msg)
        self.bounder = bound_func
        self.specs = {"name": self.__class__.__name__,
                      "dims": dims,
                      "pop_size": population_size,
                      "ctrl_bounds": (control_lb, control_ub),
                      "bound_func": str(bound_func)}

    @property
    def is_bounded(self):
        """ True if bounds were passed, false otherwise """
        if self.control_lb is None and self.control_ub is None:
            return False
        return True

    def bound_control(self, ctrl_mu, ctrl_std=None):
        return self.bounder(ctrl_mu, s=ctrl_std,
                            max_action=self.control_ub,
                            min_action=self.control_lb)

    @type_checked
    def compute_action(self, state_mu, state_var=None):
        if state_var is None:
            ctrl_mean, _ = self.model.predict(state_mu)
            return ctrl_mean
        return self._compute_action(tf.reshape(state_mu, (1, -1)), state_var)

    def _compute_action(self, state_mu, state_var):
        predictor = self.predictor_cls(self.model, self.state_dim)
        return predictor.noisy_predict(state_mu, state_var)

    def randomize(self):
        self._randomize()
        self.model.update_cov()

    @abc.abstractmethod
    def _randomize(self):
        raise NotImplementedError(ABSTRACT_ERRMSG)

    @classmethod
    def to_initer(cls, **kwargs):
        def initer(dims, population_size):
            return cls(dims, population_size, **kwargs)

        return initer


class StaticController(BasicController):
    def __init__(self, dims, population_size, u, tensor=False):
        super(StaticController, self).__init__(dims, population_size)
        if tensor:
            self.u = tf.cast(u, dtype=float_type)
        else:
            self.u = np.array(u).ravel()
        self.count = 0
        self.specs["u"] = u

    def compute_action(self, state):
        u = self.u[self.count]
        self.count = self.count + 1
        return u

    def reset(self):
        self.count = 0

    def _randomize(self):
        pass


class NonParamController(BasicController):
    def __init__(self, dims, population_size, horizon,
                 control_lb=None, control_ub=None, std_x=4,
                 bound_func=custom_bounder(tf.tanh)):
        super(NonParamController, self).__init__(dims, population_size,
                                                 control_lb=control_lb,
                                                 control_ub=control_ub,
                                                 bound_func=custom_bounder(lambda x: x))
        self.horizon = horizon
        self.std_x = std_x
        u_train = self._get_fake_data()
        self.model = NonParamPolicy(u_train)

    def _get_fake_data(self):
        if self.pop_size == 0:
            u_train = _get_fake_x((self.horizon, self.control_dim),
                                  self.std_x)
        else:
            u_train = _get_fake_x((self.pop_size, self.horizon, self.control_dim),
                                  self.std_x)
        return u_train

    @type_checked
    def compute_action(self, state, j=0):
        if self.pop_size == 0:
            return self._compute_action(state, j)
        else:
            state = tf.stack(self.pop_size * [state])
            return self._compute_action(state, j)[0, :, :]  # first controller should be best controller

    def _compute_action(self, state, j=0):
        ctrl = self.model.predict(state, j)
        if self.is_bounded:
            return self.bound_control(ctrl)
        return ctrl

    def reset(self):
        self.count = 0

    def _randomize(self):
        self.model.u.assign(self._get_fake_data())


class GPBasicController(BasicController):
    def __init__(self, dims, population_size,
                 control_lb=None, control_ub=None,
                 num_basis_functions=10, bound_func=trapezoidal_squash,
                 std_x=4, std_y=0.5):
        super(GPBasicController, self).__init__(dims,
                                                population_size,
                                                control_lb, control_ub,
                                                bound_func)
        self.n_bf = num_basis_functions
        self.std_x = std_x
        self.std_y = std_y
        self.specs["num_basis_functions"] = num_basis_functions
        self.specs["std"] = (std_x, std_y)

    def _get_fake_data(self):
        X_train = _get_fake_x((self.pop_size, self.n_bf, self.state_dim),
                              self.std_x)
        Y_train = _get_fake_y((self.pop_size, self.n_bf, self.control_dim),
                              self.std_y)
        ls_train = _get_fake_ls((self.pop_size, self.control_dim, self.state_dim))
        return X_train, Y_train, ls_train

    def _randomize(self):
        """ Randomize model data to search a new controller """
        self.model.x_train.assign(_get_fake_x((self.pop_size, self.n_bf, self.state_dim),
                                              self.std_x))
        self.model.y_train.assign(_get_fake_y((self.pop_size, self.n_bf, self.control_dim),
                                              self.std_y))
        self.model.lengthscales.assign(_get_fake_ls((self.pop_size, self.control_dim, self.state_dim)))


class RbfMMController(GPBasicController):
    """ An RBF Controller implemented as a deterministic GP
            Deisenroth et al 2015: Gaussian Processes for Data-Efficient Learning in Robotics and Control
            Section 5.3.2.

        Parameters
        ----------
        state_dim : int
            number of state dimensions
        control_dim : int
            number of control dimensions
        num_basis_functions : int
            number of basis functions aka samples of BeliefGP
        predictor_cls : class, optional
            A class from noisy_predict module, by default MomentMatching
        max_action : float or list or np.ndarray, optional
            Bounds for the maximum (and negative minimum) action, by default None
        min_action : float or list or np.ndarray, optional
            Bounds for the maximum (and negative minimum) action, by default None
        """

    def __init__(self, dims, population_size,
                 control_lb=None, control_ub=None,
                 num_basis_functions=10, predictor_cls=MomentMatching,
                 bound_func=trapezoidal_squash, std_x=4, std_y=0.5):
        super(RbfMMController, self).__init__(dims, population_size,
                                              control_lb, control_ub,
                                              num_basis_functions,
                                              bound_func, std_x, std_y)
        X_train, Y_train, ls_train = self._get_fake_data()
        self.model = RBFPolicy(X_train, Y_train, ls_train)
        self.predictor_cls = predictor_cls
        self.specs["model"] = str(self.model)
        self.specs["predictor_cls"] = str(predictor_cls)

    def compute_action(self, state_mu, state_var=None):
        """computes action given the state prediction

        Parameters
        ----------
        state_mu : tensor or array-like, shape=(n_state,)
            Mean prediction of the dynamic state models
        state_var : tensor or array-like, shape=(n_state, n_state)
            Prediction variance of the dynamic state models
        squash : bool, optional
            if True, computed action will be squashed between the given
            bounds by default True

        Returns
        -------
        ctrl_mean : tensor, shape=(n_ctrl,)
            Mean of the predicted action
        ctrl_var : tensor, shape=(n_ctrl, n_ctrl)
            Variance of the predicted action (only if state_var is not None)
        ctrl_V : tensor
        """
        if state_var is None:
            ctrl_mu = self.model.predict(state_mu)
            if self.is_bounded:
                return self.bound_control(ctrl_mu)
            return ctrl_mu
        return self._compute_action(tf.reshape(state_mu, (1, -1)), state_var)

    def _compute_action(self, state_mu, state_var):
        predictor = self.predictor_cls(self.model.x_train, self.model.y_train,
                                       self.model._inv_cov, self.model.lengthscales, is_ctrl=True)
        ctrl_mean, ctrl_var, ctrl_V = predictor.noisy_predict(state_mu, state_var)
        ctrl_var = ctrl_var - tf.linalg.diag(predictor.kernel_var - 1e-6)
        if self.is_bounded:  # if bounds passed, then squash...
            ctrl_mean, ctrl_var, ctrl_V2 = self.bound_control(ctrl_mean, ctrl_var)
            ctrl_V = ctrl_V @ ctrl_V2
        return ctrl_mean, ctrl_var, ctrl_V


class RbfController(GPBasicController):
    """ An RBF Controller implemented as a deterministic GP
            Deisenroth et al 2015: Gaussian Processes for Data-Efficient Learning in Robotics and Control
            Section 5.3.2.

        Parameters
        ----------
        state_dim : int
            number of state dimensions
        control_dim : int
            number of control dimensions
        num_basis_functions : int
            number of basis functions aka samples of BeliefGP
        predictor_cls : class, optional
            A class from noisy_predict module, by default MomentMatching
        max_action : float or list or np.ndarray, optional
            Bounds for the maximum (and negative minimum) action, by default None
        min_action : float or list or np.ndarray, optional
            Bounds for the maximum (and negative minimum) action, by default None
        """

    def __init__(self, dims, population_size, control_lb=None, control_ub=None,
                 num_basis_functions=10, bound_func=trapezoidal_squash,
                 std_x=4, std_y=0.5):
        super(RbfController, self).__init__(dims, population_size,
                                            control_lb, control_ub,
                                            num_basis_functions,
                                            bound_func, std_x, std_y)
        X_train, Y_train, ls_train = self._get_fake_data()
        if self.pop_size == 0:
            self.model = RBFPolicy(X_train, Y_train, ls_train)
        else:
            self.model = BatchRBFPolicy(X_train, Y_train, ls_train)
        self.specs["model"] = str(self.model)

    @type_checked
    def compute_action(self, state):
        if self.pop_size == 0:
            return self._compute_action(state)
        else:
            state = tf.stack(self.pop_size * [state])
            return self._compute_action(state)[0, :, :]  # first controller should be best controller

    def _compute_action(self, state):
        ctrl = self.model.predict(state)
        if self.is_bounded:
            return self.bound_control(ctrl)
        return ctrl


class GPflowController(GPBasicController):
    def __init__(self, dims, population_size, control_lb=None, control_ub=None,
                 num_basis_functions=10, kernel=gpflow.kernels.Matern32,
                 predictor_cls=GPEKF, bound_func=trapezoidal_squash,
                 std_x=4, std_y=0.5):
        super(GPflowController, self).__init__(dims, population_size,
                                               control_lb, control_ub,
                                               num_basis_functions,
                                               bound_func, std_x, std_y)
        X_train, Y_train, ls_train = self._get_fake_data()
        bounder = None
        if self.is_bounded:
            bounder = self.bound_control
        self.model = GPflowPolicy(X_train, Y_train, ls_train,
                                  kernel=kernel, bounder=bounder)
        self.specs["model"] = str(self.model)


class NNController(BasicController):
    def __init__(self, dims, population_size, nn_specs=None,
                 control_lb=None, control_ub=None):
        super(NNController, self).__init__(dims, population_size,
                                           control_lb, control_ub,
                                           bound_func=custom_bounder(lambda x: x))
        if nn_specs is None:
            self.nn_specs = {"units": (16, 1),
                             "activations": ('relu', 'tanh')}
        else:
            self.nn_specs = nn_specs
        self.model = NNPolicy(dims, self.nn_specs)
        self.specs["model"] = str(self.model)
        self.specs["nn_specs"] = self.nn_specs

    def _randomize(self):
        for p in self.model.trainable_variables:
            if 'bias' in p.name:
                p.assign(tf.zeros_like(p))
            else:
                p.assign(_get_fake_x(tf.shape(p), std=0.5))

    @type_checked
    def compute_action(self, state):
        if self.pop_size == 0:
            return self._compute_action(state)
        else:
            state = tf.stack(self.pop_size * [state])
            return self._compute_action(state)[0, :, :]  # first controller should be best controller

    def _compute_action(self, state):
        ctrl = self.model.predict(state)
        if self.is_bounded:
            return self.bound_control(ctrl)
        return ctrl


class DiscreteNNController(BasicController):
    def __init__(self, dims, population_size, num_actions,
                 action_decoder=None, nn_specs=None):
        super(DiscreteNNController, self).__init__(dims, population_size,
                                                   control_lb=None, control_ub=None)
        if nn_specs is None:
            self.nn_specs = {"units": (16, num_actions),
                             "activations": ('relu', 'softmax')}
        else:
            self.nn_specs = nn_specs
        self.num_actions = num_actions
        if self.pop_size == 0:
            self.model = NNPolicy(dims, self.nn_specs, num_actions)
        else:
            self.model = BatchNNPolicy(dims, self.pop_size, self.nn_specs, num_actions)
        self.action_decoder = action_decoder
        self.specs["model"] = str(self.model)
        self.specs["nn_specs"] = self.nn_specs
        self.specs["action_decoder"] = str(action_decoder)

    def _randomize(self):
        for p in self.model.trainable_variables:
            if 'bias' in p.name:
                p.assign(tf.zeros_like(p))
            else:
                p.assign(_get_fake_x(tf.shape(p), std=0.1))

    @type_checked
    def compute_action(self, state):
        if self.pop_size == 0:
            return self._compute_action(state)
        else:
            state = tf.stack(self.pop_size * [state])
            return self._compute_action(state)[0, :, :]  # first controller should be best controller

    def _compute_action(self, state):
        ctrl = self.model.predict(state)
        if self.pop_size == 0:
            ctrl = tf.random.categorical(tf.math.log(ctrl), 1)
        else:
            ctrl_tmp = tf.reshape(ctrl, shape=(-1, self.num_actions))
            ctrl_tmp = tf.random.categorical(tf.math.log(ctrl_tmp), 1)
            ctrl = tf.reshape(ctrl_tmp, shape=(self.pop_size, -1, 1))
        if self.action_decoder is None:
            return tf.cast(ctrl, dtype=float_type)
        return self.action_decoder(ctrl)


class PseudoDiscreteNNController(DiscreteNNController):
    @type_checked
    def compute_action(self, state):
        ctrl = self._compute_action(state)
        return tf.math.round(ctrl)

    def _compute_action(self, state):
        ctrl = self.model.predict(state)
        ctrl = tf.tensordot(tf.cast(ctrl, dtype=float_type),
                            tf.reshape(tf.range(self.num_actions, dtype=float_type), (-1, 1)),
                            axes=1)
        return ctrl


def _safe_get_action_bound(bound, control_dim):
    control_dim = int(control_dim)
    if bound is None:
        return None
    if isinstance(bound, (int, float)):
        bound = [bound] * control_dim
    try:
        bound = np.array(bound).ravel()
    except (ValueError, TypeError) as exc:
        raise ValueError("Action bound not understood") from exc
    assert bound.size == control_dim, "Inconsistent bounds"
    return tf.cast(bound, dtype=float_type)


def _get_fake_x(shape, std=4.):
    if shape[0] == 0:
        return std * tf.random.normal(shape[1:], dtype=float_type)
    return std * tf.random.normal(shape, dtype=float_type)


def _get_fake_y(shape, std=0.5):
    if shape[0] == 0:
        return std * tf.random.normal(shape[1:], dtype=float_type)
    return std * tf.random.normal(shape, dtype=float_type)


def _get_fake_ls(shape):
    if shape[0] == 0:
        return 1 + 0.1 * tf.random.normal(shape[1:], dtype=float_type)
    return 1 + 0.1 * tf.random.normal(shape, dtype=float_type)
