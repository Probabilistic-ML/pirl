import abc
import numpy as np
import tensorflow as tf
from scipy.stats import norm

from ..misc import type_checked, lhs
from ..rewards.base import TransitionReward, CombinedReward
from ..config import float_type, ABSTRACT_ERRMSG


class ProbabilisticController(abc.ABC):
    def __init__(self, controller, dyn_model, reward, reward_calc, transforms,
                 m_init, S_init, horizon):
        self.controller = controller
        self.dyn_model = dyn_model
        self.reward = reward
        if isinstance(self.reward, CombinedReward):
            self.transition = 2 * int(isinstance(reward.base_rewards[0], TransitionReward))
        else:
            self.transition = int(isinstance(self.reward, TransitionReward))

        self.transforms = transforms
        self.horizon = tf.constant(horizon, dtype=tf.int32)
        self.m_init = m_init
        self.S_init = S_init
        self.latent_dim = self.controller.state_dim
        self.pop_size = self.controller.pop_size

        self._error_reward = tf.convert_to_tensor([[-np.inf]], dtype=float_type)

        if reward_calc is None:
            self.reward_calc = mean_reward
        elif isinstance(reward_calc, list):
            self.reward_calc = get_reward_calc(reward_calc)
        elif callable(reward_calc):
            self.reward_calc = reward_calc
        else:
            msg = "reward_calc is neither callable nor a list"
            raise ValueError(msg)

        if float_type == tf.float64:
            self.eigenvalue_bound = tf.constant(1e-12, dtype=float_type)
        else:
            self.eigenvalue_bound = tf.constant(1e-8, dtype=float_type)
        self.eps = tf.constant(1e-12, dtype=float_type)

    @abc.abstractmethod
    def predict(self, *args):
        raise NotImplementedError(ABSTRACT_ERRMSG)

    @abc.abstractmethod
    def _predict(self, *args):
        raise NotImplementedError(ABSTRACT_ERRMSG)

    @abc.abstractmethod
    def compute_reward(self):
        raise NotImplementedError(ABSTRACT_ERRMSG)

    @abc.abstractmethod
    def _compute_reward(self, m_x, S_x):
        raise NotImplementedError(ABSTRACT_ERRMSG)

    def loss(self):
        total_reward, _, _, _ = self._compute_reward(self.m_init, self.S_init)
        if self.controller.__class__.__name__ in ['NNController']:
            return -1. * total_reward + tf.reduce_sum(self.controller.model.nn.losses)
        return -1. * total_reward

    def train_step(self):
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            self.controller.model.update_cov()
            loss_value = self.loss()
        grads = tape.gradient(loss_value, self.trainable_variables)
        return loss_value, grads

    def compute_action(self, x):
        x = tf.cast(x, dtype=float_type)
        return self.controller.compute_action(self.transforms["state"].transform(x))

    def randomize(self):
        self.controller.randomize()

    @property
    def trainable_variables(self):
        return self.controller.model.trainable_variables

    @property
    def params(self):
        return [param.numpy() for param in self.controller.model.trainable_variables]

    @params.setter
    def params(self, new_params):
        for i, param in enumerate(self.controller.model.trainable_variables):
            param.assign(new_params[i])
        self.controller.model.update_cov()

    def patch_up(self, S):
        e, V = tf.linalg.eigh(S)
        e = tf.maximum(e, self.eps)
        return tf.matmul(tf.matmul(V, tf.linalg.diag(e), transpose_a=True), V)


class DensityController(ProbabilisticController):
    @type_checked
    def predict(self, m_x, S_x):
        m_x = self.transforms["state"].transform(m_x)
        S_x = self.transforms["state"].transform_var(S_x)
        m_x, S_x, m_u, S_u = self._predict(m_x, S_x)
        m_x = self.transforms["state"].inverse_transform(m_x)
        S_x = self.transforms["state"].inverse_transform_var(S_x)
        return m_x.numpy(), S_x.numpy(), m_u.numpy(), S_u.numpy()

    def _predict(self, m_x, S_x):
        # unnormalized control distribution
        m_u, S_u, S_xu = self.controller._compute_action(m_x, S_x)
        # normalize
        m_u_scaled = self.transforms["ctrl"].transform(m_u)
        S_u_scaled = self.transforms["ctrl"].transform_var(S_u)
        S_xu = self.transforms["ctrl"].transform_std(S_xu)

        m = tf.concat([m_x, m_u_scaled], axis=1)

        if 'MomentMatching' in str(self.controller.predictor_cls):
            S1 = tf.concat([S_x, S_x @ S_xu], axis=1)
            S2 = tf.concat([tf.transpose(S_x @ S_xu), S_u_scaled], axis=1)
            S = tf.concat([S1, S2], axis=0)
        else:
            S1 = tf.concat([S_x, S_xu], axis=1)
            S2 = tf.concat([tf.transpose(S_xu), S_u_scaled], axis=1)
            S = tf.concat([S1, S2], axis=0)
        e = tf.linalg.eigvalsh(S, name='eigvalsh_xu')
        pred = tf.reduce_any(tf.less(e, self.eigenvalue_bound))
        S = tf.cond(pred, lambda: self.patch_up(S), lambda: S)

        m_dx, S_dx, S_xudx = self.dyn_model.noisy_predict(m, S)

        m_x = m_x + m_dx  # (10)
        if 'MomentMatching' in str(self.dyn_model):
            S_xdx = tf.matmul(S1, S_xudx)
        else:
            S_xdx = S_xudx[:self.latent_dim, :self.latent_dim]
        S_x = S_dx + S_x + S_xdx + tf.transpose(S_xdx)

        e = tf.linalg.eigvalsh(S_x, name='eigvalsh_x')
        pred = tf.reduce_any(tf.less(e, self.eigenvalue_bound))
        S_x = tf.cond(pred, lambda: self.patch_up(S_x), lambda: S_x)

        m_x.set_shape([1, self.latent_dim])
        S_x.set_shape([self.latent_dim, self.latent_dim])

        return m_x, S_x, m_u, S_u

    @type_checked
    def compute_reward(self):
        """

        Berechnung des erwarteten Rewards über gesamten Zeitraum T
        (ehemals predict)
        Schleife von 0 bis T-1
        in jedem Schritt Anwendung von predict und reward.compute_reward

        Parameters
        ----------
        m_x - tf.tensor(shape=(1, n_state))
            Mean vector of state dimensions
        S_x - tf.tensor(shape=(n_state, n_state))
            Covariance of state dimensions
        disc - float
            Discount Faktor (0 < disc <= 1)

        Returns
        -------
        R - tf.tensor(shape=(1,1))
            Estimated Reward
        """
        total_reward, sm, sv, r = self._compute_reward(self.m_init, self.S_init)
        state_means = self.transforms["state"].inverse_transform(sm.stack())
        state_means = state_means.numpy()[:, 0, :]
        state_vars = self.transforms["state"].inverse_transform_var(sv.stack())
        state_vars = state_vars.numpy()
        rewards = r.stack().numpy()[:, 0, :]
        return total_reward, state_means, state_vars, rewards

    def _loop_cond(self, j, state_means, state_vars, rewards):
        return tf.less(j, self.horizon)

    def _loop_body(self, j, state_means, state_vars, rewards):
        m_x, S_x, m_u, S_u = self._predict(state_means.read(j), state_vars.read(j))
        t = tf.cast(tf.divide(j + 1, self.horizon), dtype=float_type)
        r = self.reward.compute_reward(m_x, S_x, m_u, S_u, t)

        state_means = state_means.write(j + 1, m_x)
        state_vars = state_vars.write(j + 1, S_x)
        rewards = rewards.write(j, r)

        return j + 1, state_means, state_vars, rewards

    def _compute_reward(self, m_x, S_x):
        """
        Implementation of compute_reward

        Berechnung des erwarteten Rewards über gesamten Zeitraum T
        (ehemals predict)
        Schleife von 0 bis T-1
        in jedem Schritt Anwendung von predict und reward.compute_reward


        """

        state_means = tf.TensorArray(float_type, size=self.horizon + 1,
                                     clear_after_read=False,
                                     tensor_array_name='state_means')
        state_vars = tf.TensorArray(float_type, size=self.horizon + 1,
                                    clear_after_read=False,
                                    tensor_array_name='state_vars')
        rewards = tf.TensorArray(float_type, size=self.horizon,
                                 clear_after_read=False,
                                 tensor_array_name='rewards')

        state_means = state_means.write(0, self.transforms["state"].transform(m_x))
        state_vars = state_vars.write(0, self.transforms["state"].transform_var(S_x))

        loop_vars = [tf.constant(0, dtype=tf.int32, name="timesteps"),
                     state_means,
                     state_vars,
                     rewards]

        _, state_means, state_vars, rewards = tf.while_loop(self._loop_cond,
                                                            self._loop_body,
                                                            loop_vars)
        return self.reward_calc(rewards.stack()), state_means, state_vars, rewards


class SamplingController(ProbabilisticController):
    def __init__(self, controller, dyn_model, reward, reward_calc, transforms,
                 m_init, S_init, horizon, n_samples=50):
        super(SamplingController, self).__init__(controller, dyn_model, reward,
                                                 reward_calc, transforms,
                                                 m_init, S_init,
                                                 horizon)
        self.n_samples = n_samples
        self.lhs_samples = None

    @type_checked
    def predict(self, X):
        X = self.transforms["state"].transform(X)
        X, U = self._predict(X)
        X = self.transforms["state"].inverse_transform(X)
        return X.numpy(), U.numpy()

    def _predict(self, X):
        # unnormalized control distribution
        U = self.controller._compute_action(X)
        # normalize
        U_scaled = self.transforms["ctrl"].transform(U)

        Z = tf.concat([X, U_scaled], axis=1)

        m_dX, S_dX = self.dyn_model.predict(Z)

        e = tf.linalg.eigvalsh(S_dX, name='eigvalsh_dx')
        pred = tf.reduce_any(tf.less(e, self.eigenvalue_bound))
        S_dX = tf.cond(pred, lambda: self.patch_up(S_dX), lambda: S_dX)
        dX = self.sample(m_dX, S_dX)

        X = X + dX
        return X, U

    @type_checked
    def compute_reward(self):
        total_reward, states, rewards, _ = self._compute_reward(self.m_init, self.S_init)
        states = self.transforms["state"].inverse_transform(states.stack())
        states = states.numpy()
        rewards = rewards.stack().numpy()
        return total_reward, states, rewards

    def _loop_cond(self, j, states, rewards):
        return tf.less(j, self.horizon)

    def _loop_body(self, j, states, rewards):
        X, U = self._predict(states.read(j))
        t = tf.cast(tf.divide(j + 1, self.horizon), dtype=float_type)
        R = self.reward.compute_reward(X, U, t)

        states = states.write(j + 1, X)
        rewards = rewards.write(j, R)

        return j + 1, states, rewards

    def _compute_reward(self, m_x, S_x):
        X = self.sample_inits(m_x, S_x)

        if self.transition == 1:
            self.reward.prev_states = X
        elif self.transition == 2:
            self.reward.base_rewards[0].prev_states = X

        states = tf.TensorArray(float_type, size=self.horizon + 1,
                                clear_after_read=False,
                                tensor_array_name='states')
        rewards = tf.TensorArray(float_type, size=self.horizon,
                                 clear_after_read=False,
                                 tensor_array_name='rewards')

        states = states.write(0, self.transforms["state"].transform(X))

        loop_vars = [tf.constant(0, dtype=tf.int32, name="timesteps"),
                     states,
                     rewards]

        _, states, rewards = tf.while_loop(self._loop_cond,
                                           self._loop_body,
                                           loop_vars)
        return self.reward_calc(rewards.stack()), states, rewards, {}

    def sample_inits(self, m, S):
        lhs_samples = lhs(self.latent_dim, self.n_samples)
        self.lhs_samples = tf.constant(norm(0, 1).ppf(lhs_samples), dtype=float_type)
        R = tf.linalg.cholesky(S, name='init_sampling_choleksy')
        X = tf.transpose(tf.matmul(R, self.lhs_samples, transpose_b=True)) + m
        return X

    def sample(self, m, S):
        lhs_samples = lhs(self.latent_dim, self.n_samples)
        lhs_samples = tf.constant(norm(0, 1).ppf(lhs_samples), dtype=float_type)

        R = tf.linalg.cholesky(S, name='sampling_choleksy')
        X = tf.transpose(tf.matmul(R, lhs_samples[:, :, None]), perm=[0, 2, 1]) + m[:, None, :]
        return X[:, 0, :]


class BatchSamplingController(SamplingController):
    def _predict(self, X):
        # unnormalized control distribution
        U = self.controller._compute_action(tf.reshape(X, (self.pop_size, self.n_samples, self.latent_dim)))
        U = tf.reshape(U, (self.pop_size * self.n_samples, -1))
        # normalize
        U_scaled = self.transforms["ctrl"].transform(U)

        Z = tf.concat([X, U_scaled], axis=1)

        m_dX, S_dX = self.dyn_model.predict(Z)

        e = tf.linalg.eigvalsh(S_dX, name='eigvalsh_dx')
        pred = tf.reduce_any(tf.less(e, self.eigenvalue_bound))
        S_dX = tf.cond(pred, lambda: self.patch_up(S_dX), lambda: S_dX)
        dX = self.sample(m_dX, S_dX)

        X = X + dX  # (10)
        return X, U

    def _loop_body(self, j, states, rewards):
        X, U = self._predict(states.read(j))
        t = tf.cast(tf.divide(j + 1, self.horizon), dtype=float_type)
        R = self.reward.compute_reward(X, U, t)

        states = states.write(j + 1, X)
        rewards = rewards.write(j, tf.reshape(R, (self.pop_size, self.n_samples, 1)))

        return j + 1, states, rewards

    def _compute_reward(self, m_x, S_x):
        X = self.sample_inits(m_x, S_x)
        X = self.transforms["state"].transform(X)
        X = tf.tile(X, [self.pop_size, 1])
        states = tf.TensorArray(float_type, size=self.horizon + 1,
                                clear_after_read=False,
                                tensor_array_name='states')
        rewards = tf.TensorArray(float_type, size=self.horizon,
                                 clear_after_read=False,
                                 tensor_array_name='rewards')

        states = states.write(0, X)

        loop_vars = [tf.constant(0, dtype=tf.int32, name="timesteps"),
                     states,
                     rewards]

        _, states, rewards = tf.while_loop(self._loop_cond,
                                           self._loop_body,
                                           loop_vars)

        total_reward = tf.transpose(rewards.stack(), perm=(1, 0, 2, 3))
        total_reward = self.reward_calc(total_reward)
        return total_reward, states, rewards, {}

    def sample(self, m, S):
        lhssamples = lhs(self.latent_dim, self.n_samples * self.pop_size)
        lhssamples = tf.constant(norm(0, 1).ppf(lhssamples), dtype=float_type)

        R = tf.linalg.cholesky(S, name='sampling_choleksy')
        X = tf.transpose(tf.matmul(R, lhssamples[:, :, None]), perm=[0, 2, 1]) + m[:, None, :]
        return X[:, 0, :]

    def losses(self):
        total_reward, _, _, _ = self._compute_reward(self.m_init, self.S_init)
        return -1. * total_reward

    def loss(self):
        return tf.reduce_min(self.losses())

    @property
    def params(self):
        return [param.numpy() for param in self.controller.model.trainable_variables]

    @params.setter
    def params(self, new_params):
        for i, param in enumerate(self.controller.model.trainable_variables):
            param.assign(new_params[i])
        self.controller.model.update_cov()


def calc_inventory():
    reward_calcs = {"mean": mean_reward,
                    "min": min_reward,
                    "max": max_reward,
                    "sum": sum_reward}
    return reward_calcs


def mean_reward(rewards):
    return tf.reduce_mean(rewards, axis=[-3, -2, -1], name='total_reward')


def min_reward(rewards):
    return tf.reduce_min(rewards, name='total_reward')


def max_reward(rewards):
    return tf.reduce_max(rewards, name='total_reward')


def sum_reward(rewards):
    return tf.reduce_sum(rewards, name='total_reward')


def get_reward_calc(calc_list):
    reward_calcs = calc_inventory()
    if len(calc_list) == 1:
        return reward_calcs[calc_list[0]]
    else:
        def reward_calc(rewards):
            base_rew = []
            for i, c in enumerate(calc_list):
                r = reward_calcs[c](rewards[:, i, :, :])
                base_rew.append(r)
            return sum_reward(base_rew)

        return reward_calc
