import tensorflow as tf
import numpy as np

from ..models.auxiliary_models import FunctionModel
from ..config import float_type


class BaseReward:
    def __init__(self, state_dim, weights=None, target=None, transformable=True):
        self.state_dim = state_dim
        if target is not None:
            self.target_raw = tf.cast(tf.reshape(target, (1, state_dim)),
                                      dtype=float_type)
        else:
            self.target_raw = None
        self.target = self.target_raw
        if weights is not None:
            self.weights_raw = tf.cast(tf.reshape(weights, (state_dim, state_dim)),
                                       dtype=float_type)
        else:
            self.weights_raw = None
        self.weights = self.weights_raw
        self.transformable = transformable
        self._state_transformer = None

    def transform_reward(self):
        if self.transformable:
            raise NotImplementedError("A transformable reward should override the transform_reward method. "
                                      "This class failed to do so!")
        return

    @property
    def state_transformer(self):
        return self._state_transformer

    @state_transformer.setter
    def state_transformer(self, transformer):
        self._state_transformer = transformer
        if self.transformable:
            self.transform_reward()


class ExponentialReward(BaseReward):
    def transform_reward(self):
        self.target = self.state_transformer.transform(self.target_raw)
        self.state_dim = tf.shape(self.target)[1]
        self.weights = self.state_transformer.weight_transform(self.weights_raw)

    def inv_transform_states(self, X, U):
        if not self.transformable and self.state_transformer is not None:
            X = self.state_transformer.inverse_transform(X)
        return X, U

    def compute_reward(self, X, U, t):
        X, U = self.inv_transform_states(X, U)
        X_tmp = tf.add(X, -self.target)[:, None, :]
        dist_tmp = X_tmp @ self.weights[None, :, :] @ tf.transpose(X_tmp, perm=(0, 2, 1))
        dist_tmp = dist_tmp[:, :, 0]
        R = tf.exp(-dist_tmp / 2) - 1
        return R


class TransitionReward(BaseReward):
    def __init__(self, state_dim, transition_var_no=0, transition_fct=lambda x, y: x - y):
        super().__init__(state_dim, transformable=False)
        self.transition_var_no = transition_var_no
        self.transition_fct = transition_fct
        self.prev_states = None

    def inv_transform_states(self, X, U):
        if not self.transformable and self.state_transformer is not None:
            X = self.state_transformer.inverse_transform(X)
        return X, U

    def compute_reward(self, X, U, t):
        X, U = self.inv_transform_states(X, U)
        R = self.transition_fct(X, self.prev_states)
        self.prev_states = X
        return R[:, self.transition_var_no:self.transition_var_no + 1]


class ExponentialMMReward(BaseReward):
    def transform_reward(self):
        self.target = self.state_transformer.transform(self.target_raw)
        self.state_dim = tf.shape(self.target)[1]
        self.weights = self.state_transformer.weight_transform(self.weights_raw)

    def inv_transform_states(self, m_x, S_x):
        if not self.transformable and self.state_transformer is not None:
            m_x = self.state_transformer.inverse_transform(m_x)
            S_x = self.state_transformer.inverse_transform_var(S_x)
        return m_x, S_x

    def compute_reward(self, m_x, S_x, m_u, S_u, t):
        m_x, S_x = self.inv_transform_states(m_x, S_x)
        SW = S_x @ self.weights

        iSpW = tf.transpose(
            tf.linalg.solve((tf.eye(self.state_dim, dtype=float_type) + SW),
                            tf.transpose(self.weights), adjoint=True, name="exp_reward_solve_1"))
        # Deisenroth 3.46

        m_r = tf.exp(-(m_x - self.target) @ iSpW @ tf.transpose(m_x - self.target) / 2)
        m_r = m_r / tf.sqrt(tf.linalg.det(tf.eye(self.state_dim, dtype=float_type) + SW))
        # Deisenroth 3.45

        m_r = m_r - 1

        m_r.set_shape([1, 1])
        return m_r


class QuadraticMMReward(BaseReward):
    def transform_reward(self):
        self.target = self.state_transformer.transform(self.target_raw)
        self.state_dim = tf.shape(self.target)[1]
        self.weights = self.state_transformer.weight_transform(self.weights_raw)

    def inv_transform_states(self, m_x, S_x):
        if not self.transformable and self.state_transformer is not None:
            m_x = self.state_transformer.inverse_transform(m_x)
            S_x = self.state_transformer.inverse_transform_var(S_x)
        return m_x, S_x

    def compute_reward(self, m_x, S_x, m_u, S_u, t):
        m_x, S_x = self.inv_transform_states(m_x, S_x)
        # see Diss Deisenroth p. 57 (3.59) + (3.60)
        SW = tf.matmul(S_x, self.weights)
        m_r = - tf.linalg.trace(SW)
        m_tmp = m_x - self.target
        m_r = m_r - tf.matmul(tf.matmul(m_tmp, self.weights), m_tmp, transpose_b=True)

        m_r.set_shape([1, 1])
        return m_r


def _zero_reward():
    return tf.zeros([1, 1], dtype=float_type)


class EndpointReward(BaseReward):
    def __init__(self, state_dim, weights, target):
        super().__init__(state_dim, weights=weights, target=target)
        self.threshold = tf.constant(1. - 1e-5, dtype=float_type)

    def transform_reward(self):
        self.target = self.state_transformer.transform(self.target_raw)
        self.state_dim = tf.shape(self.target)[1]
        self.weights = self.state_transformer.weight_transform(self.weights_raw)

    def inv_transform_states(self, X, U):
        if not self.transformable and self.state_transformer is not None:
            X = self.state_transformer.inverse_transform(X)
        return X, U

    def _compute_reward(self, X, U, t):
        X, U = self.inv_transform_states(X, U)
        X_tmp = tf.add(X, -self.target)[:, None, :]
        dist_tmp = X_tmp @ self.weights[None, :, :] @ tf.transpose(X_tmp, perm=(0, 2, 1))
        dist_tmp = dist_tmp[:, :, 0]
        
        R = heavyside(1e-4 - dist_tmp)  # penalty if distance to zero too large
        return R

    def compute_reward(self, X, U, t):
        return tf.cond(tf.greater_equal(t, self.threshold),
                       lambda: self._compute_reward(X, U, t),
                       _zero_reward)


class EndpointMMReward(BaseReward):
    def __init__(self, state_dim, weights, target):
        super().__init__(state_dim, weights=weights, target=target)
        self.threshold = tf.constant(1. - 1e-5, dtype=float_type)

    def transform_reward(self):
        self.target = self.state_transformer.transform(self.target_raw)
        self.state_dim = tf.shape(self.target)[1]
        self.weights = self.state_transformer.weight_transform(self.weights_raw)

    def inv_transform_states(self, m_x, S_x):
        if not self.transformable and self.state_transformer is not None:
            m_x = self.state_transformer.inverse_transform(m_x)
            S_x = self.state_transformer.inverse_transform_var(S_x)
        return m_x, S_x

    def _compute_reward(self, m_x, S_x, m_u, S_u, t):
        m_x, S_x = self.inv_transform_states(m_x, S_x)
        SW = S_x @ self.weights

        iSpW = tf.transpose(
            tf.linalg.solve((tf.eye(self.state_dim, dtype=float_type) + SW),
                            tf.transpose(self.weights), adjoint=True, name="exp_reward_solve_1"))
        # Diss Deisenroth 3.46

        m_r = tf.exp(-(m_x - self.target) @ iSpW @ tf.transpose(m_x - self.target) / 2)
        m_r = m_r / tf.sqrt(tf.linalg.det(tf.eye(self.state_dim, dtype=float_type) + SW))
        # Deisenroth 3.45

        m_r = m_r - 1

        m_r.set_shape([1, 1])
        return m_r

    def compute_reward(self, m_x, S_x, m_u, S_u, t):
        return tf.cond(tf.greater_equal(t, self.threshold),
                       lambda: self._compute_reward(m_x, S_x, m_u, S_u, t),
                       _zero_reward)


class ControlReward(BaseReward):

    @staticmethod
    def compute_reward(X, U, t):
        R = -tf.reduce_sum(tf.square(U), axis=1, keepdims=True)
        return R


class ControlMMReward(BaseReward):

    @staticmethod
    def compute_reward(m_x, S_x, m_u, S_u, t):
        m_r = -tf.reduce_sum(tf.square(m_u), axis=1, keepdims=True)
        m_r.set_shape([1, 1])
        return m_r


class FilterBasedReward(BaseReward):
    """
    reward_function should take arguments state and t
    """

    def __init__(self, state_dim, reward_function, predictor_cls):
        super().__init__(state_dim=state_dim, transformable=False)
        self.predictor_cls = predictor_cls
        self.function = reward_function
        self.model, self.predictor = None, None

    def fix_t(self, t):
        def func(x):
            return self.function(x, t)

        return func

    def compute_reward(self, m_x, S_x, m_u, S_u, t):
        m_x = self.state_transformer.inverse_transform(m_x)
        S_x = self.state_transformer.inverse_transform_var(S_x)
        self.model = FunctionModel(self.fix_t(t), 1)
        self.predictor = self.predictor_cls(self.model, self.state_dim)
        m_r, _, _ = self.predictor.noisy_predict(m_x, S_x)
        m_r.set_shape([1, 1])
        return m_r


class ExplorationReward(BaseReward):

    @staticmethod
    def compute_reward(X, U, t):
        return tf.reduce_sum(tf.math.reduce_variance(X, axis=0))


class ExplorationDensityReward(BaseReward):
    def __init__(self, state_dim, approach):
        super().__init__(state_dim=state_dim)
        self.approach = approach
        self.X = None

    @staticmethod
    def compute_reward(m_x, S_x, m_u, S_u, t):
        return tf.reduce_sum(tf.linalg.diag_part(S_x))


class CombinedReward(BaseReward):
    def __init__(self, state_dim, rewards: list, coefs=None):
        super().__init__(state_dim)
        if not rewards:
            raise RuntimeError("You must pass at least one reward!")
        state_dims = [reward.state_dim for reward in rewards]
        state_dims.append(state_dim)
        if len(set(state_dims)) > 1:
            raise tf.errors.InvalidArgumentError("State dimensions cannot differ")
        self.BaseRewards = rewards
        if coefs is not None:
            self.coefs = [tf.constant(c, dtype=float_type) for c in coefs]
        else:
            self.coefs = len(rewards) * [tf.constant(1., dtype=float_type)]
        self.transformable = all([reward.transformable for reward in rewards])
        if not self.transformable:
            for reward in self.BaseRewards:
                reward.transformable = False

    def transform_reward(self):
        for reward in self.BaseRewards:
            reward.state_transformer = self.state_transformer

    def compute_reward(self, X, U, t):
        total_reward = 0
        for reward, coef in zip(self.BaseRewards, self.coefs):
            r = reward.compute_reward(X, U, t)
            total_reward += coef * r
        return total_reward


class CombinedMMReward(BaseReward):
    def __init__(self, state_dim, rewards: list, coefs=None):
        super().__init__(state_dim)
        if not rewards:
            raise RuntimeError("You must pass at least one reward!")
        state_dims = [reward.state_dim for reward in rewards]
        state_dims.append(state_dim)
        if len(set(state_dims)) > 1:
            raise tf.errors.InvalidArgumentError("State dimensions cannot differ")
        self.BaseRewards = rewards
        if coefs is not None:
            self.coefs = [tf.constant(c, dtype=float_type) for c in coefs]
        else:
            self.coefs = len(rewards) * [tf.constant(1., dtype=float_type)]
        self.transformable = all([reward.transformable for reward in rewards])
        if not self.transformable:
            for reward in self.BaseRewards:
                reward.transformable = False

    def transform_reward(self):
        for reward in self.BaseRewards:
            reward.state_transformer = self.state_transformer

    def compute_reward(self, m_x, S_x, m_u, S_u, t):
        m_x = self.state_transformer.inverse_transform(m_x)
        S_x = self.state_transformer.inverse_transform_var(S_x)
        total_reward = 0
        for reward, coef in zip(self.BaseRewards, self.coefs):
            r = reward.compute_reward(m_x, S_x, m_u, S_u, t)
            total_reward += coef * r
        return total_reward


class StackedReward(BaseReward):
    def __init__(self, state_dim, rewards: list, coefs=None):
        super().__init__(state_dim)
        if not rewards:
            raise RuntimeError("You must pass at least one reward!")
        state_dims = [reward.state_dim for reward in rewards]
        state_dims.append(state_dim)
        if len(set(state_dims)) > 1:
            raise tf.errors.InvalidArgumentError("State dimensions cannot differ")

        self.base_rewards = rewards
        if coefs is not None:
            self.coefs = [tf.constant(c, dtype=float_type) for c in coefs]
        else:
            self.coefs = len(rewards) * [tf.constant(1., dtype=float_type)]
        self.transformable = all([reward.transformable for reward in rewards])

        if not self.transformable:
            for reward in self.base_rewards:
                reward.transformable = False

    def transform_reward(self):
        for reward in self.base_rewards:
            reward.state_transformer = self.state_transformer

    def compute_reward(self, m_x, S_x, m_u, S_u, t):
        m_x = self.state_transformer.inverse_transform(m_x)
        S_x = self.state_transformer.inverse_transform_var(S_x)
        rewards = []
        for reward, coef in zip(self.base_rewards, self.coefs):
            r = coef * reward.compute_reward(m_x, S_x, m_u, S_u, t)
            rewards.append(r)
        rewards = tf.stack(rewards)
        return rewards


def heavyside(x, k=20):
    shape = tf.shape(x)
    return tf.reduce_sum(tf.divide(-tf.ones(shape, dtype=float_type),
                                   tf.add(tf.ones(shape, dtype=float_type),
                                          tf.exp(2. * k * x))),
                         axis=1, keepdims=True)


def _safe_get_action_bound(bound, control_dim):
    if bound is None:
        return None
    if isinstance(bound, (int, float)):
        bound = [bound] * control_dim
    try:
        bound = np.array(bound).ravel()
    except (ValueError, TypeError) as exc:
        raise ValueError("Action bound not understood") from exc
    assert bound.size == control_dim
    return tf.cast(bound, dtype=float_type)
