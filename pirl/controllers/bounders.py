import tensorflow as tf
from ..config import float_type


def sine_squash(m, s=None, max_action=None, min_action=None):
    """
    This should be rewritten better with understandable variable names
    Squashing function, passing the controls mean and variance
    through a sinus, as in gSin.m. The output is in [-max_action, max_action].
    IN: mean (m) and variance(s) of the control input, max_action
    OUT: mean (M) variance (S) and input-output (C) covariance of the squashed
         control input
    '''"""
    k = tf.shape(m)[1]
    init_box, offset = _get_init_box(k, max_action, min_action)
    if s is None:
        return init_box * tf.sin(m) + offset
    # Diss. Deisenroth, Appendix A.1
    M = init_box * tf.exp(-1 * tf.linalg.diag_part(s) / 2) * tf.sin(m) + offset
    lq = -(tf.linalg.diag_part(s)[:, None] + tf.linalg.diag_part(s)[None, :]) / 2
    q = tf.exp(lq)
    S = (tf.exp(lq + s) - q) * tf.cos(tf.transpose(m) - m)
    S = tf.add(S, - (tf.exp(lq - s) - q) * tf.cos(tf.transpose(m) + m))
    S = init_box * tf.transpose(init_box) * S / 2
    C = init_box * tf.linalg.diag(tf.exp(-1 * tf.linalg.diag_part(s) / 2) * tf.cos(m))
    return M, S, tf.reshape(C, shape=[k, k])


# from Gaussian Processes for Data-Efficient Learning in Robotics and Control
# sigma(x) = init_box * [9/8 * sin(x) + 1/8 * sin(3*x)] + offset
# sin(3*x) = 3 * sin(x) - 4 * sin(x)**3
# -> 9/8 * sin(x) + 1/8 * sin(3*x) = 1/2 * (3 * sin(x) - sin(x)**3)
# max./min. for sin(x) = +- 1 (vanishing derivative)
# |9/8 * sin(x) + 1/8 * sin(3*x)| <= 1
def trapezoidal_squash(m, s=None, max_action=None, min_action=None):
    k = tf.shape(m)[1]
    if s is None:
        init_box, offset = _get_init_box(k, max_action, min_action)
        return init_box * (9 / 8 * tf.sin(m) + 1 / 8 * tf.sin(3 * m)) + offset
    E = tf.eye(k, dtype=float_type)
    P = tf.concat((E, 3. * E), axis=0)  # 2k x k
    m2 = tf.transpose(tf.matmul(P, m, transpose_b=True))
    s2 = tf.matmul(tf.matmul(P, s), P, transpose_b=True)
    M2, S2, C2 = sine_squash(m2, s2, max_action=max_action, min_action=min_action)
    Q = tf.concat((9. * E, E), axis=1) / 8.
    M = tf.transpose(tf.matmul(Q, M2, transpose_b=True))
    S = tf.matmul(tf.matmul(Q, S2), Q, transpose_b=True)
    C = tf.matmul(tf.matmul(P, C2, transpose_a=True), Q, transpose_b=True)
    return M, S, C


def _get_init_box(dim, max_action=None, min_action=None):
    if max_action is None or min_action is None:
        init_box = tf.ones((1, dim), dtype=float_type)  # squashes in [-1,1] by default
        offset = tf.zeros((dim,), dtype=float_type)
    else:
        width = tf.add(max_action, -min_action)
        offset = tf.divide(tf.add(max_action, min_action), 2)
        init_box = tf.divide(width, 2) * tf.ones((1, dim), dtype=float_type)
    return init_box, offset


def custom_bounder(bound_fn):
    def bounder(m, s=None, max_action=None, min_action=None):
        """
        This should be rewritten better with understandable variable names
        Squashing function, passing the controls mean and variance
        through a sinus, as in gSin.m. The output is in [-max_action, max_action].
        IN: mean (m) and variance(s) of the control input, max_action
        OUT: mean (M) variance (S) and input-output (C) covariance of the squashed
             control input
        '''"""
        k = tf.shape(m)[1]
        init_box, offset = _get_init_box(k, max_action, min_action)
        if s is None:
            return init_box * bound_fn(m) + offset
        raise tf.errors.InvalidArgumentError()

    return bounder


@tf.custom_gradient
def disc_round(m, max_action=1., min_action=0., **kwargs):
    # clip has 1 grad almost everywhere and 0 otherwise so it is not useful for learning.
    def unit_grad(dm):
        return dm

    m = tf.clip_by_value(m, min_action, max_action)
    # round has 0 grad almost everywhere so it is not useful for learning.
    m = tf.math.round(m)
    return m, unit_grad
