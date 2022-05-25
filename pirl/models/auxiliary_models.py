import tensorflow as tf
from ..config import float_type


class FunctionModel:
    def __init__(self, function, state_dim):
        self.function = function
        self.state_dim = state_dim

    def predict(self, *args):
        m = tf.cast(self.function(*args), dtype=float_type)
        S = tf.zeros((tf.shape(m)[0], self.state_dim, self.state_dim),
                     dtype=float_type)
        return m, S
