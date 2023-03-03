import tensorflow as tf

from ..config import float_type
from .core import BaseTransformer


class StandardScaler(BaseTransformer):
    def __init__(self):
        self.mean = tf.Variable([0.], trainable=False, dtype=float_type,
                                shape=[None,], name='scaler_mean')
        self.std = tf.Variable([1.], trainable=False, dtype=float_type,
                               shape=[None,], name='scaler_std')
        
    @property
    def C(self):
        return tf.linalg.diag(self.std)
    
    @property
    def C_inv(self):
        # noinspection PyTypeChecker
        return tf.linalg.diag(1./self.std)
    
    def fit(self, X):
        self.mean.assign(tf.reduce_mean(X, axis=0)) 
        self.std.assign(tf.math.reduce_std(X, axis=0))

    def transform(self, X):
        return tf.divide(tf.add(X, -self.mean), self.std)
    
    def transform_std(self, sigma):
        return tf.matmul(sigma, self.C_inv)
    
    def transform_var(self, sigma2):
        return tf.matmul(tf.matmul(self.C_inv, sigma2), self.C_inv)
        
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        return tf.add(tf.multiply(self.std, X), self.mean)
    
    def inverse_transform_std(self, sigma):
        return tf.matmul(self.C, sigma)
    
    def inverse_transform_var(self, sigma2):
        return tf.matmul(tf.matmul(self.C, sigma2), self.C)
    
    def weight_transform(self, W):
        return self.inverse_transform_var(W)
