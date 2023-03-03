import tensorflow as tf
from ..config import float_type
from .core import BaseTransformer


# based on PCA from sklearn
class PCA(BaseTransformer):
    def __init__(self, n_components=None, explained_variance=1.):
        self.n_components = n_components
        self.explained_variance = explained_variance
        self.mean_ = tf.Variable([0.], trainable=False, dtype=float_type,
                                 shape=[None, ], name='pca_mean')
        self.n_components_ = tf.Variable(0, trainable=False, dtype=tf.int32,
                                         shape=None, name='pca_components')

    def fit(self, X):
        self._fit(X)
        return self

    def _fit(self, X):
        X_tmp = tf.cast(X, dtype=float_type)
        n_samples, n_features = X_tmp.shape
        max_components = min(n_samples, n_features)
        if self.n_components is None:
            n_components = max_components
            n_components_max = max_components
        elif self.n_components <= max_components:
            n_components = self.n_components
            n_components_max = self.n_components
        else:
            message = "n_components must be less or equal than "
            raise ValueError(message + "%r" % max_components)
        self.mean_.assign(tf.reduce_mean(X_tmp, axis=0))
        X_tmp = tf.add(X_tmp, -self.mean_)

        S, U, V = tf.linalg.svd(X_tmp)
        Vt = tf.transpose(V)

        explained_variance_ = tf.square(S) / (n_samples - 1)
        total_var = tf.reduce_sum(explained_variance_, axis=0)
        explained_variance_ratio_ = explained_variance_ / total_var

        comp_variance_ = tf.reduce_sum(explained_variance_ratio_[:n_components],
                                       axis=0)
        while comp_variance_ >= self.explained_variance:
            n_components -= 1
            comp_variance_ = tf.reduce_sum(explained_variance_ratio_[:n_components],
                                           axis=0)
        n_components += 1
        n_components = min(n_components, n_components_max)

        self.n_samples_, self.n_features_ = n_samples, n_features
        self.n_components_.assign(n_components)
        self.components_ = Vt[:n_components]
        self.explained_variance_ = explained_variance_[:n_components]
        self.explained_variance_ratio_ = explained_variance_ratio_[:n_components]
        self.singular_values_ = S[:n_components]
        print(f"PCA fit with {n_components} components. "
              f"Var. ratio: {tf.reduce_sum(self.explained_variance_ratio_)}")

        return U, S, Vt

    def fit_transform(self, X):
        U, S, Vt = self._fit(X)
        U = tf.multiply(S[:self.n_components_], U[:, :self.n_components_])
        return U

    def transform(self, X):
        X_tmp = tf.cast(X, dtype=float_type)
        if self.mean_ is None:
            raise AttributeError("PCA is not fitted")
        X_tmp = tf.add(X_tmp, -self.mean_)
        return tf.matmul(X_tmp, self.components_, transpose_b=True)

    def transform_var(self, S):
        return tf.matmul(tf.matmul(self.components_, S), self.components_,
                         transpose_b=True)

    def inverse_transform(self, X):
        X_tmp = tf.cast(X, dtype=float_type)
        if self.mean_ is None:
            raise AttributeError("PCA is not fitted")
        return tf.matmul(X_tmp, self.components_) + self.mean_

    def inverse_transform_var(self, S):
        return tf.matmul(tf.matmul(self.components_, S, transpose_a=True),
                         self.components_)

    def transform_std(self, S):
        return tf.matmul(S, self.components_, transpose_b=True)

    def inverse_transform_std(self, S):
        return tf.matmul(S, self.components_)

    def weight_transform(self, W):
        return self.transform_var(W)
