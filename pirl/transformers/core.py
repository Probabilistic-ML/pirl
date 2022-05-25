import abc
import tensorflow as tf


class BaseTransformer(abc.ABC):
    @abc.abstractmethod
    def fit(self, X):
        raise NotImplementedError("This is an abstract class. Please use the child classes.")

    @abc.abstractmethod
    def transform(self, X):
        raise NotImplementedError("This is an abstract class. Please use the child classes.")

    @abc.abstractmethod
    def fit_transform(self, X):
        raise NotImplementedError("This is an abstract class. Please use the child classes.")

    @abc.abstractmethod
    def inverse_transform(self, Z):
        raise NotImplementedError("This is an abstract class. Please use the child classes.")

    @abc.abstractmethod
    def transform_var(self, S2):
        raise NotImplementedError("This is an abstract class. Please use the child classes.")

    @abc.abstractmethod
    def inverse_transform_var(self, C2):
        raise NotImplementedError("This is an abstract class. Please use the child classes.")

    @abc.abstractmethod
    def transform_std(self, S):
        raise NotImplementedError("This is an abstract class. Please use the child classes.")

    @abc.abstractmethod
    def inverse_transform_std(self, C):
        raise NotImplementedError("This is an abstract class. Please use the child classes.")

    @abc.abstractmethod
    def weight_transform(self, W):
        raise NotImplementedError("This is an abstract class. Please use the child classes.")


class Pipeline(BaseTransformer):
    def __init__(self, transformers):
        self.transformers = transformers

    def fit(self, X):
        Z = tf.identity(X)
        for transformer in self.transformers:
            Z = transformer.fit_transform(Z)
        return self

    def transform(self, X):
        Z = tf.identity(X)
        for transformer in self.transformers:
            Z = transformer.transform(Z)
        return Z

    def fit_transform(self, X):
        Z = tf.identity(X)
        for transformer in self.transformers:
            Z = transformer.fit_transform(Z)
        return Z

    def inverse_transform(self, Z):
        X = tf.identity(Z)
        for transformer in self.transformers[::-1]:
            X = transformer.inverse_transform(X)
        return X

    def transform_var(self, S2):
        C2 = tf.identity(S2)
        for transformer in self.transformers:
            C2 = transformer.transform_var(C2)
        return C2

    def inverse_transform_var(self, C2):
        S2 = tf.identity(C2)
        for transformer in self.transformers[::-1]:
            S2 = transformer.inverse_transform_var(S2)
        return S2

    def transform_std(self, S):
        C = tf.identity(S)
        for transformer in self.transformers:
            C = transformer.transform_std(C)
        return C

    def inverse_transform_std(self, C):
        S = tf.identity(C)
        for transformer in self.transformers[::-1]:
            S = transformer.inverse_transform_std(S)
        return S

    def weight_transform(self, W):
        W = tf.identity(W)
        for transformer in self.transformers:
            W = transformer.weight_transform(W)
        return W
