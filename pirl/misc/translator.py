import functools
import numpy as np
import tensorflow as tf


def type_checked(func, inputs="tf", outputs=None):
    """ Decorator to type_check the inputs and outputs of a function
        It is important that the function does not return tuples on its own, as
        isinstance(res, tuple) is checked to decide if function has one or multiple outputs.
        This is not necessarily the best way of checking this. Probably will be changed in the future

        Usage:
        ------
        @type_checked(inputs='tf', outputs='np')
        def my_func(arg1, arg2, kwarg1=None, kwarg2=not None):
            ...
            return meaningful_results
    """

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        new_args = [ensure_type(arg, inputs) for arg in args]
        new_kwargs = {key: ensure_type(value, inputs) for key, value in kwargs.items()}
        res = func(*new_args, **new_kwargs)
        if isinstance(res, tuple):
            return tuple([ensure_type(r, outputs) for r in res])
        return ensure_type(res, outputs)

    return wrapped


def ensure_type(arg, type_name):
    """ ensures that arg converted to tensor or numpy array
    Parameters:
    -----------
    arg - any
        Argument that is to be converted to the desired type if necessary
    type_name - str
        name of the type. 'tf' for tensor output, 'np' for numpy output. torch should be added later

    Returns
    -------
    safe_arg - any or type_name
        arg is converted to type_name if this is meaningful i.e. only arrays are converted to tensor
        and vice versa.
    
    """
    if type_name == "tf":
        return ensure_tf(arg)
    if type_name == "np":
        return ensure_np(arg)
    return arg


def ensure_tf(arg):
    """ Converts arg to tensor if arg is a numpy array
    
    >>> ensure_tf(np.random.randn(100,2))
    """
    if isinstance(arg, np.ndarray):
        return tf.convert_to_tensor(arg)
    return arg


def ensure_np(arg):
    """ Converts arg to numpy array if arg is a tensor"""
    if tf.is_tensor(arg):
        try:
            return arg.numpy()
        except AttributeError:
            return arg.eval(session=tf.compat.v1.Session())
    return arg
