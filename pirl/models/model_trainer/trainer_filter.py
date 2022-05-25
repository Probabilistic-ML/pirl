from .dgcn_trainer_filter import get_dgcn_model_trainer
from .bnn_trainer_filter import get_bnn_model_trainer
from ..gp import GP
from .trainer_sampling import Predictor
from ...config import KERNEL_MAP


def get_filter_model_trainer(filter_cls, kernel_name='squared_exponential', **filter_kwargs):
    if kernel_name == 'dgcn':
        return get_dgcn_model_trainer(filter_cls, **filter_kwargs)
    elif kernel_name == 'bnn':
        return get_bnn_model_trainer(filter_cls, **filter_kwargs)
    else:
        kernel_cls = KERNEL_MAP[kernel_name]

        def model_trainer(X, Y):
            _, n_inps = X.shape
            gp = GP(X, Y, kernel_cls)
            Ls, lengths, sigma2s = gp.fit()
            predictor = Predictor(X, Y, Ls, lengths, sigma2s, kernel_name)
            matcher = filter_cls(predictor, n_inps, **filter_kwargs)
            return matcher

        return model_trainer
