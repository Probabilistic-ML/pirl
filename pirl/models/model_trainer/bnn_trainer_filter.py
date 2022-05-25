from .bnn_trainer_sampling import bnn_model_trainer


def get_bnn_model_trainer(filter_cls, **filter_kwargs):
    def model_trainer(X, Y):
        _, n_inps = X.shape
        predictor = bnn_model_trainer(X, Y)
        matcher = filter_cls(predictor, n_inps, **filter_kwargs)
        return matcher

    return model_trainer

