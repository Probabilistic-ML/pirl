from .dgcn_trainer_sampling import dgcn_model_trainer


def get_dgcn_model_trainer(filter_cls, **filter_kwargs):
    def model_trainer(X, Y):
        _, n_inps = X.shape
        predictor = dgcn_model_trainer(X, Y)
        matcher = filter_cls(predictor, n_inps, **filter_kwargs)
        return matcher
    return model_trainer

