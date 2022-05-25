from ..dgcn import DGCN


def dgcn_model_trainer(X, Y):
    _, n_inps = X.shape
    model = DGCN(X, Y, num_neurons=min(2 * n_inps, 10))
    model.fit(max_epochs=500)

    predictor = Predictor(model)
    return predictor


class Predictor:
    def __init__(self, model):
        self.model = model

    def predict(self, X_test):
        return self.model._predict(X_test, pred_var=True)
