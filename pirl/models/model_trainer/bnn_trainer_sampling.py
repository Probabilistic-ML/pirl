from ..bnn import make_bnn
from ...config import BNN_CONFIG


def bnn_model_trainer(X, Y):
    _, n_inps = X.shape
    _, n_outs = Y.shape
    model = make_bnn(n_inps, n_outs, BNN_CONFIG["widths"], BNN_CONFIG["activations"], BNN_CONFIG["weight_decays"],
                     BNN_CONFIG["learning_rate"])
    model.fit(X, Y, batch_size=BNN_CONFIG["batch_size"], epochs=BNN_CONFIG["epochs"])
    predictor = Predictor(model)
    return predictor


class Predictor:
    def __init__(self, model):
        self.model = model

    def predict(self, X_test):
        return self.model.predict(X_test)
