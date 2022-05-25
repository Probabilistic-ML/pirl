import os
import sys

from gpflow.kernels import SquaredExponential
from ..gp import GP

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("..\\.."))))
from pirl.uncertainty.noisy_predict import MomentMatching


def model_trainer(X, Y, budget_samples=800):
    X0 = X[-budget_samples:]
    Y0 = Y[-budget_samples:]
    gp = GP(X0, Y0, SquaredExponential, return_iK=True)
    iKs, lengths, sigma2s = gp.fit()
    matcher = MomentMatching(X0, Y0,
                             inv_cov=iKs,
                             lengthscales=lengths,
                             kernel_var=sigma2s)
    return matcher
