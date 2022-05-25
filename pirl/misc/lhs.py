import numpy as np
from scipy.spatial.distance import pdist


def lhs(variables, samples, its=1000):
    d_best = 0
    for i in range(its):
        S = np.repeat(np.linspace(1 / (samples * 2), 1 - 1 / (samples * 2),
                                  samples).reshape(-1, 1), variables, axis=1)
        order = np.random.permutation(range(samples))
        order = np.stack([np.random.choice(range(samples), samples,
                                           replace=False)
                          for _ in range(variables)], axis=1)
        for z in range(variables):
            S[:, z] = S[order[:, z], z]
        d_min = np.min(pdist(S))
        if i == 0:
            S_best = np.copy(S)
        elif d_min > d_best:
            d_best = np.copy(d_min)
            S_best = np.copy(S)
    return S_best
