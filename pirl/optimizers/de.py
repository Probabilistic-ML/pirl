from time import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class DifferentialEvolution:
    def __init__(self):
        self.model = None
        self.params = None
        self.step = 0
        self.iters = 100

    @tf.function
    def losses(self):
        return self.model.losses()

    def objective(self, *params):
        t0 = time()
        self.model.params = self._set_params(list(params))
        losses = self.losses()
        current_cost = tf.reduce_min(losses)
        tf.print(f"Step: {self.step}/{self.iters},",
                 f"loss: {current_cost:.4e}",
                 f"[{time() - t0:.2e} s]")
        self.step += 1
        return losses

    def optimize(self, model, iters=100, **kwargs):
        self.model = model
        self.iters = iters
        results = tfp.optimizer.differential_evolution_minimize(self.objective, self._get_params(),
                                                                population_size=self.model.pop_size,
                                                                max_iterations=iters)
        self.step = 0
        self.params = results.position
        return results.objective_value, results.position

    def _get_params(self):
        if 'NNController' not in self.model.controller.__class__.__name__:
            return self.model.params
        else:
            params_tmp = self.model.params
            weights = params_tmp[0::2]
            biases = params_tmp[1::2]
            pop_size = self.model.pop_size
            num_layers = int(len(weights) / pop_size)
            params = []
            for i in range(num_layers):
                params.append(np.stack(weights[i * pop_size:(i + 1) * pop_size]))
                params.append(np.stack(biases[i * pop_size:(i + 1) * pop_size]))
            return params

    def _set_params(self, params):
        if 'NNController' not in self.model.controller.__class__.__name__:
            return params
        else:
            weights = params[0::2]
            biases = params[1::2]
            pop_size = self.model.pop_size
            num_layers = len(weights)
            params_tmp = []
            for i in range(num_layers):
                for j in range(pop_size):
                    params_tmp.append(weights[i][j, ...])
                    params_tmp.append(biases[i][j, ...])
            return params_tmp
