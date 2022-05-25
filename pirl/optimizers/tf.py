from time import time
import numpy as np
import tensorflow as tf
from ..config import float_type


def not_inited_train_step(*args, **kwargs):
    raise ValueError("Model is not inited yet!")


class TensorflowOptimizer:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.lr = self.optimizer.learning_rate.numpy()
        self.model = None
        self.train_step = not_inited_train_step
        self.loss_history = None

    @tf.function
    def _train_step(self):
        loss, grads = self.model.train_step()
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def optimize(self, model, iters=1000, patience_stop=50, patience_red=10, epsilon=0,
                 discount=0.5, min_lr=0, verbose=1, **kwargs):
        self.model = model
        self.train_step = self._train_step
        loss_history_tmp = []
        self.optimizer.learning_rate.assign(self.lr)
        self.optimizer.set_weights([np.zeros_like(w) for w in self.optimizer.get_weights()])
        lowest_cost = tf.convert_to_tensor(np.inf, dtype=float_type)
        best_param = self.model.params.copy()
        slow_decr_counter = 0
        non_decr_counter_lr = 0
        for step in tf.range(iters):
            t0 = time()
            param_tmp = self.model.params.copy()
            try:
                current_cost = float(self.train_step())
            except tf.errors.InvalidArgumentError as exc:
                if "invertible" in str(exc):
                    tf.print(str(exc))
                    break
                else:
                    raise exc
            diff_low = tf.add(lowest_cost, - current_cost)
            if tf.greater(diff_low, 0):  # update best_param if cost declines
                lowest_cost = current_cost
                best_param = param_tmp.copy()
                non_decr_counter_lr = 0
            else:
                non_decr_counter_lr += 1

            if tf.less(diff_low, epsilon):
                slow_decr_counter += 1  # increase by 1 in case of slow cost decrease
            else:
                slow_decr_counter = 0  # otherwise reset counter

            loss_history_tmp.append([step + 1, current_cost, patience_stop - slow_decr_counter])
            if (verbose == 2 and step % 20 == 0) or verbose > 2:
                tf.print(f"Step: {step + 1}/{iters},",
                         f"loss: {current_cost:.4e},",
                         f"patience left: {patience_stop - slow_decr_counter}",
                         f"[{time() - t0:.2e} s]")

            if tf.equal(slow_decr_counter, patience_stop):  # early stopping
                if verbose > 1:
                    tf.print("Cost is non-decreasing...will continue.")
                return lowest_cost, best_param

            if tf.equal(non_decr_counter_lr, patience_red):
                self.adj_lr(discount=discount, min_lr=min_lr)  # adjust lr
                self.model.params = best_param
                if verbose > 2:
                    tf.print("Reducing learning rate to", self.optimizer.learning_rate)
                non_decr_counter_lr = 0  # reset counter
            self.loss_history = np.array(loss_history_tmp)
        return lowest_cost, best_param

    def adj_lr(self, discount=0.5, min_lr=0):
        new_lr = tf.math.maximum(discount * self.optimizer.learning_rate, min_lr)
        self.optimizer.learning_rate.assign(new_lr)
        return self
