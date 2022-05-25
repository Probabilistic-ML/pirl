import tensorflow as tf

from .agent import Agent
from .config import DEFAULT_OPTIMIZE_KWARGS


class PIRL:
    def __init__(self, agent: Agent):
        tf.config.run_functions_eagerly(False)
        self.agent = agent

    def refresh_agent(self):
        self.agent = Agent(self.agent.settings)

    def run_inits(self, num_inits, env=None, render=False, timesteps=None,
                  controller=None, reinit_agent=False):
        num_inits = int(num_inits)
        if reinit_agent:
            self.refresh_agent()
        self.agent.experiment(env, num=num_inits,
                              timesteps=timesteps if timesteps else self.agent.settings.horizon,
                              render=render, verbose=1, controller=controller)
        self.save()

    def run_iters(self, num_iters, env=None, render=True, timesteps=None, request_deterministic_start=True,
                  **kwargs):
        num_iters = int(num_iters)
        for n in tf.range(num_iters):
            tf.print("Starting iteration", n + 1)
            self.reload()  # required to avoid tensorflow error due to non existent variables
            self.agent.fit()
            if kwargs:
                self.agent.optimize(**kwargs)
            else:
                self.agent.optimize(**DEFAULT_OPTIMIZE_KWARGS)
            if self.agent.settings.optimizer == 'hybrid':
                self.agent.auxiliary_optimize(iters=500, verbose=3)

            self.agent.experiment(env, timesteps=timesteps if timesteps else self.agent.settings.horizon,
                                  render=render, verbose=3, request_deterministic_start=request_deterministic_start)
            self.save()
            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()
        self.agent.to_best_params(pop_size=0)
        return self.agent

    def save(self, save_name=None):
        self.agent.save(save_name=save_name)

    def reload(self):
        self.agent = Agent.load(self.agent.settings.save_name)

    @classmethod
    def load(cls, load_name):
        agent = Agent.load(load_name)
        return cls(agent)
