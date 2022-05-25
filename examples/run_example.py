import argparse

from base_settings import settings_dict

from pirl import PIRL
from pirl.agent import Agent

approaches = ['Sampling', 'EKF', 'UKF', 'PF', 'MomentMatching']  # 5 approaches
# 5 different kernels available for first 4 approaches; MomentMatching always uses SquaredExponential
model_names = ['dgcn', 'squared_exponential', 'bnn', 'exponential', 'matern32', 'matern52']


def validate_args(env_name, approach, model_name):
    if env_name not in settings_dict:
        raise ValueError("Only the following environments are supported: " + ', '.join(settings_dict.keys()) +
                         f". Got: {env_name}")
    if approach not in approaches:
        raise ValueError("approach argument must be one of " + ', '.join(approaches) + f". Got: {approach}")
    if model_name not in model_names:
        raise ValueError("model argument must be one of " + ', '.join(model_names) + f". Got: {approach}")


def main(env_name, approach, model_name, n_init, n_iter, restarts, render=True, verbose=3):
    validate_args(env_name, approach, model_name)
    settings = settings_dict[env_name](approach, model_name)
    agent = Agent(settings)
    pirl = PIRL(agent)
    pirl.run_inits(n_init, render=render)
    agent = pirl.run_iters(n_iter, restarts=restarts, iters=250, patience_stop=100,
                           patience_red=30, epsilon=1e-5, discount=0.6,
                           verbose=verbose, render=render)
    return agent, pirl, settings


def get_args():
    parser = argparse.ArgumentParser(
        description='PIRL - Probabilistic inference for reinforcement learning')
    parser.add_argument("-A", "--approach", dest="approach",
                        help="Uncertainty propagation method. One of: Sampling, MomentMatching, PF, EKF, UKF. "
                             "The last ones correspond to particle, extended Kalman and unscented Kalman filters, "
                             "respectively. Note that MomentMatching is only compatible with the "
                             "squared_exponential model.")
    parser.add_argument("-M", "--model", dest="model",
                        help="Name of the model to be used. One of:  dgcn, squared_exponential, bnn, exponential, "
                             "matern32, matern52.")
    parser.add_argument("-E", "--env", dest="environment",
                        help="Name of the environment. One of: " + ', '.join(settings_dict.keys()))

    parser.add_argument("-B", "--background", dest="render", action="store_false",
                        help="Deactivates rendering the experiments.")

    parser.add_argument("-N", "--init", dest="n_init", type=int, help="Number of initial experiments", default=1)
    parser.add_argument("-T", "--iter", dest="n_iter", type=int, help="Number of iterations (policy updates)",
                        default=10)
    parser.add_argument("-R", "--restarts", dest="restarts", type=int, default=4,
                        help="Number of restarts for the inner optimization. Has a large impact on the duration")
    parser.add_argument("-V", "--verbose", dest="verbose", type=int, help="Level of verbosity (0 for non-verbose)",
                        default=3)
    parser.set_defaults(approach="Sampling", model="dgcn", n_init=1, n_iter=10, restarts=4, render=True, verbose=3)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    main(args.environment, args.approach, args.model, args.n_init, args.n_iter, args.restarts, render=args.render,
         verbose=args.verbose)
