# PIRL
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-1-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

Probabilistic inference for reinforcement learning using arbitrary probabilistic models.

<img style="margin-right: auto;" src="https://user-images.githubusercontent.com/74900668/170347036-1aaae5ea-c02e-445e-96ea-c35e60b22509.svg" width="420"/> <img style="margin-left: auto;" src="https://user-images.githubusercontent.com/74900668/170347151-c6022103-1ae9-4512-80d1-2373adb05f65.gif" width="380"/> 
<br/>

# Installing PIRL

Please ensure that **Python 3.8** is installed and install PIRL by running

    pip install -e .

This will also automatically install the python dependencies required to run PIRL itself.

# PIRL ingredients

A [`PirlSettings`](https://github.com/Probabilistic-ML/pirl/blob/main/pirl/settings.py#L19) instance is required to initialize an instance of PIRL. 
The settings need to specify the following main components:

- method for propagation of uncertainty: trajectory sampling, moment matching, GP-EKF, GP-UKF or GP-PF
- probabilistic model, e.g., DGCN, BNN or GP
- reward, e.g., [`ExponentialReward`](https://github.com/Probabilistic-ML/pirl/blob/main/pirl/rewards/base.py#L43)
- policy, e.g., [`RBFController`](https://github.com/Probabilistic-ML/pirl/blob/main/pirl/controllers/base.py#L248) or [`NNController`](https://github.com/Probabilistic-ML/pirl/blob/main/pirl/controllers/base.py#L316)
- environment for experiments
- optimizer, e.g., `Adam` optimizer from Tensorflow or `'DE'` for differential evolution from Tensorflow Probability

Please note that moment matching is only applicable in combination with a Gaussian process with squared exponential kernel.

Generally, it is only necessary to provide a gym-like environment to perform probabilistic reinforcement learning using PIRL. More precisely, an environment is expected to possess

- the methods  `reset`, `render`, `step` and `close` (if `close_env=True` in `PirlSettings`),
- a `state` attribute,
- an `action_space` attribute with a sample method.

# Examples

## Additional requirements

The enclosed examples use **gym** as well as the physics engines **mujoco** and **pybullet**.

In order to install mujoco [download version 2.1](https://github.com/deepmind/mujoco/releases) and move the extracted `mujoco210` directory to `~/.mujoco/mujoco210`. Furthermore, add the path to the environment variable `LD_LIBRARY_PATH` (Linux) or the system variable `PATH` (Windows). 

The python packages [gym](https://github.com/openai/gym) and [mujoco-py](https://github.com/openai/mujoco-py) will be added by running

    pip install -r optional-requirements.txt

Finally, [pybullet-gym](https://github.com/benelot/pybullet-gym) needs to be installed manually with

    git clone https://github.com/benelot/pybullet-gym.git
    cd pybullet-gym
    pip install -e .

[pybullet](https://github.com/bulletphysics/bullet3) will be installed automatically.

## Running experiments

Experiments for the provided examples can be run using

    cd examples
    python run_example.py
    -A APPROACH, --approach APPROACH
                          Uncertainty propagation method. One of: Sampling, MomentMatching, PF, EKF, UKF. The last ones correspond to particle, extended Kalman and unscented Kalman filters, respectively. Note that MomentMatching is only
                          compatible with the squared_exponential model.
    -M MODEL, --model MODEL
                          Name of the model to be used. One of: dgcn, squared_exponential, bnn, exponential, matern32, matern52.
    -E ENVIRONMENT, --env ENVIRONMENT
                          Name of the environment. One of: InvPendulumSwingUp, InvDoublePendulum, ContinuousMountainCar, Pendulum
    -B, --background      Deactivates rendering the experiments.
    -N N_INIT, --init N_INIT
                          Number of initial experiments
    -T N_ITER, --iter N_ITER
                          Number of iterations (policy updates)
    -R RESTARTS, --restarts RESTARTS
                          Number of restarts for the inner optimization. Has a large impact on the duration
    -V VERBOSE, --verbose VERBOSE
                          Level of verbosity (0 for non-verbose)

# DGCN

The current implementation of DGCN is enclosed as a compiled binary file. In order to execute the DGCN code **Python 3.8** is necessary.

## Details on usage of DGCN

A DGCN instance is initialized in use of 

    model = DGCN(X, y, num_neurons=20)

where `X` and `y` are the training samples and labels, respectively, and `num_neurons` specifies the number of neurons in the hidden layers of the neural network.

Afterwards, model training takes place in use of the fit method

    model.fit(max_epochs=500, batch_size=None, noise=False)

where max_epochs is the maximal number of epochs, batch_size enables to use batch training with a given batch size and noise specifies if the model is supposed to learn aleatoric uncertainty.

Finally, the predict method is used to make predictions on test samples

    y_pred, var = model.predict(X, pred_var=True)


where `y_pred` is the mean prediction of the model, `var` denotes the variance of the predictions, `X` are the test samples. `var` is `None` if `pred_var` is `False`. Instead of the predict method, it is also possible to use `_predict` which will not cast `X` to the required datatype before prediction.
This increases the speed but may result in undefined behaviour in case of wrong types.

# Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/rvosshall"><img src="https://avatars.githubusercontent.com/u/74900668?v=4?s=100" width="100px;" alt="Robert Voßhall"/><br /><sub><b>Robert Voßhall</b></sub></a><br /><a href="https://github.com/Probabilistic-ML/pirl/commits?author=rvosshall" title="Code">💻</a> <a href="https://github.com/Probabilistic-ML/pirl/commits?author=rvosshall" title="Documentation">📖</a> <a href="#data-rvosshall" title="Data">🔣</a> <a href="#ideas-rvosshall" title="Ideas, Planning, & Feedback">🤔</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
