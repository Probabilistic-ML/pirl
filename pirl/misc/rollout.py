import numpy as np
import time


def rollout(env, timesteps=None, substeps=1, controller=None, render=False, verbose=1,
            x_init=None, deplete_env=False, close=False, render_kwargs=None):
    U, R, X = [], [], []
    if not render_kwargs:
        render_kwargs = {}
    if render:
        try:
            env.render(**render_kwargs)
            X.append(np.array(env.reset()).ravel())  # x(t=0)
        except AttributeError:
            X.append(np.array(env.reset()).ravel())  # x(t=0)
            env.render(**render_kwargs)
    else:
        X.append(np.array(env.reset()).ravel())  # x(t=0)
    if x_init is not None:
        X[0] = x_init.copy().ravel()  # set x(t=0) to x_init
        env.state = x_init.copy()
    if timesteps is None:
        timesteps = 9999  # should be finite to force end of while loop
    assert timesteps > 0, "timesteps must be a positive"
    try:
        timesteps = int(timesteps)
    except (TypeError, ValueError):
        raise ValueError("timesteps must be an integer")

    assert substeps > 0, "substeps must be a positive"
    assert isinstance(substeps, int), "substeps must be an integer"
    for t in range(timesteps):
        if render:
            env.render(**render_kwargs)
            time.sleep(0.05)

        if controller is None:
            u = np.array(env.action_space.sample(), dtype=np.float64)
        else:
            u = np.array(controller.compute_action(X[-1].reshape(1, -1))).ravel()
        if verbose:
            print(f"t={t}, u={u}")

        for i in range(substeps):
            U.append(u.ravel())  # u(t)
            if 'discrete' in str(env.action_space.__class__):
                u_tmp = int(u)
            else:
                u_tmp = u
            x_new, r_new, done, _ = env.step(u_tmp)
            if not type(r_new) == np.ndarray:
                r_new = np.array(r_new)
            R.append(r_new.ravel())
            X.append(x_new.ravel())
        if render:
            env.render(*render_kwargs)
        if done:
            if verbose:
                print("Environment depleted!")
            break

    while deplete_env and not done:
        _, _, done, _ = env.step(env.action_space.sample())

    if close:
        try:
            env.close()
        except AttributeError:
            pass

    return min2darray(X), min2darray(U), min2darray(R)


def min2darray(vals):
    arr = np.array(vals)
    if arr.ndim < 2:
        arr = arr.reshape((-1, 1))
    return arr
