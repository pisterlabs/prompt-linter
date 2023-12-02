import random
import time

import gym
import tensorflow as tf
import inspect
import numpy as np
import collections


def explained_variance(ypred, y):
    """
    From openai - baselines
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    """

    assert y.ndim == 1 and ypred.ndim == 1

    """s = time.perf_counter()
    _, vary = tf.nn.moments(y, axes=[0])
    _, varpred = tf.nn.moments(y - ypred, axes=[0])
    ev = tf.subtract(1.0,tf.divide(varpred, vary))
    print(time.perf_counter() - s)"""
    vary = np.var(y)
    ev = np.nan if vary == 0 else 1 - np.var(y - ypred) / vary
    return np.clip(ev, -1, 1)


def explained_variance_2d(ypred, y):
    assert y.ndim == 2 and ypred.ndim == 2
    vary = np.var(y, axis=0)
    out = 1 - np.var(y - ypred) / vary
    out[vary < 1e-10] = 0
    return out


def update(d, u):
    for k, v in u.items():
        # TODO - this might be retarded (loses flexibility?)
        #if isinstance(v, collections.Mapping):
        #
        #    d[k] = update(d.get(k, {}), v)
        #else:
        d[k] = v
    return d


def pollute_namespace(o, kwargs):
    for k, v in kwargs.items():
        setattr(o, k, v)


def arguments():
    """Retrieve locals from caller"""
    frame = inspect.currentframe()
    try:
        args = frame.f_back.f_locals["kwargs"]
    except KeyError:
        args = {}

    try:
        cls = frame.f_back.f_locals["__class__"]
        DEFAULTS = cls.DEFAULTS.copy()
    except KeyError:
        DEFAULTS = {}


    """self = frame.f_back.f_locals["self"]
    named_args = {k: v for k, v in frame.f_back.f_locals.copy().items() if k != "kwargs"}

    if not hasattr(self, "_hyperparameters"):
        setattr(self, "_hyperparameters", dict())

    _hyperparameters = getattr(self, "_hyperparameters")
    _hyperparameters.update(named_args)"""

    del frame

    args = update(DEFAULTS, args)
    return args

import collections

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, str(v)))
    return dict(items)

def hyperparameters_to_table(_hyperparameters):
    """
    The input is a dictionary of mixed things. We convert this to a 2d list of only ints, strs..
    :param _hyperparameters:
    :return:
    """


    flattened = flatten(_hyperparameters)
    data = [[k, v] for k, v in flattened.items()]
    data.insert(0, ["**Hyperparameter**", "**Value**"])
    return data



def get_defaults(o, additionally: dict):
    """Retrieve defaults for all of the classes in the inheritance hierarchy"""
    blacklist = ["tensorflow", "keras", "tf"]
    hierarchy = inspect.getmro(o.__class__)[:-1]
    hierarchy = [elem for elem in hierarchy if all(c not in elem.__module__ for c in blacklist)]

    DEFAULTS = {}
    for cls in hierarchy:
        update(DEFAULTS, cls.DEFAULTS)

    if additionally:
        update(DEFAULTS, additionally)

    return DEFAULTS


def is_gpu_faster(model, env_name):
    env = gym.make(env_name)
    obs = env.reset()
    test_times = 1000

    def test(fn):
        start = time.time()
        for i in range(test_times):
            fn()
        end = time.time()
        return end - start

    def train(model, observations):
        start = time.time()
        with tf.GradientTape() as tape:
            predicted_logits = model(observations)

            predicted_logits = predicted_logits["policy_logits"]
            # loss = tf.keras.losses.categorical_crossentropy(predicted_logits, predicted_logits, from_logits=True)
            loss = tf.reduce_mean(predicted_logits * predicted_logits)

        grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return time.time() - start

    devices = {"cpu": tf.device('/cpu:0'), "gpu": tf.device('/gpu:0')}

    for device_type, device in devices.items():
        with device:
            print("Inference: %s - %sms" % (device_type, test(lambda: model(obs[None, :]))))

            for n in [1, 16, 32, 64, 128, 256]:
                print("Training %s: %s - %sms" % (n, device_type, test(lambda: train(model, [obs for _ in range(n)]))))


if __name__ == "__main__":
    # print("--------\ntf.float16")
    # is_gpu_faster(PGPolicy(action_space=2,
    #                        dtype=tf.float16,
    #                        optimizer=tf.keras.optimizers.RMSprop(lr=0.001)), "CartPole-v0")
    # print("--------\ntf.float32")
    # is_gpu_faster(PGPolicy(action_space=2,
    #                        dtype=tf.float32,
    #                        optimizer=tf.keras.optimizers.RMSprop(lr=0.001)), "CartPole-v0")
    print("--------\ntf.float64")
    is_gpu_faster(PGPolicy(action_space=2,
                           dtype=tf.float64,
                           optimizer=tf.keras.optimizers.RMSprop(lr=0.001)), "CartPole-v0")
