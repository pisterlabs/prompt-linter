import multiprocessing
import multiprocessing.connection
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines import bench
import os


def worker_process(remote: multiprocessing.connection.Connection, parameters,
                   worker_id, env):
    """
    This function is used as target by each of the threads in the multiprocess
    to build environment instances and define the commands that can be executed
    by each of the workers.
    """
    # The Atari wrappers are now imported from openAI baselines
    # https://github.com/openai/baselines
    # log_dir = './log'
    # env = make_atari(parameters['scene'])
    # env = bench.Monitor(
    #             env,
    #             os.path.join(log_dir, str(worker_id)),
    #             allow_early_resets=False)
    # env = wrap_deepmind(env)
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            obs, reward, done, info = env.step(data)
            if done is True:
                obs = env.reset()
            remote.send((obs, reward, done, info))
        elif cmd == 'reset':
            remote.send(env.reset())
        elif cmd == 'action_space':
            remote.send(env.action_space)
        elif cmd == 'close':
            remote.close()
            break
        else:
            raise NotImplementedError


class Worker(object):
    """
    Creates workers (actors) and starts single parallel threads in the
    multiprocess. Commands can be send and outputs received by calling
    child.send() and child.recv() respectively
    """
    def __init__(self, env, parameters, worker_id):

        self.child, parent = multiprocessing.Pipe()
        self.process = multiprocessing.Process(target=worker_process,
                                               args=(parent, parameters,
                                                     worker_id, env))
        self.process.start()
