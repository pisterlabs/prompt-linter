import multiprocessing
import multiprocessing.connection
# from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines import bench
from environments.warehouse_memory.mini_warehouse import MiniWarehouse
from environments.sumo.LoopNetwork import LoopNetwork
import os
# from gym_minigrid.wrappers import *
from gym import wrappers

def worker_process(remote: multiprocessing.connection.Connection, parameters,
                   worker_id, seed):
    """
    This function is used as target by each of the threads in the multiprocess
    to build environment instances and define the commands that can be executed
    by each of the workers.
    """
    # The Atari wrappers are now imported from openAI baselines
    # https://github.com/openai/baselines
    log_dir = './log'
    if parameters['env_type'] == 'atari':
        env = make_atari(parameters['scene'])
        env = bench.Monitor(
                    env,
                    os.path.join(log_dir, str(worker_id)),
                    allow_early_resets=False)
        env = wrap_deepmind(env, True)
    if parameters['env_type'] == 'warehouse':
        env = MiniWarehouse(seed)
    if parameters['env_type'] == 'sumo':
        env = LoopNetwork(parameters, seed)
    if parameters['env_type'] == 'minigrid':
        env = gym.make(parameters['scene'])
        # env = RGBImgPartialObsWrapper(env, tile_size=12) # Get pixel observations
        env = ImgObsWrapper(env) # Get rid of the 'mission' field
        env = wrappers.GrayScaleObservation(env, keep_dim=True) # Gray scale
        env = FeatureVectorWrapper(env)
        env.seed(seed)
        
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            obs, reward, done, info = env.step(data)
            if done:
                obs = env.reset()
            remote.send((obs, reward, done, info))
        elif cmd == 'reset':
            remote.send(env.reset())
        elif cmd == 'action_space':
            remote.send(env.action_space.n)
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
    def __init__(self, parameters, worker_id, seed):

        self.child, parent = multiprocessing.Pipe()
        self.process = multiprocessing.Process(target=worker_process,
                                               args=(parent, parameters,
                                                     worker_id, seed))
        self.process.start()


# class FeatureVectorWrapper(gym.core.ObservationWrapper):
#     """
#     Use the image as the only observation output, no language/mission.
#     """

#     def __init__(self, env):
#         super().__init__(env)
#         obs_shape = env.observation_space.shape
#         self.observation_space = obs_shape[0]*obs_shape[1]

#     def observation(self, obs):
#         return np.reshape(obs,-1)