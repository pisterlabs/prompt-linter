# The code is based on the ACKTR implementation from OpenAI baselines
# This code implements method by Kangin & Pugeault "On-Policy Trust Region Policy Optimisation with Replay Buffers"
import numpy as np
import tensorflow as tf
from baselines import logger
import baselines.common as common
from baselines.common import tf_util as U
from baselines.trpo_replay.filters import ZFilter
from data_collector import rollout
from baselines.common.cmd_util import make_mujoco_env
from baselines.trpo_replay.policies import GaussianMlpPolicy
from baselines.trpo_replay.value_functions import NeuralNetValueFunction
from baselines.trpo_replay.replay_buffer import ReplayBufferParameters, ReplayBuffer

def pathlength(path):
    return path["reward"].shape[0]# Loss function that we'll differentiate to get the policy gradient

def estimate_v_targets (vf, paths, parameters):
    vtargs = []
    for path in paths:
        rew_t = path["reward"]
        return_t = common.discount(rew_t, parameters.gamma)
        vtargs.append(return_t)
    return vtargs

def estimate_advantage_function (vf, paths, parameters):
    advs = []
    for path in paths:
        rew_t = path["reward"]
        print('len(rew_t): ', len(rew_t))
        return_t = common.discount(rew_t, parameters.gamma)
        vpred_t = vf.predict(path)
        vpred_t = np.append(vpred_t, 0.0 if path["terminated"] else vpred_t[-1])
        delta_t = rew_t + parameters.gamma*vpred_t[1:] - vpred_t[:-1]
        adv_t = common.discount(delta_t, parameters.gamma * parameters.lam)
        print ('len(adv_t): ', len(adv_t))
        advs.append(adv_t)
    return advs

def collect_paths (env, policy, max_pathlength, parameters, obfilter, animate):
    timesteps_this_batch = 0
    paths = []
    i  = 0
    while True:
        path = rollout(env, policy, max_pathlength, animate=(len(paths)==0 and animate), obfilter=obfilter)
        paths.append(path)
        n = pathlength(path)
        timesteps_this_batch += n
        i = i + 1
        if timesteps_this_batch > parameters.timesteps_per_batch:
            break
    return timesteps_this_batch, paths

def log_results (paths, kl, timesteps_so_far, tr_index, logger):
    logger.record_tabular('EpRewMean', np.mean([path["reward"].sum() for path in paths]))
    logger.record_tabular('EpRewSEM', np.std([path["reward"].sum()/np.sqrt(len(paths)) for path in paths]))
    logger.record_tabular('EpLenMean', np.mean([pathlength(path) for path in paths]))
    logger.record_tabular('KL', kl)
    logger.record_tabular('TRIndex', tr_index)
    logger.record_tabular('TimestepsSoFar', timesteps_so_far)
    logger.dump_tabular()

def compute_kl (policy, stepsize, ob_no, oldac_dist, advs_new_flag, logger, parameters):
    #print ('oldac_dist: ', oldac_dist)
    print ('advs_new_flag: ', advs_new_flag)
    print ('len(advs_new_flag): ', len(advs_new_flag))
    print ('len(oldac_dist): ', len(oldac_dist))
    kl = policy.compute_kl(ob_no, oldac_dist, 1.0, advs_new_flag)
    return kl


def forfor(a):
    return [item for sublist in a for item in sublist]

def update_policy (paths, advs, advs_new_flag, policy, stepsize, do_update, logger, parameters):
    # Build arrays for policy update
    observations = np.concatenate([path["observation"] for path in paths])
    action = np.concatenate([path["action"] for path in paths])
    old_action_dist = np.concatenate([path["action_dist"] for path in paths])
    adv_n = np.concatenate(advs)
    standardized_adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)
    print (old_action_dist.shape[0])
    print (adv_n.shape[0])
    assert (old_action_dist.shape[0] == adv_n.shape[0])
    # Policy update
    for _ in range (5): do_update(observations, action, standardized_adv_n, advs_new_flag, old_action_dist, 1.0)
    kl = compute_kl (policy, stepsize, observations, old_action_dist, advs_new_flag, logger, parameters)
    opt = policy.get_loss(observations, action, standardized_adv_n, old_action_dist)
    return kl, opt

def learn(env, policy, vf, parameters, callback=None):
    policy_weights = policy.get_trainable_weights ()
    #policy_previous_weights = policy_previous.get_trainable_weights()
    #update_previous_weights_op = policy.get_target_updates(policy_weights, policy_previous_weights) 
    #update_new_weights_op = policy_previous.get_target_updates(policy_previous_weights, policy_weights)

    obfilter = ZFilter(env.observation_space.shape)

    max_pathlength = env.spec.timestep_limit
    stepsize = tf.Variable(initial_value=np.float32(np.array(1e-2)), name='stepsize')
    do_update, q_runner = policy.create_policy_optimiser (stepsize)

    # start queue runners
    enqueue_threads = []
    coord = tf.train.Coordinator()
    for qr in [q_runner]:
        assert (qr != None)
        enqueue_threads.extend(qr.create_threads(tf.get_default_session(), coord=coord, start=True))

    advantage_buffer_parameters = ReplayBufferParameters (parameters.advantage_replay_buffer_size, parameters.advantage_sample_size)
    advantage_function_buffer = ReplayBuffer (advantage_buffer_parameters)
    policy_buffer_parameters = ReplayBufferParameters (parameters.policy_replay_buffer_size)
    policy_function_buffer = ReplayBuffer (policy_buffer_parameters)    
    #policy_model_buffer = ReplayBuffer (policy_buffer_parameters)
    policy_adv_function_buffer = ReplayBuffer (policy_buffer_parameters)
    bottomless_parameters = ReplayBufferParameters (10000000, 1000000)
    bottomless_buffer = ReplayBuffer (bottomless_parameters)
    i = 0
    timesteps_so_far = 0
    while timesteps_so_far < parameters.num_timesteps:
        logger.log("********** Iteration %i ************"%i)

        timesteps_this_batch, paths = collect_paths (env, policy, max_pathlength, parameters, obfilter, animate=((i % 10 == 0) and parameters.animate)) 
        
        timesteps_so_far += timesteps_this_batch
        
        advantage_function_buffer.push (paths)
        policy_function_buffer.push (paths)
        bottomless_buffer.push (paths)
        paths_vbuf = advantage_function_buffer.sample()
        paths_vbuf = [j_ind for ind in paths_vbuf for j_ind in ind] 
        vtargs = estimate_v_targets (vf, paths_vbuf, parameters)
        
        vf.fit(paths_vbuf, vtargs)
        policy_adv_function_buffer.clear ()
        for j in range(len(policy_function_buffer.get())):
            print ('#', j)
            policy_adv_function_buffer.push (estimate_advantage_function (vf, policy_function_buffer.get()[j], parameters))
          
        optima = []
        updated_policies_list = []
        #print ('policy_adv_function_buffer.get()', policy_adv_function_buffer.get())
        #for j in range(len(policy_adv_function_buffer.get())):
        #    print ('policy_adv_function_buffer.get()[j].shape', policy_adv_function_buffer.get()[j].shape)
        advs = forfor(policy_adv_function_buffer.get())
        advs_new_flag = np.array([False] * np.concatenate(advs).shape[0])
        last_adv_functions = np.concatenate(policy_adv_function_buffer.buff[-1]).shape[0]
        print (np.concatenate(policy_adv_function_buffer.buff[-1]).shape[0])
        #advs_new_flag[:] = True
        advs_new_flag [-last_adv_functions:]  = True
        kl, opt = update_policy (np.concatenate(policy_function_buffer.get(), axis=0), advs, advs_new_flag, policy, stepsize, do_update, logger, parameters)
           
        log_results (paths, kl, timesteps_so_far, 0, logger)
        
        if callback:
            callback()
        i += 1
    np.save('bottomless_buffer.npy', bottomless_buffer.sample())
    coord.request_stop()
    coord.join(enqueue_threads)

def train(env_id, parameters, seed):
    env = make_mujoco_env(env_id, seed)

    with tf.Session(config=tf.ConfigProto()):
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        with tf.variable_scope("vf"):
            vf = NeuralNetValueFunction(ob_dim, ac_dim)
        
        with tf.variable_scope('pi'):
            policy = GaussianMlpPolicy(ob_dim, ac_dim)
        #with tf.variable_scope ('policy_prev'):
        #    policy_previous = GaussianMlpPolicy(ob_dim, ac_dim, name='policy_prev') 
        
        learn(env, policy=policy, vf=vf, parameters = parameters)

        env.close()
