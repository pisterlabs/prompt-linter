# Initially copied from OpenAI Baselines:
# /dependencies/baselines/baselines/trpo_mpi/trpo_mpi.py
from baselines.common import explained_variance, zipsame, dataset
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common import colorize
from mpi4py import MPI
from collections import deque
from baselines.common import set_global_seeds
from baselines.common.mpi_adam import MpiAdam
from baselines.common.cg import cg
from utils import observation_placeholder
from policies import build_policy
from classifiers import build_classifier
from contextlib import contextmanager
import sys; sys.path.insert(0, '..')
from optimizers import SVGD, Ensemble
from utils import save_state, FileWriter
from statistics import Statistics
from gym import spaces
import os
import pickle
import random


def traj_segment_generator(pi, env, horizon, stochastic, render=False):
    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    true_rew = 0.0
    ob = env.reset()

    cur_ep_true_ret = 0
    cur_ep_len = 0
    ep_true_rets = []
    ep_lens = []

    max_horizon = horizon + 1000  # since maximum episode length is 1000 in MuJoCo env

    # Initialize history arrays
    obs = np.array([ob.reshape(-1) for _ in range(max_horizon)])
    true_rews = np.zeros(max_horizon, 'float32')
    vpreds = np.zeros(max_horizon, 'float32')
    news = np.zeros(max_horizon, 'int32')
    acs = np.array([ac for _ in range(max_horizon)])
    prevacs = acs.copy()

    while True:
        if render:
            env.render()
        prevac = ac
        ac, vpred, _, _ = pi.step(ob, stochastic=stochastic)
        if isinstance(env.action_space, spaces.Discrete):
            ac = ac[0]
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t >= horizon and new:
            yield {"ob" : obs[:t], "true_rew" : true_rews[:t], "vpred" : vpreds[:t], "new" : news[:t],
                    "ac" : acs[:t], "prevac" : prevacs[:t], "nextvpred": vpred * (1 - new),
                    "ep_true_rets" : ep_true_rets, "ep_lens" : ep_lens}
            _, vpred, _, _ = pi.step(ob, stochastic=stochastic)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_true_rets = []
            ep_lens = []
            t = 0

        obs[t] = ob
        vpreds[t] = vpred
        news[t] = new
        acs[t] = ac
        prevacs[t] = prevac

        ob, true_rew, new, _ = env.step(ac)
        true_rews[t] = true_rew

        cur_ep_true_ret += true_rew
        cur_ep_len += 1
        if new:
            ep_true_rets.append(cur_ep_true_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_true_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1


def expert_traj_segment_generator(env, expert_trajs_path, timesteps_per_batch, num_expert_trajs):
    env_id = env.env.spec.id
    env_name = env_id.split('-')[0]
    if env_name not in ['Hopper', 'Walker2d', 'HalfCheetah', 'Ant', 'Humanoid',
                        'CartPole', 'MountainCar', 'Reacher']:
        raise NotImplementedError

    path = os.path.join(expert_trajs_path, 'trajs_'+env_name.lower()+'.pkl')
    with open(path, 'rb') as f:
        expert_trajs = pickle.load(f)[0:num_expert_trajs]

    obs, acs, ep_lens, ep_true_rets = [], [], [], []
    i = 0
    while True:
        ob, ac, ep_true_ret = expert_trajs[i]["ob"], expert_trajs[i]["ac"], expert_trajs[i]["ep_ret"]
        if isinstance(env.action_space, spaces.Discrete):
            ac = ac.reshape(-1)
        obs.append(ob)
        acs.append(ac)
        ep_lens.append(ob.shape[0])
        ep_true_rets.append(ep_true_ret)
        if sum(ep_lens) >= timesteps_per_batch:
            yield {"ob": np.concatenate(obs, axis=0), "ac": np.concatenate(acs, axis=0),
                   "ep_true_rets": ep_true_rets, "ep_lens": ep_lens}
            obs, acs, ep_lens, ep_true_rets = [], [], [], []

        i += 1
        if i == num_expert_trajs:
            random.shuffle(expert_trajs)
            i = 0


def add_vtarg_and_adv(seg, gamma, lam):
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def learn(*,
          policy_network,
          classifier_network,
          env,
          max_iters,
          timesteps_per_batch=1024, # what to train on
          max_kl=0.001,
          cg_iters=10,
          gamma=0.99,
          lam=1.0, # advantage estimation
          seed=None,
          entcoeff=0.0,
          cg_damping=1e-2,
          vf_stepsize=3e-4,
          vf_iters =3,
          expert_trajs_path='./expert_trajs',
          num_expert_trajs=500,
          g_step=1,
          d_step=1,
          classifier_entcoeff=1e-3,
          num_particles=5,
          d_stepsize=3e-4,
          max_episodes=0, total_timesteps=0,  # time constraint
          callback=None,
          load_path=None,
          save_path=None,
          render=False,
          use_classifier_logsumexp=True,
          use_reward_logsumexp=False,
          use_svgd=True,
          **policy_network_kwargs
          ):
    '''
    learn a policy function with TRPO algorithm
    
    Parameters:
    ----------

    network                 neural network to learn. Can be either string ('mlp', 'cnn', 'lstm', 'lnlstm' for basic types)
                            or function that takes input placeholder and returns tuple (output, None) for feedforward nets
                            or (output, (state_placeholder, state_output, mask_placeholder)) for recurrent nets

    env                     environment (one of the gym environments or wrapped via baselines.common.vec_env.VecEnv-type class

    timesteps_per_batch     timesteps per gradient estimation batch

    max_kl                  max KL divergence between old policy and new policy ( KL(pi_old || pi) )

    entcoeff                coefficient of policy entropy term in the optimization objective

    cg_iters                number of iterations of conjugate gradient algorithm

    cg_damping              conjugate gradient damping 

    vf_stepsize             learning rate for adam optimizer used to optimie value function loss

    vf_iters                number of iterations of value function optimization iterations per each policy optimization step

    total_timesteps         max number of timesteps

    max_episodes            max number of episodes
    
    max_iters               maximum number of policy optimization iterations

    callback                function to be called with (locals(), globals()) each policy optimization step

    load_path               str, path to load the model from (default: None, i.e. no model is loaded)

    **network_kwargs        keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network

    Returns:
    -------

    learnt model

    '''

    nworkers = MPI.COMM_WORLD.Get_size()
    if nworkers > 1:
        raise NotImplementedError
    rank = MPI.COMM_WORLD.Get_rank()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.49)
    U.get_session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))

    policy = build_policy(env, policy_network, value_network='copy', **policy_network_kwargs)
    set_global_seeds(seed)

    np.set_printoptions(precision=3)
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space

    ob = observation_placeholder(ob_space)
    with tf.variable_scope("pi"):
        pi = policy(observ_placeholder=ob)
    with tf.variable_scope("oldpi"):
        oldpi = policy(observ_placeholder=ob)

    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    entbonus = entcoeff * meanent

    vferr = tf.reduce_mean(tf.square(pi.vf - ret))

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # advantage * pnew / pold
    surrgain = tf.reduce_mean(ratio * atarg)

    optimgain = surrgain + entbonus
    losses = [optimgain, meankl, entbonus, surrgain, meanent]
    loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]

    dist = meankl

    all_var_list = get_trainable_variables("pi")
    # var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("pol")]
    # vf_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("vf")]
    var_list = get_pi_trainable_variables("pi")
    vf_var_list = get_vf_trainable_variables("pi")

    vfadam = MpiAdam(vf_var_list)

    get_flat = U.GetFlat(var_list)
    set_from_flat = U.SetFromFlat(var_list)
    klgrads = tf.gradients(dist, var_list)
    flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
    shapes = [var.get_shape().as_list() for var in var_list]
    start = 0
    tangents = []
    for shape in shapes:
        sz = U.intprod(shape)
        tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
        start += sz
    gvp = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zipsame(klgrads, tangents)]) #pylint: disable=E1111
    fvp = U.flatgrad(gvp, var_list)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(get_variables("oldpi"), get_variables("pi"))])

    compute_losses = U.function([ob, ac, atarg], losses)
    compute_lossandgrad = U.function([ob, ac, atarg], losses + [U.flatgrad(optimgain, var_list)])
    compute_fvp = U.function([flat_tangent, ob, ac, atarg], fvp)
    compute_vflossandgrad = U.function([ob, ret], U.flatgrad(vferr, vf_var_list))

    D = build_classifier(env, classifier_network, num_particles,
                         classifier_entcoeff, use_classifier_logsumexp, use_reward_logsumexp)
    grads_list, vars_list = D.get_grads_and_vars()

    if use_svgd:
        optimizer = SVGD(grads_list, vars_list, lambda: tf.train.AdamOptimizer(learning_rate=d_stepsize))
    else:
        optimizer = Ensemble(grads_list, vars_list, lambda: tf.train.AdamOptimizer(learning_rate=d_stepsize))

    @contextmanager
    def timed(msg):
        if rank == 0:
            print(colorize(msg, color='yellow'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds"%(time.time() - tstart), color='blue'))
        else:
            yield

    def allmean(x):
        assert isinstance(x, np.ndarray)
        out = np.empty_like(x)
        MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
        out /= nworkers
        return out

    U.initialize()

    if rank == 0:
        saver = tf.train.Saver(var_list=get_variables("pi"), max_to_keep=10000)
        writer = FileWriter(os.path.join(save_path, 'logs'))
        stats = Statistics(scalar_keys=["average_return", "average_episode_length"])

    if load_path is not None:
        # pi.load(load_path)
        saver.restore(U.get_session(), load_path)

    th_init = get_flat()
    MPI.COMM_WORLD.Bcast(th_init, root=0)
    set_from_flat(th_init)
    vfadam.sync()
    print("Init param sum", th_init.sum(), flush=True)

    # Prepare for rollouts
    # ----------------------------------------
    if load_path is not None:
        seg_gen = traj_segment_generator(pi, env, 1, stochastic=False, render=render)
    else:
        seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True, render=render)
    seg_gen_e = expert_traj_segment_generator(env, expert_trajs_path, timesteps_per_batch, num_expert_trajs)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=40) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=40) # rolling buffer for episode rewards

    if sum([max_iters>0, total_timesteps>0, max_episodes>0])==0:
        # nothing to be done
        return pi

    assert sum([max_iters>0, total_timesteps>0, max_episodes>0]) < 2, \
        'out of max_iters, total_timesteps, and max_episodes only one should be specified'

    while True:
        if callback: callback(locals(), globals())
        if total_timesteps and timesteps_so_far >= total_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        logger.log("********** Iteration %i ************"%iters_so_far)

        if iters_so_far % 500 == 0 and save_path is not None and load_path is None:
            fname = os.path.join(save_path, 'checkpoints', 'checkpoint')
            save_state(fname, saver, iters_so_far)

        with timed("sampling"):
            seg = seg_gen.__next__()

        if load_path is not None:
            iters_so_far += 1
            logger.record_tabular("EpRew", int(np.mean(seg["ep_true_rets"])))
            logger.record_tabular("EpLen", int(np.mean(seg["ep_lens"])))
            logger.dump_tabular()
            continue

        seg["rew"] = D.get_reward(seg["ob"], seg["ac"])

        add_vtarg_and_adv(seg, gamma, lam)

        ob, ac, ep_lens, atarg, tdlamret = seg["ob"], seg["ac"], seg["ep_lens"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate

        if hasattr(pi, "ret_rms"): pi.ret_rms.update(tdlamret)
        if hasattr(pi, "rms"): pi.rms.update(ob) # update running mean/std for policy

        args = seg["ob"], seg["ac"], atarg
        fvpargs = [arr[::5] for arr in args]

        def fisher_vector_product(p):
            return allmean(compute_fvp(p, *fvpargs)) + cg_damping * p

        assign_old_eq_new() # set old parameter values to new parameter values
        with timed("computegrad"):
            *lossbefore, g = compute_lossandgrad(*args)
        lossbefore = allmean(np.array(lossbefore))
        g = allmean(g)
        if np.allclose(g, 0):
            logger.log("Got zero gradient. not updating")
        else:
            with timed("cg"):
                stepdir = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose=rank==0)
            assert np.isfinite(stepdir).all()
            shs = .5*stepdir.dot(fisher_vector_product(stepdir))
            lm = np.sqrt(shs / max_kl)
            # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
            fullstep = stepdir / lm
            expectedimprove = g.dot(fullstep)
            surrbefore = lossbefore[0]
            stepsize = 1.0
            thbefore = get_flat()
            for _ in range(10):
                thnew = thbefore + fullstep * stepsize
                set_from_flat(thnew)
                meanlosses = surr, kl, *_ = allmean(np.array(compute_losses(*args)))
                improve = surr - surrbefore
                logger.log("Expected: %.3f Actual: %.3f"%(expectedimprove, improve))
                if not np.isfinite(meanlosses).all():
                    logger.log("Got non-finite value of losses -- bad!")
                elif kl > max_kl * 1.5:
                    logger.log("violated KL constraint. shrinking step.")
                elif improve < 0:
                    logger.log("surrogate didn't improve. shrinking step.")
                else:
                    logger.log("Stepsize OK!")
                    break
                stepsize *= .5
            else:
                logger.log("couldn't compute a good step")
                set_from_flat(thbefore)
            if nworkers > 1 and iters_so_far % 20 == 0:
                paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), vfadam.getflat().sum())) # list of tuples
                assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])

        for (lossname, lossval) in zip(loss_names, meanlosses):
            logger.record_tabular(lossname, lossval)

        with timed("vf"):

            for _ in range(vf_iters):
                for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]),
                                                         include_final_partial_batch=False, batch_size=1000):
                    g = allmean(compute_vflossandgrad(mbob, mbret))
                    vfadam.update(g, vf_stepsize)

        with timed("sample expert trajectories"):
            ob_a, ac_a, ep_lens_a = ob, ac, ep_lens
            seg_e = seg_gen_e.__next__()
            ob_e, ac_e, ep_lens_e = seg_e["ob"], seg_e["ac"], seg_e["ep_lens"]

        if hasattr(D, "rms"):
            obs = np.concatenate([ob_a, ob_e], axis=0)
            if isinstance(ac_space, spaces.Box):
                acs = np.concatenate([ac_a, ac_e], axis=0)
                D.rms.update(np.concatenate([obs, acs], axis=1))
            elif isinstance(ac_space, spaces.Discrete):
                D.rms.update(obs)
            else:
                raise NotImplementedError

        with timed("SVGD"):
            sess = tf.get_default_session()
            feed_dict = {D.Xs['a']: ob_a, D.As['a']: ac_a, D.Ls['a']: ep_lens_a,
                         D.Xs['e']: ob_e, D.As['e']: ac_e, D.Ls['e']: ep_lens_e}
            for _ in range(d_step):
                sess.run(optimizer.update_op, feed_dict=feed_dict)

        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))

        lrlocal = (seg["ep_lens"], seg["ep_true_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)

        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1

        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)

        if rank == 0:
            logger.dump_tabular()
            stats.add_all_summary(writer, [np.mean(rewbuffer), np.mean(lenbuffer)], iters_so_far)
            rewbuffer.clear()
            lenbuffer.clear()

    return pi


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]


def get_variables(scope):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)


def get_trainable_variables(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)


def get_vf_trainable_variables(scope):
    return [v for v in get_trainable_variables(scope) if 'vf' in v.name[len(scope):].split('/')]    


def get_pi_trainable_variables(scope):
    return [v for v in get_trainable_variables(scope) if 'pi' in v.name[len(scope):].split('/')]    

