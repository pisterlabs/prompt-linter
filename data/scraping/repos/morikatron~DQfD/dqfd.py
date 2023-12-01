# This code is based on code from OpenAI baselines. (https://github.com/openai/baselines)
import os.path as osp
from tqdm import tqdm
from time import time
from collections import deque
import pickle
import tensorflow as tf
import numpy as np

from common.schedules import LinearSchedule, ConstantSchedule
from common.misc_util import set_global_seeds, timedelta
from common import logger

from replay_buffer import PrioritizedReplayBuffer
# from stable_baselines.common.buffers import PrioritizedReplayBuffer

from models import build_q_func
from dqfd_learner import DQfD


def get_n_step_sample(buffer, gamma):
    reward_n = 0
    for i, step in enumerate(buffer):
        reward_n += step[2] * (gamma ** i)
    obs        = buffer[0][0]
    action     = buffer[0][1]
    rew        = buffer[0][2]
    new_obs    = buffer[0][3]
    done       = buffer[0][4]
    is_demo    = buffer[0][5]
    n_step_obs = buffer[-1][3]
    done_n     = buffer[-1][4]
    return obs[0], action, rew, new_obs[0], float(done), float(is_demo), n_step_obs[0], reward_n, done_n


def learn(env,
          network,
          seed=None,
          lr=5e-5,
          total_timesteps=100000,
          buffer_size=500000,
          exploration_fraction=0.1,
          exploration_final_eps=0.01,
          train_freq=1,
          batch_size=32,
          print_freq=10,
          checkpoint_freq=100000,
          checkpoint_path=None,
          learning_starts=0,
          gamma=0.99,
          target_network_update_freq=10000,
          prioritized_replay=True,
          prioritized_replay_alpha=0.4,
          prioritized_replay_beta0=0.6,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-3,
          param_noise=False,
          callback=None,
          load_path=None,
          load_idx=None,
          demo_path=None,
          n_step=10,
          demo_prioritized_replay_eps=1.0,
          pre_train_timesteps=750000,
          epsilon_schedule="constant",
          **network_kwargs
          ):
    # Create all the functions necessary to train the model
    set_global_seeds(seed)
    q_func = build_q_func(network, **network_kwargs)

    with tf.device('/GPU:0'):
        model = DQfD(
            q_func=q_func,
            observation_shape=env.observation_space.shape,
            num_actions=env.action_space.n,
            lr=lr,
            grad_norm_clipping=10,
            gamma=gamma,
            param_noise=param_noise
        )

    # Load model from checkpoint
    if load_path is not None:
        load_path = osp.expanduser(load_path)
        ckpt = tf.train.Checkpoint(model=model)
        manager = tf.train.CheckpointManager(ckpt, load_path, max_to_keep=None)
        if load_idx is None:
            ckpt.restore(manager.latest_checkpoint)
            print("Restoring from {}".format(manager.latest_checkpoint))
        else:
            ckpt.restore(manager.checkpoints[load_idx])
            print("Restoring from {}".format(manager.checkpoints[load_idx]))

    # Setup demo trajectory
    assert demo_path is not None
    with open(demo_path, "rb") as f:
        trajectories = pickle.load(f)

    # Create the replay buffer
    replay_buffer = PrioritizedReplayBuffer(buffer_size, prioritized_replay_alpha)
    if prioritized_replay_beta_iters is None:
        prioritized_replay_beta_iters = total_timesteps
    beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                   initial_p=prioritized_replay_beta0,
                                   final_p=1.0)
    temp_buffer = deque(maxlen=n_step)
    is_demo = True
    for epi in trajectories:
        for obs, action, rew, new_obs, done in epi:
            obs, new_obs = np.expand_dims(np.array(obs), axis=0), np.expand_dims(np.array(new_obs), axis=0)
            if n_step:
                temp_buffer.append((obs, action, rew, new_obs, done, is_demo))
                if len(temp_buffer) == n_step:
                    n_step_sample = get_n_step_sample(temp_buffer, gamma)
                    replay_buffer.demo_len += 1
                    replay_buffer.add(*n_step_sample)
            else:
                replay_buffer.demo_len += 1
                replay_buffer.add(obs[0], action, rew, new_obs[0], float(done), float(is_demo))
    logger.log("trajectory length:", replay_buffer.demo_len)
    # Create the schedule for exploration
    if epsilon_schedule == "constant":
        exploration = ConstantSchedule(exploration_final_eps)
    else:  # not used
        exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                     initial_p=1.0,
                                     final_p=exploration_final_eps)

    model.update_target()

    # ============================================== pre-training ======================================================
    start = time()
    num_episodes = 0
    temp_buffer = deque(maxlen=n_step)
    for t in tqdm(range(pre_train_timesteps)):
        # sample and train
        experience = replay_buffer.sample(batch_size, beta=prioritized_replay_beta0)
        batch_idxes = experience[-1]
        if experience[6] is None:  # for n_step = 0
            obses_t, actions, rewards, obses_tp1, dones, is_demos = tuple(map(tf.constant, experience[:6]))
            obses_tpn, rewards_n, dones_n = None, None, None
            weights = tf.constant(experience[-2])
        else:
            obses_t, actions, rewards, obses_tp1, dones, is_demos, obses_tpn, rewards_n, dones_n, weights = tuple(map(tf.constant, experience[:-1]))
        td_errors, n_td_errors, loss_dq, loss_n, loss_E, loss_l2, weighted_error = model.train(obses_t, actions, rewards, obses_tp1, dones, is_demos, weights, obses_tpn, rewards_n, dones_n)

        # Update priorities
        new_priorities = np.abs(td_errors) + np.abs(n_td_errors) + demo_prioritized_replay_eps
        replay_buffer.update_priorities(batch_idxes, new_priorities)

        # Update target network periodically
        if t > 0 and t % target_network_update_freq == 0:
            model.update_target()

        # Logging
        elapsed_time = timedelta(time() - start)
        if print_freq is not None and t % 10000 == 0:
            logger.record_tabular("steps", t)
            logger.record_tabular("episodes", num_episodes)
            logger.record_tabular("mean 100 episode reward", 0)
            logger.record_tabular("max 100 episode reward", 0)
            logger.record_tabular("min 100 episode reward", 0)
            logger.record_tabular("demo sample rate", 1)
            logger.record_tabular("epsilon", 0)
            logger.record_tabular("loss_td", np.mean(loss_dq.numpy()))
            logger.record_tabular("loss_n_td", np.mean(loss_n.numpy()))
            logger.record_tabular("loss_margin", np.mean(loss_E.numpy()))
            logger.record_tabular("loss_l2", np.mean(loss_l2.numpy()))
            logger.record_tabular("losses_all", weighted_error.numpy())
            logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
            logger.record_tabular("pre_train", True)
            logger.record_tabular("elapsed time", elapsed_time)
            logger.dump_tabular()

    # ============================================== exploring =========================================================
    sample_counts = 0
    demo_used_counts = 0
    episode_rewards = deque(maxlen=100)
    this_episode_reward = 0.
    best_score = 0.
    saved_mean_reward = None
    is_demo = False
    obs = env.reset()
    # Always mimic the vectorized env
    obs = np.expand_dims(np.array(obs), axis=0)
    reset = True
    for t in tqdm(range(total_timesteps)):
        if callback is not None:
            if callback(locals(), globals()):
                break
        kwargs = {}
        if not param_noise:
            update_eps = tf.constant(exploration.value(t))
            update_param_noise_threshold = 0.
        else:  # not used
            update_eps = tf.constant(0.)
            update_param_noise_threshold = -np.log(1. - exploration.value(t) + exploration.value(t) / float(env.action_space.n))
            kwargs['reset'] = reset
            kwargs['update_param_noise_threshold'] = update_param_noise_threshold
            kwargs['update_param_noise_scale'] = True
        action, epsilon, _, _ = model.step(tf.constant(obs), update_eps=update_eps, **kwargs)
        action = action[0].numpy()
        reset = False
        new_obs, rew, done, _ = env.step(action)

        # Store transition in the replay buffer.
        new_obs = np.expand_dims(np.array(new_obs), axis=0)
        if n_step:
            temp_buffer.append((obs, action, rew, new_obs, done, is_demo))
            if len(temp_buffer) == n_step:
                n_step_sample = get_n_step_sample(temp_buffer, gamma)
                replay_buffer.add(*n_step_sample)
        else:
            replay_buffer.add(obs[0], action, rew, new_obs[0], float(done), 0.)
        obs = new_obs

        # invert log scaled score for logging
        this_episode_reward += np.sign(rew) * (np.exp(np.sign(rew) * rew) - 1.)
        if done:
            num_episodes += 1
            obs = env.reset()
            obs = np.expand_dims(np.array(obs), axis=0)
            episode_rewards.append(this_episode_reward)
            reset = True
            if this_episode_reward > best_score:
                best_score = this_episode_reward
                ckpt = tf.train.Checkpoint(model=model)
                manager = tf.train.CheckpointManager(ckpt, './best_model', max_to_keep=1)
                manager.save(t)
                logger.log("saved best model")
            this_episode_reward = 0.0

        if t % train_freq == 0:
            experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
            batch_idxes = experience[-1]
            if experience[6] is None:  # for n_step = 0
                obses_t, actions, rewards, obses_tp1, dones, is_demos = tuple(map(tf.constant, experience[:6]))
                obses_tpn, rewards_n, dones_n = None, None, None
                weights = tf.constant(experience[-2])
            else:
                obses_t, actions, rewards, obses_tp1, dones, is_demos, obses_tpn, rewards_n, dones_n, weights = tuple(
                    map(tf.constant, experience[:-1]))
            td_errors, n_td_errors, loss_dq, loss_n, loss_E, loss_l2, weighted_error = model.train(obses_t, actions, rewards, obses_tp1,
                                                                              dones, is_demos, weights, obses_tpn,
                                                                              rewards_n, dones_n)
            new_priorities = np.abs(td_errors) + np.abs(n_td_errors) + demo_prioritized_replay_eps * is_demos + prioritized_replay_eps * (1. - is_demos)
            replay_buffer.update_priorities(batch_idxes, new_priorities)

            # for logging
            sample_counts += batch_size
            demo_used_counts += np.sum(is_demos)

        if t % target_network_update_freq == 0:
            # Update target network periodically.
            model.update_target()

        if t % checkpoint_freq == 0:
            save_path = checkpoint_path
            ckpt = tf.train.Checkpoint(model=model)
            manager = tf.train.CheckpointManager(ckpt, save_path, max_to_keep=10)
            manager.save(t)
            logger.log("saved checkpoint")

        elapsed_time = timedelta(time() - start)
        if done and num_episodes > 0 and num_episodes % print_freq == 0:
            logger.record_tabular("steps", t)
            logger.record_tabular("episodes", num_episodes)
            logger.record_tabular("mean 100 episode reward", np.mean(episode_rewards))
            logger.record_tabular("max 100 episode reward", np.max(episode_rewards))
            logger.record_tabular("min 100 episode reward", np.min(episode_rewards))
            logger.record_tabular("demo sample rate", demo_used_counts / sample_counts)
            logger.record_tabular("epsilon", epsilon.numpy())
            logger.record_tabular("loss_td", np.mean(loss_dq.numpy()))
            logger.record_tabular("loss_n_td", np.mean(loss_n.numpy()))
            logger.record_tabular("loss_margin", np.mean(loss_E.numpy()))
            logger.record_tabular("loss_l2", np.mean(loss_l2.numpy()))
            logger.record_tabular("losses_all", weighted_error.numpy())
            logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
            logger.record_tabular("pre_train", False)
            logger.record_tabular("elapsed time", elapsed_time)
            logger.dump_tabular()

    return model




