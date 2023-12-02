"""

Description: New version, more clear

Date: 07/02/2023

Modification:

"""

import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser


from agent_utilities import TD3, TD3AE
from openAI_memory_utilities import Memory, FrameStack


def define_set_seed(env, seed):
    env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def plot_functions(total_reward, env_name, style):
    plt.title("Rewards")
    plt.plot(total_reward)
    plt.savefig(f"plot_results/{style}_TD3_{env_name}_reward_curve.png")
    np.savetxt(f"plot_results/{style}_TD3_{env_name}_reward_curve.txt", total_reward)
    print("plots have been saved...")
    plt.show()


def define_parse_args():
    parser = ArgumentParser()
    parser.add_argument('--k',           type=int, default=3)
    parser.add_argument('--G',           type=int, default=10)
    parser.add_argument('--batch_size',  type=int, default=32) #256
    parser.add_argument('--seed',        type=int, default=0)

    parser.add_argument('--memory_size',           type=int, default=int(1e6))
    parser.add_argument('--max_exploration_steps', type=int, default=int(3000))
    parser.add_argument('--max_training_steps',    type=int, default=int(80e3))  # 150e3

    parser.add_argument('--env_name',   type=str, default='Pendulum-v1')  # BipedalWalker-v3, Pendulum-v1
    parser.add_argument('--train_mode', type=str, default='autoencoder')  # normal, autoencoder
    args   = parser.parse_args()
    return args


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args   = define_parse_args()

    env      = gym.make(args.env_name)
    env_name = args.env_name

    define_set_seed(env, args.seed)
    memory_buffer = Memory(args.memory_size, device)

    if args.train_mode == "autoencoder":
        print("AE TD3")
        agent        = TD3AE(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high.max(), device, env_name)
        frames_stack = FrameStack(args.k, env)
        state        = frames_stack.reset()

    else:
        print("Normal TD3")
        agent = TD3(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high.max(), device, env_name)
        state = env.reset()


    total_rewards      = []
    episode_num        = 0
    episode_reward     = 0
    episode_time_steps = 0
    done = False

    for t in range(args.max_training_steps):
        episode_time_steps += 1

        if t < args.max_exploration_steps:
            print(f"exploration steps:{t}/{args.max_exploration_steps}")
            action = env.action_space.sample()
        else:
            action = agent.get_action_from_policy(state)
            noise = np.random.normal(0, scale=0.1 * env.action_space.high.max(), size=env.action_space.shape[0])
            action = action + noise
            action = np.clip(action, -env.action_space.high.max(), env.action_space.high.max())

        if args.train_mode == "autoencoder":
            new_state, reward, done, _ = frames_stack.step(action)
        else:
            new_state, reward, done, _ = env.step(action)

        memory_buffer.save_experience_to_buffer(state, action, reward, new_state, done)
        state = new_state
        episode_reward += reward

        if t >= args.max_exploration_steps:
            for _ in range(1, args.G + 1):
                agent.update_policy(memory_buffer, args.batch_size)

        if done:
            print(f"Total Steps: {t + 1}/{args.max_training_steps},  Episode Num: {episode_num + 1}, Steps per Episode: {episode_time_steps}, Reward: {episode_reward:.3f}")
            total_rewards.append(episode_reward)

            # Reset environment
            if args.train_mode == "autoencoder":
                state = frames_stack.reset()
            else:
                state = env.reset()

            done = False
            episode_reward = 0
            episode_time_steps = 0
            episode_num += 1

    agent.save_models()
    agent.plot_loss()
    plot_functions(total_rewards, env_name, args.train_mode)



if __name__ == '__main__':
    main()