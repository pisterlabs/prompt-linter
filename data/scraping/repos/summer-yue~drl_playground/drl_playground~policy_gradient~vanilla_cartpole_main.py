# A simple policy gradient approach for CartPole based on OpenAI
# spinning up:
# https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#other-forms-of-the-policy-gradient
#
# The trained model is able to get solve CartPole under a specified
# default_max_steps.

import gym
import numpy as np
import tensorflow as tf

from absl import logging
from drl_playground.test_env.bandit import BanditEnv
from tqdm import tqdm

logging.set_verbosity(logging.INFO)

# Set up the gym environment and global variables related to the environment.
env = gym.make('CartPole-v0')

# Swap to a simple bandit testing environment.
# env = BanditEnv()
observation_dim = env.observation_space.shape[0]  # 4
actions_dim = env.action_space.n  # 2
default_max_steps = 100


def build_policy_net():
    """
    Build the model for the policy network. The input to the model is a batch of
        observations (None, 4,) and the output is a batch of actions (None, 2,).

    Returns:
        model(tf.keras.Model): the sequential policy network.

    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(6, batch_input_shape=(None, observation_dim)),
        tf.keras.layers.Activation('relu'),
        # Hidden layers. TODO: make the hidden layers neuron counts tunable.
        tf.keras.layers.Dense(actions_dim),
    ])
    return model


def compute_average_return(model, n_episodes, max_steps=default_max_steps,
                           render=False):
    """Computes the average cumulative rewards for a model among n episodes.

    Args:
        model(tf.keras.Model): the model to be evaluated.
        n_episodes(int): the number of episodes to run, defaults to 20.
        max_steps(int): the max number of steps before terminating an episode.
        render(bool): whether we render the CartPole environments while
            running these simulations.

    Returns:
        avg_return(float): the average cumulative reward for the n episodes.

    """
    sum_episodes_returns = 0
    for episode in range(n_episodes):
        episode_return = 0
        observation = env.reset()
        for t in range(max_steps):
            if render:
                env.render()

            action_logits = model.predict(np.expand_dims(observation, axis=0))[
                0]
            # Select the action greedily at inference time.
            action = np.argmax(action_logits)
            logging.debug("selected action: {}".format(action))
            observation, reward, done, _ = env.step(action)
            episode_return += reward
            if done:
                logging.info("Episode finished after {} time steps".format(t +
                                                                           1))
                break

        sum_episodes_returns += episode_return
        logging.info("The return for episode {} is {}".format(episode,
                                                              episode_return))

    avg_return = sum_episodes_returns * 1.0 / n_episodes

    return avg_return


@tf.function
def get_loss(reward_weights, action_logits, actions, in_progress):
    """Get the loss tensor (None, ) where None represents the batch size.

    This follows the simple policy gradient loss function from OpenAI
    spinning up:
    https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#other-forms-of-the-policy-gradient

    Args:
        reward_weights(Tensor): shape (None, 1), dtype float32, cumulative
            rewards per episode used as weights for the policy gradient. None
            represents the number of episodes.
        action_logits(Tensor): shape (None, None, 2), dtype float32, - (episode
            number, time step, the policy model's output logit).
        actions(Tensor): shape(None, None, ) dtype int64 - (batch size,
            time step, ), the true action that was taken at this step.
        in_progress(Tensor): shape (None, None, ), dtype float32, a 0/1 value
            indicating whether the episode was in progress at an action.
    Returns:
        A loss tensor with shape (None, ), dtype float 32 - (batch_size, ). The
            gradient of the defined loss is equivalent to the policy gradient.

    """
    # actions_one_hot shape: (batch_size, action_steps, action_dim)
    actions_one_hot = tf.one_hot(actions, depth=actions_dim,
                                 dtype=tf.float32, axis=-1)
    # masked_log_softmax shape: (batch_size, action_steps, 2)
    masked_log_softmax = tf.nn.log_softmax(action_logits) * tf.expand_dims(
        in_progress, -1)
    # log_probs shape: (batch_size, action_steps)
    log_probs = tf.reduce_sum(
        masked_log_softmax * tf.cast(
            actions_one_hot, dtype=tf.float32), axis=-1)
    # loss shape: (batch_size, )
    loss = -tf.reduce_mean(reward_weights * log_probs, axis=-1)
    return loss


def train(model, batch_size, max_steps=default_max_steps):
    """Perform one gradient update to the model for a batch of episodes.

    This follows the simple policy gradient loss function from OpenAI
    spinning up:
    https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#other-forms-of-the-policy-gradient

    Args:
        model(tf.keras.Model): the model to be trained, generated from
            build_policy_net.
        batch_size: the number of episodes in a batch.
        max_steps: the max number of steps the agent can take before we
            declare the game as "done".

    """
    # TODO: Run each episode in parallel to speed up training.
    # a list of action logits tensors for all actions in all episodes. Shape: (
    # batch_size, max_steps, action_logits_tensor).
    all_action_logits = []
    all_actions = []
    # a list of 1/0s representing whether the episode is still in progress or
    # has already finished, for all actions and all episodes.
    all_in_progress = []
    # a list of cumulative rewards tensors for all episodes. Shape (
    # batch_size, reward_tensor).
    all_rewards = []

    with tf.GradientTape() as tape:
        tape.watch(model.trainable_weights)
        for _ in range(batch_size):
            obs = env.reset()

            eps_rewards = 0
            eps_observations = []

            time = 0
            eps_action_logits = []
            eps_actions = []
            eps_in_progress = []
            done = False
            while time < max_steps:
                eps_in_progress.append(
                    tf.constant(int(not done), dtype=tf.float32))
                if done:
                    eps_action_logits.append(tf.constant(
                        0, dtype=tf.float32, shape=(actions_dim,)))
                    eps_actions.append(tf.constant(0, dtype=tf.int64))
                    eps_observations.append(tf.constant(0,
                                                        dtype=tf.float32,
                                                        shape=(
                                                            observation_dim,)))
                else:
                    eps_observations.append(obs)
                    # action_logit shape: (1, 2).
                    action_logit = model(np.expand_dims(obs, axis=0))
                    eps_action_logits.append(action_logit[0])

                    # TODO: add temperature for exploration tuning.
                    action = tf.random.categorical(action_logit,
                                                   num_samples=1)[0][0]
                    eps_actions.append(action)
                    obs, reward, done, _ = env.step(action.numpy())
                    eps_rewards += reward

                time += 1

            all_action_logits.append(eps_action_logits)
            all_actions.append(eps_actions)
            all_rewards.append([eps_rewards])
            all_in_progress.append(eps_in_progress)

        packed_all_action_logits = tf.stack(all_action_logits)
        packed_all_action_logits = tf.stack(packed_all_action_logits)
        packed_all_actions = tf.stack(all_actions)
        packed_all_actions = tf.stack(packed_all_actions)

        loss = get_loss(tf.stack(all_rewards),
                        tf.stack(packed_all_action_logits),
                        tf.stack(packed_all_actions),
                        tf.stack(all_in_progress))

    gradient = tape.gradient(loss, model.trainable_weights)
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    logging.debug("loss: {} \n, gradient: {} \n, trainable weights: {} "
                  "\n".format(
        loss, gradient, model.trainable_weights))
    opt.apply_gradients(zip(gradient, model.trainable_weights))


# Initialize the agent with random weights and evaluate its performance.
policy_net = build_policy_net()
random_model_reward = compute_average_return(policy_net, n_episodes=10)
logging.info("The average reward among all episodes for a randomly initialized "
             "model is {}".format(random_model_reward))

num_batch = 100
for i in tqdm(range(num_batch)):
    train(policy_net, batch_size=128)
trained_model_reward = compute_average_return(policy_net, n_episodes=10)
logging.info(
    "The average reward among all episodes for a trained model is {}.".format(
        trained_model_reward))
env.close()
