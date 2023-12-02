""" 
Implementation of DDPG - Deep Deterministic Policy Gradient

Algorithm and hyperparameter details can be found here: 
    http://arxiv.org/pdf/1509.02971v2.pdf

The algorithm is tested on the Pendulum-v0 OpenAI gym task and developed with tflearn + Tensorflow

Will change tflearn to vanilla TensorFlow once the system is working adequately.  Original Tensorflow is much more
intuitive than the high level tflearn package.
"""
import tensorflow as tf
import numpy as np
import gym
import tflearn
import argparse
import pprint as pp
import os

from copy import deepcopy
from CSTR_model import MimoCstr
from gym import wrappers
from replay_buffer import ReplayBuffer

# Suppress useless TensorFlow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
The Actor and Critic deep neural network architectures.
"""


class ActorNetwork(object):
    """
    Input to the network is the state, output is the action under a deterministic policy.

    The output layer activation is a tanh to keep the action bounded between -action_bound and action_bound
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        # Mixing factor of updating the target network weights with the online network weights
        # Done very slowly to avoid the moving target problem
        self.tau = tau
        # Amount sampled from the replay memory
        self.batch_size = batch_size

        """
        Actor network
            inputs: The states of the system
            out: The tanh output, bounded between [-1, 1]
            scaled_out: The tanh output scaled by the action bound, i.e., the physical action out
            
            network_params: Creates a call function that can call the actor network weights only.
        """
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        """
        Target actor network
            target_inputs: The states of the system, identical to above
            out: The tanh output of the target network, bounded between [-1, 1]
            scaled_out: The tanh output scaled by the action bound.  The physical action out.
            
            target_network_params: Creates a call function that can call the target actor network weights only.
        """
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
            len(self.network_params):]

        """
        Operations for periodically updating target network with online network weights
        Target Network Θ' = Θ * τ + Θ' * (1 - τ)
        """

        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        """
        Combines gradients here.
        tf.gradients function creates symbolic derivatives with respect to x.  
        tf.gradients(variable, w.r.t, initial guess)
        
        Then the gradients are averaged, i.e., divided by the batch size.
        """

        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        """
        Optimization algorithm.
        Using Adaptive Momentum Gradient Descent to update the Θ parameters from the actor_gradients.
        """
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        # Calculates for the total amount of variables in the actor network.  This will be used in the critic network so
        # we can create the direct call to the critic network and target critic network.
        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        """
        The actor network architecture:
            Input Data: [None, Num of States]

            400 Neuron Fully Connected
            Batch normalization. https://arxiv.org/pdf/1502.03167v3.pdf.  Can serve as dropout
                Put after ReLU activation so half the neurons are not killed off due to normalization to 0.
            ReLU activation function

            300 Neuron Fully Connected
            Batch normalization
            ReLU activation function

            Weight initialization as uniform [-0.003, 0.003]
            Num of Action Neurons
            TanH activation function

            Scaled_out = TanH out * action bound
        """
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 50)
        # net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        net = tflearn.fully_connected(net, 40)
        # net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        # Final layer weights are init to Uniform[-3e-3, 3e-3].  Try to ensure policies are near 0 at the start.
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(
            net, self.a_dim, activation='tanh', weights_init=w_init)

        # Scale output to -action_bound to action_bound
        # scaled_out = out
        scaled_out = tf.multiply(out, self.action_bound)

        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):

    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        """
        Critic network
            inputs: The states of the system
            action: The action from the actor
            out: The expected Q(s, a) value

            network_params: Creates a call function that can call the critic network weights only.
        """

        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        """
        Target critic network
            inputs: The states of the system
            action: The action from the actor
            out: The expected Q(s, a) value

            network_params: Creates a call function that can call the target critic network weights only.
        """

        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        """
        Operations for periodically updating target network with online network weights
        Target Network Θ' = Θ * τ + Θ' * (1 - τ)
        """

        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau)
             + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        """
        Using mean squared error loss function.  L = (Q(s, a | Θ') - Q(s, a | Θ)^2
        
        Updates Θ weights using adaptive momentum gradient descent. Fed value is predicted_q, optimizer optimizes critic
        network via self.out.
        """

        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the mini-batch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        """
        Input Data: [None, Num of States]
        Action Data: [None, NUm of Actions]

        400 Fully Connected Neurons
        ReLU activation function
        Batch Normalization

        t1 from original input data, 300 fully connected neurons
        t2 added onto the existing computational graph, 300 fully connected neurons

        Activation is the t1 and t2 merged together, then passed through a ReLU functon

        Output weights initiated between -0.003 to 0.003
        1 Neuron fully connected layer with no activation
        """

        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inputs, 40)
        # net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 30)

        t2 = tflearn.fully_connected(action, 30)

        net = tflearn.activation(
            tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t1.b + t2.b, activation='relu', name='merge')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-4, 3e-4].  Ensure output Q are close to 0 initially.
        w_init = tflearn.initializations.uniform(minval=-0.0003, maxval=0.0003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab


class OrnsteinUhlenbeckActionNoise:

    """
    Introduces time correlated noise into the action term taken by the deterministic policy.
    he Ornstein-Uhlenbeck process satisfies the following stochastic differential equation:

        dxt = theta*(mu - xt)*dt + sigma*dWt

    where dWt can be simplified into dt*N(0, 1), i.e., white noise.

    Mu: Mean to be arrived at in the end of the process
    Sigma:  Amount of noise injected into the system.  Volatility of average magnitude
    Theta: How much weight towards going towards the mean, mu.  Rate of mean reversion.
    dt: Sampling time
    """

    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __call__(self):

        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x

        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


"""
Tensorflow summary operations used for Tensorboard

Two operations built are episode reward and average max Q value.
"""


def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars


"""
Agent training procedure.
"""


def train(sess, env, args, actor, critic, actor_noise):

    # Set up summary operations
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))

    # Needed to enable BatchNorm. 
    # This hurts the performance on Pendulum but could be useful
    # in other environments.
    # tflearn.is_training(True)
    a_list = []

    for i in range(int(args['max_episodes'])):

        # Reset the environment, initial action 0, and initialize the action list for observability during analysis
        s = env.reset()
        actor_noise.reset()
        a = 0

        # Evaluation Period
        eval_time = 999

        # Episode reward and episode average max Q initializations
        ep_reward = 0
        ep_ave_max_q = 0

        # Initialize zero mean and st_dev.  Will be corrected before
        mean = env.xs
        st_dev = [0.8, 15, 0.7]

        if i % 50 == 0 and i != 0:
            print("Evaluation Episode")

        # Loop for max_episode_len
        for j in range(1, int(args['max_episode_len']) + 1):

            # Take action every "sampling time" time steps to ensure steady state is reached
            if j % int(args['sampling_time']) == 0:

                # Correct for the initial state bug
                if j == int(args['sampling_time']):
                    s = deepcopy(env.x[j - 1, :])

                # Normalize the states by subtracting the mean and dividing by the variance
                s -= mean
                s /= st_dev

                # Every 50th episode, the action will have no noise to evaluate performance.
                if i % 50 == 0 and i != 0:
                    a = actor.predict(np.reshape(s, (1, actor.s_dim)))

                    if i == (args['max_episodes'] - 1):
                        a_list.append(a)

                # Add Ornstein-Ulhenbeck exploration noise to the action
                else:
                    noise = actor_noise()
                    a = actor.predict(np.reshape(s, (1, actor.s_dim))) + noise

                    if i == (args['max_episode_len'] - 1):
                        a_list.append(a - noise)

                # Take the action
                env.u[j, 0] = env.u[j - 1, 0] + a[0]

                # Define evaluation time for feedback
                eval_time = j + int(args['sampling_time']) - 1

            else:
                # If it is not the sampling time, keep input constant
                env.u[j, 0] = env.u[j - 1, 0]

            # Simulate the next step
            env.x[j, :] = env.cstr_sim.sim(env.x[j - 1, :], env.u[j, :])

            # Determines if its the end of the current episode.  If the input is very far from ideal, episode ends.
            if j == env.Nsim or env.u[j, 0] < 150 or env.u[j, 0] > 450:
                terminal = True
            else:
                terminal = False

            # Feedback for RL
            if j == eval_time:

                # Ensure feedback is evaluated correctly
                assert((j + 1) % int(args['sampling_time']) == 0)

                # Reward for RL
                r = env.reward_function(j, a[0][0])

                # Next state for RL
                s2 = deepcopy(env.x[j, :])

                # Add the latest states, action, reward, terminal, and new state to the replay memory
                replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r,
                                  terminal, np.reshape(s2, (actor.s_dim,)))

                # Update the new state to be the current state
                s = s2

                # Add the step's reward towards the whole episodes' reward
                ep_reward += r

            # Keep adding experience to the memory until there are at least mini-batch size samples
            # Batch Training area
            if replay_buffer.size() > int(args['minibatch_size'] * 5):

                # mini-batch size
                mini_batch_size = np.power(i, 1/3) * int(args['minibatch_size'])

                # Obtain a batch of data from replay buffer
                s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(int(mini_batch_size))

                # Calculate critic target Q-value, feeding in the actor target action
                # States is the s2 from the replay buffer
                target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))

                # Calculate the Q values
                y_i = []
                for k in range(int(mini_batch_size)):
                    # Terminal state, Q = r because there is no additional trajectory beyond this point
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    # If state is not terminal, Q = r + gamma * argmax-a * Q(s', a)
                    else:
                        y_i.append(r_batch[k] + critic.gamma * target_q[k])

                """
                Update the critic given the targets
                Exact algorithm:  critic.train() returns predicted_q_value, optimize.
                Optimize takes MSE of y_i and predicted q value out.  Then does Adam Gradient Descent updating the
                critic network.
                """

                predicted_q_value, _ = critic.train(s_batch, a_batch, np.reshape(y_i, (int(mini_batch_size), 1)))

                # Output is 64 dimen predicted_q_value, then find the max of them.
                ep_ave_max_q += np.amax(predicted_q_value)

                """
                Update the actor policy using the sampled gradient
                """

                # Scaled output action given the s_batch states.
                a_outs = actor.predict(s_batch)

                # Inputs the states, and the actions given those states.
                # Forms symbolic function of the gradients as a function of the action
                grads = critic.action_gradients(s_batch, a_outs)

                # Updates actors given the gradients
                actor.train(s_batch, grads[0])

                # Update target networks by tau
                actor.update_target_network()
                critic.update_target_network()

            if terminal:
                # Update the summary ops
                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / float(j)
                })

                writer.add_summary(summary_str, i)
                writer.flush()

                print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(int(ep_reward),
                                                                             i, (ep_ave_max_q / float(j))))
                break

    return replay_buffer, a_list


def main(args):

    """
    Environment used in this code is Pendulum-v0 from OpenAI gym.

        States: cos(theta), sin(theta), theta_dt
        Actions: Force application between -2 to 2
        Reward: -(Θ^2 + 0.1*Θ_dt^2 + 0.001*action^2)

    Objective:  Pendulum is vertical, with 0 movement.
    Initialization: Starts at a random angle, and at a random velocity.
    End: After all the steps are exhausted
    """

    # Initialize saver
    saver = tf.train.Saver()

    with tf.Session() as sess:

        # Create the gym environment
        env = MimoCstr(nsim=args['max_episode_len'])
        # env = gym.make(args['env'])

        # Set all the random seeds for the random packages
        np.random.seed(int(args['random_seed']))
        tf.set_random_seed(int(args['random_seed']))
        env.seed(int(args['random_seed']))
        tflearn.init_graph(seed=1)

        # Define all the state and action dimensions, and the bound of the action
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high

        # Ensure action bound is symmetric
        assert (env.action_space.high == -env.action_space.low)

        # Restore old model
        # saver.restore(sess, args['ckpt_dir'])

        # Initialize the actor and critic
        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             float(args['actor_lr']), float(args['tau']),
                             int(args['minibatch_size']))

        critic = CriticNetwork(sess, state_dim, action_dim,
                               float(args['critic_lr']), float(args['tau']),
                               float(args['gamma']),
                               actor.get_num_trainable_vars())

        # Initialize Ornstein Uhlenbeck Noise
        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

        # Train the Actor-Critic Model
        replay_buffer, action_list = train(sess, env, args, actor, critic, actor_noise)

        # Save the model
        saver.save(sess, args['ckpt_dir'])

        return actor, critic, env, replay_buffer, action_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--tau', help='soft target update parameter', default=0.001)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=100000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=64)

    # run parameters
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='Pendulum-v0')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=301)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=315)
    parser.add_argument('--sampling-time', help='Sampling time by RL controller', default=1)
    parser.add_argument('--render-env', help='render the gym env', action='store_true')
    parser.add_argument('--use-gym-monitor', help='record gym results', action='store_true')
    parser.add_argument('--monitor-dir', help='directory for storing gym results', default='./results/gym_ddpg')
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='./results/tf_ddpg')
    parser.add_argument('--ckpt-dir', help='directory for saved models', default='./results/checkpoints/a.ckpt')

    parser.set_defaults(render_env=False)
    parser.set_defaults(use_gym_monitor=True)

    # Parse the args
    Args = vars(parser.parse_args())

    # Pretty print the args
    pp.pprint(Args)

    # Run the function
    Actor, Critic, Model, Replay_Buffer, Action_List = main(Args)
