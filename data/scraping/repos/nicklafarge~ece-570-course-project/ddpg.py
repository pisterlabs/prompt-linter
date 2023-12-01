"""
DDPG Algorithm Implementation
Author: Nick LaFarge (reimplemented from OpenAI Spinning Up - DDPG)

See https://github.com/openai/spinningup/blob/master/spinup/algos/tf1/ddpg/ddpg.py for original DDPG implementation that
this version is based off.
"""

from agent_base import GymAgent
from DDPG.models import Actor, Critic
from DDPG.utils import get_vars, ReplayBuffer
import numpy as np
import tensorflow as tf


class DdpgAgent(GymAgent):
    def __init__(self, state_size, action_size,
                 gamma=0.99,
                 pi_lr=1e-4,
                 q_lr=1e-3,
                 polyak=0.995,
                 epochs=10,
                 replay_size=int(1e6),
                 batch_size=64,
                 n_initial_random_actions=10000,
                 n_initial_replay_steps=1000,
                 update_frequency=50,
                 actor_noise=0.1,
                 restore_from_file=None,
                 save_frequency=500,
                 actor_network_layer_sizes=(400, 300),
                 critic_network_layer_sizes=(400, 300),
                 actor_limit=1.0,
                 episode_number=None,
                 **kwargs
                 ):
        """
        DDPG implementation for ECE 570 Final Project (Fall 2020)

        :param state_size: Size of the state signal from the learning environment
        :param action_size: Dimension of the environment action
        :param gamma: discount factor
        :param pi_lr: learning rate for actor network
        :param q_lr: learning rate for critic network
        :param polyak: target network update hyperparameter
        :param epochs: Number of epochs to train each time training is conducted
        :param replay_size: Max size of replay buffer
        :param batch_size: Size of each minibatch for training
        :param n_initial_random_actions: Number o
        :param n_initial_replay_steps: Number of random steps before policy is used
        :param update_frequency: How often optimization step should run
        :param actor_noise: Gaussian noise std. dev. added to policy during training
        :param restore_from_file: True if the network should be loaded from a saved file
        :param save_frequency: How often to save the netowrk in number of episodes
        :param actor_network_layer_sizes: Tuple containing the sizes of actor network layers (size1, size2)
        :param critic_network_layer_sizes: Tuple containing the sizes of critic network layers (size1, size2)
        :param actor_limit: Action size limit (scaled from [-1,1] to [-actor_limit, actor_limit]. NOTE: this assumes
                            all actions are scaled the same (may not be true for all environments)
        :param episode_number: Which episode number to restore (if restore_from_file is set to True). Default is the
                               most recent save (highest episode number)
        :param kwargs: Passed to GymAgent initialization
        """
        super().__init__(state_size, action_size, **kwargs)
        self.gamma = gamma
        self.actor_lr = pi_lr
        self.critic_lr = q_lr
        self.polyak = polyak
        self.epochs = epochs
        self.replay_size = replay_size
        self.batch_size = batch_size
        self.n_initial_random_actions = n_initial_random_actions
        self.n_initial_replay_steps = n_initial_replay_steps
        self.update_frequency = update_frequency
        self.actor_noise = actor_noise
        self.restore_from_file = restore_from_file
        self.save_frequency = save_frequency
        self.actor_network_layer_sizes = actor_network_layer_sizes
        self.critic_network_layer_sizes = critic_network_layer_sizes
        self.actor_limit = actor_limit

        # Create specified save directory if it does not already exist
        if not self.save_filename.exists():
            self.save_filename.mkdir()

        # Initialize values
        self.critic_losses = []
        self.actor_losses = []
        self.total_step_counter = 0

        # Define scopes for keeping track of trainable parameters
        self.main_scope = 'main'
        self.target_scope = 'target'
        self.actor_scope = 'actor'
        self.critic_scope = 'critic'

        # This was necessary for training with tensorflow2
        tf.compat.v1.disable_eager_execution()

        # Set up the tensorflow placeholders
        self._setup_placeholders()

        # Create the actor and critic networks (+target networks and second critic/second critic target)
        akwargs = dict(learning_rate=self.actor_lr, layer_sizes=self.actor_network_layer_sizes)
        ckwargs = dict(learning_rate=self.critic_lr, layer_sizes=self.critic_network_layer_sizes)
        self.actor = Actor(self.x_ph, self.action_size, self.actor_limit, self.main_scope, self.actor_scope, **akwargs)
        self.actor_target = Actor(self.x2_ph, self.action_size, self.actor_limit, self.target_scope, self.actor_scope,
                                  **akwargs)
        self.critic = Critic(self.x_ph, self.main_scope, self.critic_scope, **ckwargs)
        self.critic_target = Critic(self.x2_ph, self.target_scope, self.critic_scope, **ckwargs)

        # Build main networks
        self._build_main()

        # Build target networks
        self._build_target()

        # Build optimizers for actor and critic networks
        self._build_optimizers()

        # Define how actions are sampled from the network
        self._define_actions()

        # Initialize the target networks using polyak averaging
        self._init_target()

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(self.state_size, self.action_size, self.replay_size)

        # To run on a separate computer, this was necessary to ensure tensorflow remained on its own core (a requirement
        # for the computer I trained on)
        session_conf = tf.compat.v1.ConfigProto(
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1)

        # Create tensorflow session
        self.sess = tf.compat.v1.InteractiveSession(config=session_conf)

        # Setup the saver and, if specified, restore the agent from a saved location
        successful = self._setup_saver(self.sess, restore_from_file, episode_number)

        # If a previous network was not loaded, run the target network initialization process
        if not successful:
            self.sess.run(self.target_init)

    def on_t_update(self, old_state, action, new_state, reward, done):
        """
        Called every time an agent-environment interaction occurs

        :param old_state: State prior to action
        :param action: Action chosen by the actor network
        :param new_state: New state produced by the environment as a result of the given action
        :param reward: Reward given at the new state
        :param done: True if the episode is complete
        """

        # Keep track of which step
        self.total_step_counter += 1

        # Store the agent-env interaction in the replay buffer
        self.replay_buffer.store(old_state, action, new_state, reward, done)

        # Only optimize after replay buffer is ready
        if self.total_step_counter < self.n_initial_replay_steps:
            return

        # Optimize every if self.update_frequency evenly divides self.total_step_counter
        if self.total_step_counter % self.update_frequency == 0:
            self.optimize()

    def on_episode_complete(self, episode_number):
        """
        Called every time an episode is complete

        :param episode_number: The number episode that just completed
        """

        # Store the current episode number (for neural net saving)
        self.current_episode_number = episode_number

        # Save the network if self.save_frequency evenly divides self.current_episode_number
        if self.current_episode_number > 0 and self.current_episode_number % self.save_frequency == 0:
            self.save(self.sess, self.current_episode_number)

    def optimize(self):
        """
        Run the optimization processes defined in the models
        """

        # keep track of the loss functions across optimization
        q_losses_batch = []
        pi_losses_batch = []

        # Optimize over the specified number of epochs
        for _ in range(self.epochs):

            # Sample a batch from the replay buffer
            batch = self.replay_buffer.sample_batch(self.batch_size)
            a_grads = self.sess.run(self.q_pi_action_grads, feed_dict={self.x_ph: batch['states']})[0]

            # Add the batch into the placeholders
            feed_dict = {self.x_ph: batch['states'],
                         self.x2_ph: batch['new_states'],
                         self.a_ph: batch['actions'],
                         self.r_ph: batch['rewards'],
                         self.d_ph: batch['dones'],
                         self.action_grads_ph: a_grads
                         }

            # Run optimizer on batch
            self.sess.run([self.update_actor, self.update_critic], feed_dict)

            # Compute losses
            q_loss, pi_loss = self.sess.run([self.critic_loss, self.actor_loss], feed_dict)
            q_losses_batch.append(q_loss)
            pi_losses_batch.append(pi_loss)

            # Update target network
            self.sess.run(self.target_update, feed_dict)

        # Average the losses and add them to the saved lists (to observe trends later)
        self.critic_losses.append(np.average(q_losses_batch))
        self.actor_losses.append(np.average(pi_losses_batch))

    def act(self, state, deterministic=False):
        """
        Defines how the agent selects an action based on an observed state form the environment. During training, this
        is a stochastic process, but if the deterministic flag is given, the random variables are dropped and the
        deterministic actor network is directly used.

        :param state: Current environmental state (or observation)
        :param deterministic: True if no randomness should be included in the action
        :return: Action sampled from actor network
        """

        # Sample the action from a uniform random distribution for the first few episodes -- allows greater initial
        # exploration of the action space (we are allowed to do this because DDPG is an off-policy algorithm)
        if not deterministic and len(self.replay_buffer) < self.n_initial_random_actions:
            action = np.squeeze(np.random.uniform(-self.actor_limit, self.actor_limit, (self.action_size, 1)))

            # Dimensionality fix for tasks that have a single action variable
            if self.action_size == 1:
                action = np.array([float(action)])
            return action

        # Sample action from the neural network
        feed_dict = {self.x_ph: np.reshape(state, (1, self.state_size))}
        action_fn = self.deterministic_action if deterministic else self.sample_action
        action = self.sess.run(action_fn, feed_dict=feed_dict)[0]

        return action

    def _init_target(self):
        """
        Define update and init functions for the target networks. Updates are done by polyak averaging values
        from the main networks, and initialization is done by directly copying values.
        """
        self.target_update = tf.group([tf.compat.v1.assign(v_targ, self.polyak * v_targ + (1 - self.polyak) * v_main)
                                       for v_main, v_targ in
                                       zip(get_vars(self.main_scope), get_vars(self.target_scope))])
        self.target_init = tf.group([tf.compat.v1.assign(v_targ, v_main)
                                     for v_main, v_targ in
                                     zip(get_vars(self.main_scope), get_vars(self.target_scope))])

    def _setup_placeholders(self):
        """
        Define tensorflow placeholders for the state, new state, action, action gradient, reward, and done signals
        """
        self.x_ph = tf.compat.v1.placeholder(tf.float32, (None, self.state_size), 'state')
        self.x2_ph = tf.compat.v1.placeholder(tf.float32, (None, self.state_size), 'new_state')
        self.a_ph = tf.compat.v1.placeholder(tf.float32, (None, self.action_size), 'action')
        self.action_grads_ph = tf.compat.v1.placeholder(tf.float32, (None, self.action_size), 'action_grads')
        self.r_ph = tf.compat.v1.placeholder(tf.float32, (None,), 'reward')
        self.d_ph = tf.compat.v1.placeholder(tf.float32, (None,), 'done')

    def _build_main(self):
        """
        Build the main networks. These include the actor, two critics that depend on a specified action, and a
        critic that depends on the actor network
        """
        self.pi = self.actor.build_network()
        self.q, self.q_action_grads = self.critic.build_network(self.a_ph)
        self.q_pi, self.q_pi_action_grads = self.critic.build_network(self.pi, reuse=True)

    def _build_target(self):
        """
        Build the target networks. These include the actor and the two critics.
        """
        self.pi_target = self.actor_target.build_network()
        self.q_pi_target, self.q_pi_target_action_grads = self.critic_target.build_network(self.pi_target)

    def _define_actions(self):
        """
        Define the stochastic and deterministic actions
        """

        with tf.compat.v1.variable_scope('sample_action'):
            epsilon = self.actor_noise * np.random.randn(self.action_size)
            self.sample_action = self.pi + epsilon

        with tf.compat.v1.variable_scope('deterministic_action'):
            self.deterministic_action = self.pi

    def _build_optimizers(self):
        """
        Build the optimizers that will improve the actor and critic networks. The optimizers are defined the model
        classes, and are built here.
        """

        self.actor_loss, self.update_actor = self.actor.define_optimizer(self.pi, self.action_grads_ph, self.q_pi)
        self.critic_loss, self.update_critic = self.critic.define_optimizer(self.r_ph, self.gamma, self.d_ph, self.q,
                                                                            self.q_pi_target)