"""Policy gradient agents that support changing state shapes.

Currently includes policy gradient agent (i.e., Monte Carlo policy
gradient or vanilla policy gradient) and proximal policy optimization
agent.
"""

import numpy as np
import multiprocessing as mp
import tensorflow as tf


PACKET_SIZE = 10 # this must divide the number of episodes, and ideally should divide (episodes)/(number of cores)


def discount_rewards(rewards, gam):
    """Discount the list or array of rewards by gamma in-place.

    Parameters
    ----------
    rewards : list or ndarray of ints or floats
        A list of rewards from a single complete trajectory.
    gam : float
        The discount rate.

    Returns
    -------
    rewards : list or ndarray
        The input array with each reward replaced by discounted reward-to-go.

    Examples
    --------
    >>> rewards = [1, 2, 3, 4, 5]
    >>> discount_rewards(rewards, 0.5)
    [1, 2, 6.25, 6.5, 5]

    Note that the input rewards list is modified in place. The return
    value is just a reference to the original list.

    >>> rewards = [1, 2, 3, 4, 5]
    >>> discounted_rewards = discount_rewards(rewards, 0.5)
    >>> rewards
    [1, 2, 6.25, 6.5, 5]

    """
    cumulative_reward = 0
    for i in reversed(range(len(rewards))):
        cumulative_reward = rewards[i] + gam * cumulative_reward
        rewards[i] = cumulative_reward
    return rewards


def np_one_hot(length, index):
    a = np.zeros(length)
    a[index] = 1
    return a.astype('int')


class TrajectoryBuffer:
    """A buffer to store and compute with trajectories.

    The buffer is used to store information from each step of interaction
    between the agent and environment. When a trajectory is finished it
    computes the discounted rewards and generalized advantage estimates. After
    some number of trajectories are finished it can return batches grouped by
    state shape with normalized advantage estimates.

    Parameters
    ----------
    gam : float, optional
        The discount rate.
    lam : float, optional
        The parameter for generalized advantage estimation.

    See Also
    --------
    discount_rewards : Discount the list or array of rewards by gamma in-place.

    Notes
    -----
    The implementation is based on the implementation of buffers used in the
    policy gradient algorithms from OpenAI Spinning Up. Formulas for
    generalized advantage estimation are from [1]_. The major implementation
    difference is that we allow for different sized states and action
    dimensions, and only assume that each state shape corresponds to some fixed
    action dimension.

    References
    ----------
    .. [1] Schulman et al, "High-Dimensional Continuous Control Using
       Generalized Advantage Estimation," ICLR 2016.

    Examples
    --------
    >>> buffer = TrajectoryBuffer()
    >>> tau = [(np.array([3]), 0.4, 3, 1, 1),
    ...        (np.array([1, 3, 7]), 0.1, 2, 0, 0),
    ...        (np.array([1, 4, 2]), 0.7, 1, 2, 2),
    ...        (np.array([2, 5]), 0.6, 2, 1, 1),
    ...        (np.array([1, 7]), 0.3, 0, 0, 1)]
    >>> for t in tau:
    ...     buffer.store(*t)
    >>> buffer.finish()
    >>> buffer.get()

    """

    def __init__(self, gam=0.99, lam=0.97):
        self.gam = gam
        self.lam = lam
        self.states = []
        self.probas = []
        self.values = []
        self.pred_values = []
        self.actions = []
        self.rewards = []
        self.start = 0
        self.end = 0

    def store(self, state, proba, value, action, reward):
        """Store the information from one interaction with the environment.

        Parameters
        ----------
        state : ndarray
           The observation of the state.
        proba : float
           The agent's previous probability of picking the chosen action.
        value : float
           The agent's computed value of the state.
        action : int
           The chosen action in this trajectory.
        reward : float
           The reward received in the next transition.

        """
        self.states.append(state)
        self.probas.append(proba)
        self.values.append(value)
        self.pred_values.append(value)
        self.actions.append(action)
        self.rewards.append(reward)
        self.end += 1

    def finish(self):
        """Finish an episode and compute advantages and discounted rewards in-place.

        Advantages are stored in place of `values` and discounted rewards are
        stored in place of `rewards` for the current trajectory.
        """
        tau = slice(self.start, self.end)
        rewards = np.array(self.rewards[tau], dtype=np.float)
        values = np.array(self.values[tau], dtype=np.float)
        delta = rewards - values
        delta[:-1] += self.gam * values[1:]
        self.rewards[tau] = list(discount_rewards(rewards, self.gam))
        self.values[tau] = list(discount_rewards(delta, self.gam * self.lam))
        self.start = self.end

    def clear(self):
        """Reset the buffer."""
        self.states.clear()
        self.probas.clear()
        self.values.clear()
        self.pred_values.clear()
        self.actions.clear()
        self.rewards.clear()
        self.start = 0
        self.end = 0

    def get(self, normalize_advantages=True):
        """Return a dictionary of state shapes to training data.

        Parameters
        ----------
        normalize_advantages : bool, optional
            Whether to normalize the returned advantages.

        Returns
        -------
        data : dict
            A dictionary mapping state shape to training data.

            Each value of the dictionary is a dictionary with keys
            'states', 'probas', 'values', 'actions', 'advants', and values
            ndarrays.

        """
        advantages = np.array(self.values[:self.start])
        if normalize_advantages:
            advantages -= np.mean(advantages)
            advantages /= np.std(advantages)
        shapes = {}
        for i, state in enumerate(self.states[:self.start]):
            shapes.setdefault(state.shape, []).append(i)
        data = {}
        for shape, indices in shapes.items():
            data[shape] = {
                'states': np.array([self.states[i] for i in indices],
                                   dtype=np.float32),
                'probas': np.array([self.probas[i] for i in indices],
                                   dtype=np.float32),
                'values': np.array([[self.rewards[i]] for i in indices],
                                   dtype=np.float32),
                'actions': np.array([self.actions[i] for i in indices],
                                    dtype=np.int),
                'advants': np.array([advantages[i] for i in indices],
                                    dtype=np.float32),
                'pred_values': np.array([[self.pred_values[i]] for i in indices],
                                        dtype=np.float32),
            }
        return data

    def __len__(self):
        return len(self.states)


def _merge_buffers(bufferlist):
    output = bufferlist[0]
    assert output.start==output.end, "Must apply self.finish() before merging buffers"
    for b in bufferlist[1:]:
        assert b.start==b.end, "Must apply self.finish() before merging buffers"
        output.states += b.states
        output.probas += b.probas
        output.values += b.values
        output.pred_values += b.pred_values
        output.actions += b.actions
        output.rewards += b.rewards
    output.end = len(output.states)
    output.start = output.end
    return output


def print_status_bar(i, epochs, history, verbose=1):
    """Print a formatted status line."""
    metrics = "".join([" - {}: {:.4f}".format(m, history[m][i])
                       for m in ['mean_returns']])
    end = "\n" if verbose == 2 or i+1 == epochs else ""
    print("\rEpoch {}/{}".format(i+1, epochs) + metrics, end=end)


class Agent:
    """Abstract base class for policy gradient agents.
    
    All functionality for policy gradient is implemented in this
    class. Derived classes must define the property `policy_loss`
    which is used to train the policy.

    Parameters
    ----------
    policy_network : network
        The network for the policy model.
    policy_lr : float, optional
        The learning rate for the policy model.
    policy_updates : int, optional
        The number of policy updates per epoch of training.
    value_network : network, optional
        The network for the value model.
    value_lr : float, optional
        The learning rate for the value model.
    value_updates : int, optional
        The number of value updates per epoch of training.
    gam : float, optional
        The discount rate.
    lam : float, optional
        The parameter for generalized advantage estimation.
    normalize : bool, optional
        Whether to normalize advantages.
    action_dim_fn : function, optional
        The function that maps state shape to action dimension.
    kld_limit : float, optional
        The limit on KL divergence for early stopping policy updates.
    """

    def __init__(self,
                 policy_network, policy_lr=1e-4, policy_updates=1,
                 value_network=None, value_lr=1e-3, value_updates=25,
                 gam=0.99, lam=0.97, normalize_advantages=True, eps=0.2,
                 action_dim_fn=lambda s: s[0], kld_limit=0.01):
        self.policy_model = policy_network
        self.policy_loss = NotImplementedError
        self.policy_optimizer = tf.keras.optimizers.Adam(lr=policy_lr)
        self.policy_updates = policy_updates

        self.value_model = value_network
        self.value_loss = tf.keras.losses.mse
        self.value_optimizer = tf.keras.optimizers.Adam(lr=value_lr)
        self.value_updates = value_updates

        self.lam = lam
        self.gam = gam
        self.buffer = TrajectoryBuffer(gam=gam, lam=lam)
        self.normalize_advantages = normalize_advantages
        self.action_dim_fn = action_dim_fn
        self.kld_limit = kld_limit

    def act(self, state, greedy=False, return_probs=False):
        """Return an action for the given state using the policy model.

        Parameters
        ----------
        state : np.array
            The state of the environment.
        greedy : bool, optional
            Whether to sample or pick the action with max probability.
        return_probs : bool, optional
            Whether to return the probability vector.
        """
        probs = self.policy_model.predict(state[np.newaxis])[0]
        action = np.argmax(probs) if greedy else np.random.choice(len(probs), p=probs)
        return (action, probs[action]) if return_probs else action

    def train(self, env, episodes=10, epochs=1, max_episode_length=None, verbose=0, save_freq=1,
              logdir=None, test_env=None, parallel=True):
        """Train the agent on env.

        Parameters
        ----------
        env : environment
            The environment to train on.
        episodes : int, optional
            The number of episodes to perform per epoch of training.
        epochs : int, optional
            The number of epochs to train.
        max_episode_length : int, optional
            The maximum number of steps of interaction in an episode.
        verbose : int, optional
            How much information to print to the user.
        save_freq : int, optional
            How often to save the model weights, measured in epochs.
        logdir : str, optional
            The directory to store Tensorboard logs and model weights.
        test_env : environment, optional
            The environment to report performance on.

        Returns
        -------
        history : dict
            Dictionary with statistics from training.

        """
        tb_writer = None if logdir is None else tf.summary.create_file_writer(logdir)
        history = {'mean_returns': np.zeros(epochs),
                   'min_returns': np.zeros(epochs),
                   'max_returns': np.zeros(epochs),
                   'std_returns': np.zeros(epochs),
                   'mean_ep_lens': np.zeros(epochs),
                   'min_ep_lens': np.zeros(epochs),
                   'max_ep_lens': np.zeros(epochs),
                   'std_ep_lens': np.zeros(epochs),
                   'policy_updates': np.zeros(epochs),
                   'delta_policy_loss': np.zeros(epochs),
                   'policy_ent': np.zeros(epochs),
                   'policy_kld': np.zeros(epochs),
                   'mean_value': np.zeros(epochs),
                   'mean_value_error': np.zeros(epochs)}

        for i in range(epochs):
            self.buffer.clear()
            return_history = self.run_episodes(
                env, episodes=episodes, max_episode_length=max_episode_length,
                store=True, parallel=parallel
            )
            batches = self.buffer.get(normalize_advantages=self.normalize_advantages)
            policy_history = self._fit_policy_model(batches, epochs=self.policy_updates)
            value_history = self._fit_value_model(batches, epochs=self.value_updates)

            if test_env is not None:
                return_history = self.run_episodes(
                    test_env, episodes=episodes, max_episode_length=max_episode_length, store=False, 
                    parallel=parallel
                )
                try:
                    env.env.ideal_gen.update()  # for binned training using FromDirectoryIdealGenerator
                except AttributeError:
                    pass

            history['mean_returns'][i] = np.mean(return_history['returns'])
            history['min_returns'][i] = np.min(return_history['returns'])
            history['max_returns'][i] = np.max(return_history['returns'])
            history['std_returns'][i] = np.std(return_history['returns'])
            history['mean_ep_lens'][i] = np.mean(return_history['lengths'])
            history['min_ep_lens'][i] = np.min(return_history['lengths'])
            history['max_ep_lens'][i] = np.max(return_history['lengths'])
            history['std_ep_lens'][i] = np.std(return_history['lengths'])
            history['policy_updates'][i] = len(policy_history['loss'])
            history['delta_policy_loss'][i] = policy_history['loss'][-1] - policy_history['loss'][0]
            history['policy_ent'][i] = policy_history['ent'][-1]
            history['policy_kld'][i] = policy_history['kld'][-1]
            history['mean_value'][i] = np.mean(np.vstack([data['pred_values']
                                                          for shape, data in batches.items()]))
            history['mean_value_error'][i] = np.mean(np.square(np.vstack([data['pred_values'] - data['values']
                                                                         for shape, data in batches.items()])))

            if logdir is not None and (i+1) % save_freq == 0:
                self.save_policy_weights(logdir + "/policy-" + str(i+1) + ".h5")
                self.save_value_weights(logdir + "/value-" + str(i+1) + ".h5")
            if tb_writer is not None:
                with tb_writer.as_default():
                    tf.summary.scalar('mean_returns', history['mean_returns'][i], step=i)
                    tf.summary.scalar('min_returns', history['min_returns'][i], step=i)
                    tf.summary.scalar('max_returns', history['max_returns'][i], step=i)
                    tf.summary.scalar('std_returns', history['std_returns'][i], step=i)
                    tf.summary.scalar('mean_ep_lens', history['mean_ep_lens'][i], step=i)
                    tf.summary.scalar('min_ep_lens', history['min_ep_lens'][i], step=i)
                    tf.summary.scalar('max_ep_lens', history['max_ep_lens'][i], step=i)
                    tf.summary.scalar('std_ep_lens', history['std_ep_lens'][i], step=i)
                    tf.summary.histogram('returns', return_history['returns'], step=i)
                    tf.summary.histogram('lengths', return_history['lengths'], step=i)
                    tf.summary.scalar('policy_updates', history['policy_updates'][i], step=i)
                    tf.summary.scalar('delta_policy_loss', history['delta_policy_loss'][i], step=i)
                    tf.summary.scalar('policy_ent', history['policy_ent'][i], step=i)
                    tf.summary.scalar('policy_kld', history['policy_kld'][i], step=i)
                    tf.summary.scalar('mean_value', history['mean_value'][i], step=i)
                    tf.summary.scalar('mean_value_error', history['mean_value_error'][i], step=i)
                tb_writer.flush()
            if verbose > 0:
                print_status_bar(i, epochs, history, verbose=verbose)

        return history


    def run_episode(self, env, max_episode_length=None, greedy=False, buffer=None):
        """Run an episode and return total reward and episode length.

        Parameters
        ----------
        env : environment
            The environment to interact with.
        max_episode_length : int, optional
            The maximum number of interactions before the episode ends.
        greedy : bool, optional
            Whether to choose the maximum probability or sample.
        buffer : TrajectoryBuffer object, optional
            If included, it will store the whole rollout in the given buffer.

        Returns
        -------
        (total_reward, episode_length) : (float, int)
            The total nondiscounted reward obtained in this episode and the
            episode length.

        """
        state = env.reset()
        done = False
        episode_length = 0
        total_reward = 0
        while not done:
            action, prob = self.act(state, return_probs=True)
            if self.value_model is None:
                value = 0
            elif hasattr(self.value_model, 'agent'):  # this is an AgentBaseline
                if hasattr(self.value_model.agent, 'strategy'):  # with a BuchbergerAgent
                    value = self.value_model.predict(env.env)
                else:  # with a PGAgent/PPOAgent
                    value = self.value_model.predict(env)
            else:
                value = self.value_model.predict(state[np.newaxis])[0][0]
            next_state, reward, done, _ = env.step(action)
            if buffer is not None:
                buffer.store(state, prob, value, action, reward)
            episode_length += 1
            total_reward += reward
            if max_episode_length is not None and episode_length > max_episode_length:
                break
            state = next_state
        if buffer is not None:
            buffer.finish()
        return total_reward, episode_length

    def _parallel_run_episode(self, env, max_episode_length, greedy, random_seed, output, packet_size):
        np.random.seed(random_seed)
        buff = TrajectoryBuffer(gam=self.gam, lam=self.lam)
        results =[]
        for i in range(packet_size):
            results.append(self.run_episode(env, max_episode_length=max_episode_length, greedy=greedy, buffer=buff))
        output.put((results,buff))

    def run_episodes(self, env, episodes=100, max_episode_length=None, greedy=False, store=False,
                     parallel=True):
        """Run several episodes, store interaction in buffer, and return history.

        Parameters
        ----------
        env : environment
            The environment to interact with.
        episodes : int, optional
            The number of episodes to perform.
        max_episode_length : int, optional
            The maximum number of steps before the episode is terminated.
        greedy: bool, optional
            Whether to choose the maximum probability or sample.
        store: bool, optional
            Whether or not to store the rollout in self.buffer

        Returns
        -------
        history : dict
            Dictionary which contains information from the runs.

        """
        history = {'returns': np.zeros(episodes),
                   'lengths': np.zeros(episodes)}
        if parallel:
            output = mp.Queue()
            assert episodes % PACKET_SIZE == 0, "PACKET_SIZE must divide the number of episodes"
            num_processes = int(episodes / PACKET_SIZE)
            processes = [mp.Process(target = self._parallel_run_episode,args=(env, max_episode_length, greedy, seed, output, PACKET_SIZE)) for seed in np.random.randint(0,4294967295,num_processes)]
            for p in processes:
                p.start()
            results = [output.get() for p in processes]
            for p in processes:
                p.join()
            self.buffer=_merge_buffers([b for (_, b) in results])
            returns = [x for (t,_) in results for x in t]
            for i in range(episodes):
                (history['returns'][i], history['lengths'][i]) = returns[i]
        else:
            for i in range(episodes):
                R, L = self.run_episode(env,max_episode_length=max_episode_length, greedy=greedy, buffer=self.buffer)
                history['returns'][i] = R
                history['lengths'][i] = L

        return history

    def _fit_policy_model(self, batches, epochs=1):
        """Fit policy model with one gradient update per epoch."""
        history = {'loss': [], 'kld': [], 'ent': []}
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                losses, klds, ents = [], [], []
                for shape, data in batches.items():
                    action_dim = self.action_dim_fn(shape)
                    if action_dim == 1:
                        continue
                    probs = self.policy_model(data['states'])
                    new_prob = tf.reduce_sum(tf.one_hot(data['actions'], action_dim) * probs, axis=1)
                    losses.append(self.policy_loss(new_prob, data['probas'], data['advants']))
                    klds.append(tf.math.log(data['probas'] / new_prob))
                    ents.append(-tf.math.log(new_prob))
                loss = tf.reduce_mean(tf.concat(losses, axis=0))
                kld = tf.reduce_mean(tf.concat(klds, axis=0))
                ent = tf.reduce_mean(tf.concat(ents, axis=0))
            varis = self.policy_model.trainable_variables
            grads = tape.gradient(loss, varis)
            self.policy_optimizer.apply_gradients(zip(grads, varis))
            history['loss'].append(loss)
            history['kld'].append(kld)
            history['ent'].append(ent)
            if self.kld_limit is not None and kld > self.kld_limit:
                break
            self.policy_model.get_weights()  # for fast wrappers
        history = {k: np.array(v) for k, v in history.items()}
        return history

    def load_policy_weights(self, filename):
        """Load weights from filename into the policy model."""
        self.policy_model.load_weights(filename)

    def save_policy_weights(self, filename):
        """Save the current weights in the policy model to filename."""
        self.policy_model.save_weights(filename)

    def _fit_value_model(self, batches, epochs=1):
        """Fit value model with one gradient update per epoch."""
        if self.value_model is None or hasattr(self.value_model, 'agent'):
            epochs = 0
        history = {'loss': np.zeros(epochs)}
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                losses = []
                for shape, data in batches.items():
                    values = self.value_model(data['states'])
                    losses.append(self.value_loss(values, data['values']))
                loss = tf.reduce_mean(tf.concat(losses, axis=0))
            varis = self.value_model.trainable_variables
            grads = tape.gradient(loss, varis)
            self.value_optimizer.apply_gradients(zip(grads, varis))
            history['loss'][epoch] = loss
        if self.value_model is not None:
            self.value_model.get_weights()  # for fast wrappers
        return history

    def load_value_weights(self, filename):
        """Load weights from filename into the value model."""
        if self.value_model is not None:
            self.value_model.load_weights(filename)

    def save_value_weights(self, filename):
        """Save the current weights in the value model to filename."""
        if self.value_model is not None:
            self.value_model.save_weights(filename)


@tf.function(experimental_relax_shapes=True)
def pg_surrogate_loss(new_prob, old_prob, advantages):
    """Return loss with gradient for policy gradient.

    Parameters
    ----------
    new_probs : Tensor (batch_dim,)
        The output of the current model for the chosen action.
    old_prob : Tensor (batch_dim,)
        The previous probability of the chosen action.
    advantages : Tensor (batch_dim,)
        The computed advantages.

    Returns
    -------
    loss : Tensor (batch_dim,)
        The loss for each interaction.

    """
    return -tf.math.log(new_prob) * advantages


class PGAgent(Agent):
    """A policy gradient agent.
    
    Parameters
    ----------
    policy_network : network
        The network for the policy model.
    
    """

    def __init__(self, policy_network, **kwargs):
        super().__init__(policy_network, **kwargs)
        self.policy_loss = pg_surrogate_loss


def ppo_surrogate_loss(eps=0.2):
    """Return loss function with gradient for proximal policy optimization.

    Parameters
    ----------
    eps : float
        The clip ratio for PPO.

    """
    @tf.function(experimental_relax_shapes=True)
    def loss(new_prob, old_prob, advantages):
        """Return loss with gradient for proximal policy optimization.

        Parameters
        ----------
        new_probs : Tensor (batch_dim,)
            The output of the current model for the chosen action.
        old_probs : Tensor (batch_dim,)
            The old model probability for the chosen action.
        advantages : Tensor (batch_dim,)
            The computed advantages.

        Returns
        -------
        loss : Tensor (batch_dim,)
            The loss for each interaction.
        """
        min_adv = tf.where(advantages > 0, (1 + eps) * advantages, (1 - eps) * advantages)
        return -tf.minimum(new_prob / old_prob * advantages, min_adv)
    return loss


class PPOAgent(Agent):
    """A proximal policy optimization agent.

    Parameters
    ----------
    policy_network : network
        The network for the policy model.
    eps : float, optional
        The clip ratio for PPO.
        
    """

    def __init__(self, policy_network, eps=0.2, **kwargs):
        super().__init__(policy_network, **kwargs)
        self.policy_loss = ppo_surrogate_loss(eps)
