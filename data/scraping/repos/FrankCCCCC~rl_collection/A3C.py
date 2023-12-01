import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import numpy as np
# import queue

class Agent:
    def __init__(self, state_size, num_action, reward_discount, learning_rate, exploration_strategy, session = None):
        self.state_size = state_size
        self.num_action = num_action
        self.reward_discount = reward_discount
        self.exploration_strategy = exploration_strategy
        self.session = session
        self.iter = 0
        self.eps = 0
        self.is_shutdown_explore = False

        with tfv1.Session() as sess:
            self.data_type = tf.float32
            self.optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
            self.avg_loss = tf.keras.metrics.Mean(name = 'loss')
            self.model = self.build_model('model')

        # For A2C loss function coefficients
        self.coef_entropy = 0
        self.coef_value = 1
        
    def build_model(self, name):
        # # Shared layers
        # nn_input = tf.keras.Input(shape = self.state_size, dtype = self.data_type)
        # x = tf.keras.layers.Dense(units = 64)(nn_input)
        # x = tf.keras.layers.ReLU()(x)
        # x = tf.keras.layers.Dense(units = 128)(x)
        # common = tf.keras.layers.ReLU()(x)

        # # Actor Model
        # actor_layer = tf.keras.layers.Dense(units = 64)(common)
        # actor_layer = tf.keras.layers.ReLU()(actor_layer)
        # actor_layer = tf.keras.layers.Dense(units = self.num_action)(actor_layer)
        # actor_nn_output = tf.keras.activations.softmax(actor_layer)

        # # Critic Model
        # critic_layer = tf.keras.layers.Dense(units = 64)(common)
        # critic_layer = tf.keras.layers.ReLU()(critic_layer)
        # critic_nn_output = tf.keras.layers.Dense(units = 1)(critic_layer)

        # # Combine into a model
        # model = tf.keras.Model(name = name, inputs = nn_input, outputs = [actor_nn_output, critic_nn_output])

        inputs = tf.keras.layers.Input(shape=self.state_size, name = 'inputs')
#         x = tf.keras.layers.Dense(128, activation="relu")(inputs)
        common = tf.keras.layers.Dense(128, activation="relu")(inputs)
        action = tf.keras.layers.Dense(self.num_action, activation="softmax", name = 'action_outputs')(common)
        critic = tf.keras.layers.Dense(1, name = 'value_output')(common)

        model = tf.keras.Model(inputs=inputs, outputs=[action, critic])

        return model

    def predict(self, state):
        return self.model(tf.convert_to_tensor(state, self.data_type))
    
    def loss(self, action_probs, critic_values, rewards):
        # Calculate accumulated reward Q(s, a) with discount
        np_rewards = np.array(rewards)
        num_reward = np_rewards.shape[0]
        discounts = np.logspace(0, num_reward, base = self.reward_discount, num = num_reward)
        
        q_values = np.zeros(num_reward)
        for i in range(num_reward):
            q_values[i] = np.sum(np.multiply(np_rewards[i:], discounts[:num_reward - i]))
        q_values = (q_values - np.mean(q_values)) / (np.std(q_values) + 1e-9)

        # Calculate the Actor Loss and Advantage A(s, a) = Q_value(s, a) - value(s)
        action_log_prbs = tf.math.log(action_probs)
        advs = q_values - critic_values
        actor_loss = -action_log_prbs * advs
        
        
        # Calculate the critic loss 
        huber = tf.keras.losses.Huber()
        critic_loss = huber(tf.convert_to_tensor(critic_values, dtype = self.data_type), tf.convert_to_tensor(q_values, dtype = self.data_type))

        # Calculate the cross entropy of action distribution
        entropy = tf.reduce_sum(action_probs * action_log_prbs * -1)
        
        # Compute loss as formular: loss = Sum of a trajectory(-log(Pr(s, a| Theta)) * Advantage + coefficient of value * Value - coefficient of entropy * cross entropy of action distribution)
        # Advantage: A(s, a) = Q_value(s, a) - value(s)
        # The modification refer to the implement of Baseline A2C from OpenAI
        # Update model with a trajectory Every time.
        return tf.reduce_sum(actor_loss + self.coef_value * critic_loss - self.coef_entropy * entropy)

    def get_metrics_loss(self):
        return self.avg_loss.result()
    
    def reset_metrics_loss(self):
        self.avg_loss.reset_states()

    def select_action(self, state):
        # Predict the probability of each action(Stochastic Policy)
        act_dist, value = self.predict([state])
        # act_dist = tf.squeeze(act_dist)
        # value = tf.squeeze(value)
        # Assume using Epsilon Greedy Strategy
        action = self.exploration_strategy.select_action()
        
        # If the index of action (return value) is -1, choose the action with highest probability that model predict
#         if action == -1 or self.shutdown_explore == True:
#             # Pick then action with HIGHTEST probability
#             act_idx = tf.argmax(act_dist, axis = 0).numpy()
#             return act_idx, act_dist, value
#         else:
#             # If the index of action (return value) is != -1, act randomly    
#             return action, act_dist, value
        # tf.enable_eager_execution() 
        # tf.executing_eagerly()
        # return np.random.choice(self.num_action, p=np.squeeze(act_dist.eval())), act_dist, value
        return tf.squeeze(tf.random.categorical(act_dist, 1)), act_dist, value

    def shutdown_explore(self):
        self.is_shutdown_explore = True
        self.exploration_strategy.shutdown_explore()
    
    def _get_gradients(self, loss, tape, cal_gradient_vars):
        return tape.gradient(loss, cal_gradient_vars)
    
    def update(self, loss, gradients, apply_gradient_vars = None):
        if apply_gradient_vars == None:
            apply_gradient_vars = self.model.trainable_variables
        self.optimizer.apply_gradients(zip(gradients, apply_gradient_vars))
        self.avg_loss.update_state(loss)

        # Update exploration rate of Epsilon Greedy Strategy
        self.exploration_strategy.update_epsilon()

        self.iter += 1
        self.eps += 1

    def train_on_env(self, env, is_show = False, cal_gradient_vars = None):
        # By default, update agent's own trainable variables
        if cal_gradient_vars == None:
            cal_gradient_vars = self.model.trainable_variables
#         if apply_gradient_vars == None:
#             apply_gradient_vars = self.model.trainable_variables
            
        
        episode_reward = 0
        state = env.reset(is_show)

        action_probs = []
        critic_values = []
        rewards = []
        trajectory = []

        while not env.is_over():
            # env.render()
            # with self.session as sess:
            with tf.GradientTape() as tape:
                tape.watch(cal_gradient_vars)
                action, act_prob_dist, value = self.select_action(state)

            action = tf.squeeze(action)
            act_prob_dist = tf.squeeze(act_prob_dist)
            value = tf.squeeze(value)

            # print(act_prob_dist)
            act_prob = tf.gather_nd(act_prob_dist, [action])
            # act_indice = np.arange(self.num_action)
            # print(act_indice)
            # action = tf.gather_nd(act_indice, [action])
            
            # print(act_prob)
            print(action)
            # act_prob = act_prob_dist[action]
            # with tfv1.Session() as sess:
            state_prime, reward, is_done, info = env.act(sess.run(action))
            # print(f'State: {state}, Action: {action}, Reward: {reward}, State_Prime: {state_prime}')
            
            action_probs.append(act_prob)
            critic_values.append(value)
            rewards.append(reward)
            trajectory.append({'state': state, 'action': action, 'reward': reward, 'state_prime': state_prime, 'is_done': is_done})

            state = state_prime
            episode_reward += reward

        with self.session as sess:
            loss = self.loss(action_probs, critic_values, rewards)
            gradients = self._get_gradients(loss, tape, cal_gradient_vars)
        env.reset()

        return episode_reward, loss, gradients, trajectory