"""A commented Tensorflow 2.0 Keras implementation of DDPG for open AI gym continuous environments.
This implementation of Deep Deterministic Policy Gradient is different from other implementations
only in that regard, that I kept to keras only style, meaning that also the somewhat complicated
loss function is implemented by keras type custom loss mehtods.
I did this for learning and undestanding only. It is most closely to the OpenAi Spinning up explanation
and thier implementation:
https://spinningup.openai.com/en/latest/algorithms/ddpg.html
https://github.com/openai/spinningup/blob/master/spinup/algos/ddpg/ddpg.py

TODO: Currently it does not apear to converge in any of the tested envs but did not test for long.
Could be only HP optimization problem but can't gurantee.
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/enric/.mujoco/mujoco200/bin')

import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.optimizers as KO
import tensorflow.keras as K

#%%
class Memory:
    """A FIFO experiene replay buffer.
    """
    def __init__(self,obs_dim,act_dim,size):
        self.states = np.zeros([size, obs_dim], dtype=np.float32)
        self.actions = np.zeros([size, act_dim], dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.next_states = np.zeros([size, obs_dim], dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self,state,action,reward,next_state,done):
        self.states[self.ptr] = state
        self.next_states[self.ptr] = next_state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def get_sample(self, batch_size = 32):
        idxs = np.random.randint(0,self.size,size=batch_size)
        return self.states[idxs],self.actions[idxs],self.rewards[idxs],self.next_states[idxs],self.dones[idxs]


class DDPGAgent:
    """The DDPG Agent object.
    The actor and critic are both created in here.
    Requires some intel form the env to be creates.
    Public Methods to interact with:
        :store: store an experience pair, aka trajectory, aka SARS'D
        :get_action: get an action from the current actor
        :get_target_action: get an action from the current target actor - ideally the optimal policy
        :train: call a training run of the network. Will select a randsom batch and uptadte the neural nets and perform a soft update of target -actor and -critic
    Params:
        :state_dim: dimension from the observation space of the environment.
        :action_n: number of actions. Here, it is implicitly expected to work in continous environments as DDPG was designed for.
        :act_limit: min and max of the number generates by the network output. Usually 1.
    """
    def __init__(self,state_dim,action_n,act_limit):
        #env intel
        self.action_n = action_n
        self.state_dim = state_dim
        self.state_n = state_dim[0]
        #constants
        self.ACT_LIMIT = act_limit #requiered for clipping prediciton aciton
        self.GAMMA = 0.99 #discounted reward factor
        self.TAU = 0.005 #soft update factor
        self.BUFFER_SIZE = int(1e6) #total replay buffer size. Should be quite large.
        self.BATCH_SIZE = 100 #training batch size.
        self.ACT_NOISE_SCALE = 0.1 #the noise is for exploration. simple randomnormal is used here insead of OUNoise like in the orgininal paper an din many examples. OUNoise does not bring much benefit.
        #create networks
        self.actor = self._gen_actor_network() #the local actor wich is trained on.
        self.actor_target = self._gen_actor_network() #the target actor which is slowly updated toward optimum
        self.critic = self._gen_critic_network()
        self.critic_target = self._gen_critic_network()
        #Other vital elements
        self.memory = Memory(self.state_n,self.action_n,self.BUFFER_SIZE)
        #Dummies. These are required due to the model declartion and loss declarion
        #style of keras. In short, for complex losses extra inputs need to be declared
        #but for a model.predict call, they obvisually don't play a role so dummies are passed
        self.dummy_Q_target_prediction_input = np.zeros((self.BATCH_SIZE, 1))
        self.dummy_dones_input = np.zeros((self.BATCH_SIZE, 1))


    """-----------------------------ACTOR declarions and methods------------------------------------
    The Actor takes a state and predicts an action
    """
    def _gen_actor_network(self):
        state_input = KL.Input(shape=self.state_dim)
        dense = KL.Dense(400,activation='relu')(state_input)
        dense = KL.Dense(300,activation='relu')(dense)
        out = KL.Dense(self.action_n,activation = 'tanh')(dense)
        model = K.Model(inputs=state_input,outputs=out)
        model.compile(optimizer = 'adam', loss = self._ddpg_actor_loss)
        model.summary()
        return model


    def _ddpg_actor_loss(self,y_true,y_pred):
        #y_true is Q_prediction = Q_critic_predicted(s,a_actor_predicted)
        return -K.backend.mean(y_true)


    def get_action(self,states,noise=None):
        """Returns an action (=prediction of local actor) given a state.
        Adds a gaussion noise for exploration.
        params:
            :state: the state batch
            :noise: add noise. If None defaults self.ACT_NOISE_SCALE is used.
                    If 0 ist passed, no noise is added and clipping passed
        """
        if noise is None: noise = self.ACT_NOISE_SCALE
        if len(states.shape) == 1: states = states.reshape(1,-1)
        action = self.actor.predict_on_batch(states)
        if noise != 0:
            action += noise * np.random.randn(self.action_n)
            action = np.clip(action, -self.ACT_LIMIT, self.ACT_LIMIT)
        return action


    def get_target_action(self,states):
        return self.actor_target.predict_on_batch(states)


    def train_actor(self,states,actions):
        #Q_predictions can not be calculated in keras loss function because it depends on the prediction
        #of another model (critic) with the prediction of this model (actions) as a parameter.
        actions_predict = self.get_action(states, noise=0)
        Q_predictions = self.get_Q(states,actions_predict)
        self.actor.train_on_batch(states,Q_predictions)


    """-----------------------------CRITIC declarion and methods------------------------------------
    The Critic, in the DDPG case, is the Q(s,a)
    """
    def _gen_critic_network(self):
        #Inputs to network. Most of them are for the loss function, not for the feed forward
        state_input = KL.Input(shape=self.state_dim,name='state_input')
        action_input = KL.Input(shape=(self.action_n,),name='action_input')
        Q_target_prediction_input = KL.Input(shape=(1,),name='Q_target_prediction_input')
        dones_input = KL.Input(shape=(1,),name='dones_input')
        #define network structure
        concat_state_action = KL.concatenate([state_input,action_input])
        dense = KL.Dense(400,activation='relu')(concat_state_action)
        dense = KL.Dense(300,activation='relu')(dense)
        out = KL.Dense(1,activation = 'linear')(dense)
        model = K.Model(inputs=[state_input,action_input,Q_target_prediction_input,dones_input],outputs=out)
        model.compile(optimizer = 'adam', loss = self._ddpg_critic_loss(Q_target_prediction_input,dones_input))
        model.summary()
        return model


    def _ddpg_critic_loss(self,Q_target_prediction_input,dones_input):
        def loss(y_true,y_pred):
            #remember: y_true = rewards ; y_pred = Q
            target_Q = y_true + (self.GAMMA * Q_target_prediction_input * (1 - dones_input))
            mse = K.losses.mse(target_Q,y_pred)
            return mse
        return loss


    def train_critic(self,states,next_states,actions,rewards,dones):
        """Train the critic using a trajectory batch from memory.
        This is part of the core algorithm. https://spinningup.openai.com/en/latest/algorithms/ddpg.html
        The loss of the critic requires the predicted Q(s,a) which is the prediction of the critic.
        """
        next_actions = self.get_target_action(next_states)
        Q_target_prediction = self.get_target_Q(next_states,next_actions)
        self.critic.train_on_batch([states,actions,Q_target_prediction,dones],rewards)


    def get_Q(self,states,actions):
        return self.critic.predict([states,actions,self.dummy_Q_target_prediction_input,self.dummy_dones_input])


    def get_target_Q(self,states,actions):
        return self.critic_target.predict_on_batch([states,actions,self.dummy_Q_target_prediction_input,self.dummy_dones_input])


    """-----------------------------AGENT interface and logic------------------------------------
    The Critic, in the DDPG case, is the Q(s,a)
    """

    def _soft_update_actor_and_critic(self):
        """Makes a soft update of the target models with the latest local model weights.
        Uses the factor self.TAU to determine the how soft. Usually, tau is small.
        """
        #Critic soft update:
        weights_critic_local = np.array(self.critic.get_weights())
        weights_critic_target = np.array(self.critic_target.get_weights())
        self.critic.set_weights(self.TAU * weights_critic_local + (1.0-self.TAU)*weights_critic_target)
        #Actor soft update
        weights_actor_local = np.array(self.actor.get_weights())
        weights_actor_target = np.array(self.actor_target.get_weights())
        self.actor_target.set_weights(self.TAU * weights_actor_local + (1.0-self.TAU)*weights_actor_target)


    def store(self,state,action,reward,next_state,done):
        """Stores a trajectory
        Just passes though to memory to keep object structure.
        """
        self.memory.store(state,action,reward,next_state,done)


    def train(self):
        """Trains the networks of the agent (local actor and critic) and soft-updates thier target.
        """
        states,actions,rewards,next_states,dones = self.memory.get_sample(batch_size = self.BATCH_SIZE)
        self.train_critic(states,next_states,actions,rewards,dones)
        self.train_actor(states,actions)
        self._soft_update_actor_and_critic()


#%%
#-----------------------------MAIN------------------------
#The main is from OpenAI spinning up libary
import gym
if __name__ == "__main__":
    GAME = 'LunarLanderContinuous-v2' #'BipedalWalker-v2'   #'LunarLanderContinuous-v2'   #'HalfCheetah-v2'
    GAMMA = 0.99
    EPOCHS = 1000
    MAX_EPISODE_LENGTH = 3000
    START_STEPS = 10000
    RENDER_EVERY = 10

    env = gym.make(GAME)
    agent = DDPGAgent(env.observation_space.shape,env.action_space.shape[0],max(env.action_space.high))

    state, reward, done, ep_rew, ep_len, ep_cnt = env.reset(), 0, False, [0.0], 0, 0
    total_steps = MAX_EPISODE_LENGTH * EPOCHS

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        #render from time for time because it floats my boat
        if ep_cnt % RENDER_EVERY == 0:
            env.render()
        #get action, at the beginning randomly later by neural net output+noise
        if t > START_STEPS:
            action = agent.get_action(state)
            action = np.squeeze(action)
        else:
            action = env.action_space.sample()

        # Step the env
        next_state, reward, done, _ = env.step(action)
        ep_rew[-1] += reward #keep adding to the last element till done
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        done = False if ep_len==MAX_EPISODE_LENGTH else done

        # Store experience to replay buffer
        agent.store(state,action,reward,next_state,done)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        state = next_state

        if done or (ep_len == MAX_EPISODE_LENGTH):
            ep_cnt += 1
            if True: #ep_cnt % RENDER_EVERY == 0:
                print(f"Episode: {len(ep_rew)-1}, Reward: {np.mean(ep_rew[-12:-2])}")
            ep_rew.append(0.0)
            """Perform all DDPG updates at the end of the trajectory.
            Train on a randomly sampled batch as often there were steps in this episode.
            I don't understand why it is updated that often, this is from the TD3 paper and 
            in accordance with the openai implementation.
            """
            for _ in range(ep_len):
                agent.train()

            state, reward, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    #make simple moving average over 5 episodes (smoothing) and plot
    SMA_rewards = np.convolve(ep_rew, np.ones((5,))/5, mode='valid')
    #Plot learning curve
    plt.style.use('seaborn')
    plt.plot(SMA_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()
