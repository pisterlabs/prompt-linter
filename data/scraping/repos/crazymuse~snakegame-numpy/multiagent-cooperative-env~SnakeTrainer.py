import SnakeEnv as snakeEnv
import tensorflow as tf
import numpy as np
import SnakeModel
from baselines.common import tf_util
from baselines.a2c.utils import cat_entropy, mse
from baselines.a2c.utils import Scheduler, make_path, find_trainable_variables
from baselines.a2c.utils import discount_with_dones
from baselines.common import set_global_seeds, explained_variance
import time
from IPython.display import clear_output
from time import sleep

import joblib


# Some of the code blocks have been taken from openAI baselines repo 
# Coding style such as closure and naming conventions of Baselines have been followed to maintain high code quality standards
# and code reusability
# Baselines Github link : https://github.com/openai/baseline




class Util:
    @staticmethod
    def epsilon_greedy_single_discrete(action_prob,num_actions=3,epsilon=0.3):
        randval = np.random.rand();
        best_action = np.argmax(randval)
        if randval<epsilon:
            return best_action
        else:
            a_list = list(range(num_actions))
            a_list.remove(best_action);
            randval2 = np.random.randint(0,num_actions-1)
            return a_list[randval2]

    def discount_with_dones(self,rewards, dones, gamma):
        discounted = []
        r = 0
        for reward, done in zip(rewards[::-1], dones[::-1]):
            r = reward + gamma*r*(1.-done) # fixed off by one bug
            discounted.append(r)
        return discounted[::-1]




class SingleSnakeGameHyperParameters:
    GRID_LENGTH = 5
    GRID_WIDTH = 5
    # Game Parameters
    NUM_ACTIONS  = 3
    # Constant Coefficients 
    ENTROPY_COEFF = 0.002       
    VALUE_FUNC_COEFF = 0.5
    MAX_GRAD_NORM = 0.5
    LEARNING_RATE = 1e-4
    LEARNING_RATE_SCHEDULE = 'linear' # Linear decay of epsilon
    EPSILON=1e-5
    ALPHA = 0.99
    GAMMA_DF = 0.99
    LOG_INTERVAL = 5
    RMS_DECAY = 0.99 #Discounting factor for the history/coming gradient

    #TRAINING PARAMENTERS
    N_SNAKES = 1 # Total number of snakes in a game
    N_ENVS = 25 # Total number of games
    N_STATES = 4 # Total number of states for each move 
    N_STEPS = 10
    N_BATCH = N_STEPS*N_ENVS*N_SNAKES
    N_TIMESTEPS = 500000000 # Total number of iterations (1 iter = 1 batch pumped through model)

    N_SEED = 1

    # TRAINING SHAPES
    OBS_SHAPE = (None, GRID_LENGTH, GRID_WIDTH, 1)
    BATCH_OBS_SHAPE = (N_BATCH,GRID_LENGTH,GRID_WIDTH,1)
    STATE_SHAPE = (None,N_STATES)
    BATCH_STATE_SHAPE = (N_BATCH,N_STATES)
    
    # PLAYING SETTINGS (Play for visual display)
    ENABLE_PLAY = True
    PLAY_SKIP = 240 # Skip from training after 30 seconds
    PLAY_INTERVAL = 30 # Play for 30 seconds
    PLAY_REFRESH_RATE = 10 # Frames per second
    TRAIN_REFRESH_RATE = 30
    # PATHS
    LOG_PATH = './log'
    
    SAVE_SKIP = 1000 # Save after 1000 iterations
    DEFAULT_SAVE_PATH = "./model/model.ckpt"
    
    def update_params(self,n_envs,n_snakes,grid_length,grid_width):
        if n_envs != None:
            self.N_ENVS=n_envs
        if n_snakes != None:
            self.N_SNAKES=n_snakes
        if grid_length != None:
            self.GRID_LENGTH = grid_length
            self.GRID_WIDTH = grid_length
        if grid_width != None:
            self.GRID_WIDTH = grid_width
        self.N_BATCH = self.N_STEPS*self.N_ENVS*self.N_SNAKES
        self.OBS_SHAPE = (None, self.GRID_LENGTH, self.GRID_WIDTH, 1)
        self.BATCH_OBS_SHAPE = (self.N_BATCH,self.GRID_LENGTH,self.GRID_WIDTH,1)
        self.STATE_SHAPE = (None,self.N_STATES)
        self.BATCH_STATE_SHAPE = (self.N_BATCH,self.N_STATES)

            
class Model:
    def __init__(self,policy,p,has_state):
        """
        policy : Internal Policy model such as  SnakeModel.CNNPolicy
        p : Hyperparameters required for training
        """
        sess = tf_util.make_session()
        # Tensorflow model initiallization
        step_model  = policy(sess = sess, p = p,train_phase=False,has_state = has_state) # Deploy model settings
        train_model = policy(sess = sess, p = p,train_phase=True,has_state = has_state) # Training model settings
        saver = tf.train.Saver()

        #Step 2 : Initialize the training parameters
        A = tf.placeholder(tf.int32, [p.N_BATCH])
        ADV = tf.placeholder(tf.float32, [p.N_BATCH])
        R = tf.placeholder(tf.float32, [p.N_BATCH])
        LR = tf.placeholder(tf.float32, [])

        #Step 3 : Define the loss Function
        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=A)  # 
        pg_loss = tf.reduce_mean(ADV * neglogpac)
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))
        entropy = tf.reduce_mean(cat_entropy(train_model.pi))
        loss = pg_loss - entropy*p.ENTROPY_COEFF + vf_loss * p.VALUE_FUNC_COEFF

        #Step 4 : Define the loss optimizer
        params = find_trainable_variables("model")
        grads = tf.gradients(loss, params)
        if p.MAX_GRAD_NORM is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, p.MAX_GRAD_NORM) # Clipping the gradients to protect learned weights
        grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=p.RMS_DECAY, epsilon=p.EPSILON)
        _train = trainer.apply_gradients(grads) # This is the variable which will be used 
        lr = Scheduler(v=p.LEARNING_RATE, nvalues=p.N_TIMESTEPS, schedule=p.LEARNING_RATE_SCHEDULE) # Learning rate changes linearly or as per arguments

        # Step 5 : Write down the summary parameters to be used
        writer = tf.summary.FileWriter(p.LOG_PATH)#summary writer

        def train(obs,rewards,masks,actions,values,states):
            """
            obs     : batch x n x m x 1 snake matrix
            rewards : batch x 1 rewards corrosponding to action 
            actions : batch x 1 discrete action taken
            values  : batch x 1 output of value function during the training process  
            """
            advs = rewards - values;
            for step in range(len(obs)):
                cur_lr = lr.value()
            td_map = {train_model.X:obs, train_model.S:states, A:actions, ADV:advs, R:rewards, LR:cur_lr}
            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train],
                td_map
            )
            return policy_loss, value_loss, policy_entropy


        def save(save_path):
            #ps = sess.run(params)
            #make_path(save_path)
            #joblib.dump(ps, save_path)
            saver.save(sess, save_path)

        def load(load_path):
            #loaded_params = joblib.load(load_path)
            #restores = []
            #for p, loaded_p in zip(params, loaded_params):
            #    restores.append(p.assign(loaded_p))
            #ps = sess.run(restores)
            saver.restore(sess,load_path)

        def add_scalar_summary(tag,value,step):
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=value)])
            writer.add_summary(summary, step)
        # Expose the user to closure functions 
        self.train = train
        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.hidden_value = step_model.hidden_value
        self.initial_state = step_model.initial_state
        self.add_scalar_summary = add_scalar_summary
        self.save = save
        self.load = load
        # Initialize global variables and add tf graph
        tf.global_variables_initializer().run(session=sess)
        writer.add_graph(tf.get_default_graph())#write graph
        
        





class SnakeRunner(object):
    """
    This class will take the Model and interface it with Snake Env
    """
    def __init__(self,n_envs=None,n_snakes=None,grid_length=None,grid_width=None):
        self.p = SingleSnakeGameHyperParameters() # Game Parameters;
        self.p.update_params(n_envs=n_envs,n_snakes=n_snakes,grid_length=grid_length,grid_width=grid_width)
        p = self.p
        self.env = snakeEnv.MultiAgentSnakeGame(length = p.GRID_LENGTH, 
                                width =  p.GRID_WIDTH,n_envs=self.p.N_ENVS,n_snakes=self.p.N_SNAKES)
        self.model = Model(policy = SnakeModel.CNNPolicy,p = self.p, has_state = False) # This creates A2C Model from the policy model given by SnakeModel.CNNPolicy
        # Initialize observation
        self.obs = np.zeros((p.N_ENVS*p.N_SNAKES, p.GRID_LENGTH, p.GRID_WIDTH, 1), dtype=np.uint8)
        self.states = self.env.get_state()
        self.dones = [False for _ in range(p.N_ENVS*p.N_SNAKES)]

        
    def run(self,verbose=False):
        """
        Get batchwise data for training
        """
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones,mb_states = [],[],[],[],[],[]
        for n in range(self.p.N_STEPS):
            actions, values , _ = self.model.step(self.obs,self.states)
            mb_obs.append(np.copy(self.obs))
            mb_states.append(np.copy(self.states))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)            
            obs, rewards, dones,states = self.env.step(actions,verbose=verbose)
            self.dones = dones            
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n]*0
            self.obs = obs
            self.states = states
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.p.BATCH_OBS_SHAPE)
        mb_states = np.asarray(mb_states, dtype=np.float).swapaxes(1, 0).reshape(self.p.BATCH_STATE_SHAPE)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_values = self.model.value(self.obs, self.states, self.dones ).tolist()
        #discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = np.array(dones).astype('int32').tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards+[value], dones+[0], self.p.GAMMA_DF)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.p.GAMMA_DF)
            mb_rewards[n] = rewards
        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_rewards, mb_masks, mb_actions, mb_values, mb_states


    def train(self,save_path=""):
        """
        Trains the environment batchwise
        """
        if save_path == "": save_path = self.p.DEFAULT_SAVE_PATH
        tf.reset_default_graph()
        set_global_seeds(self.p.N_SEED)
        
        tstart = time.time()
        last_played=time.time()
        for update in range(1, self.p.N_TIMESTEPS//self.p.N_BATCH+1):
            obs, rewards, masks, actions, values, states = self.run()
            policy_loss, value_loss, policy_entropy = self.model.train(obs, rewards, masks, actions, values, states)
            # Mask is not required for CNNPolicy but still keeping it for generality
            nseconds = time.time()-tstart
            fps = int((update*self.p.N_BATCH)/nseconds)
            
            if update % self.p.LOG_INTERVAL == 0 or update == 1:
                ev = explained_variance(values, rewards)
                # Logging in tensorflow starts
                #self.model.add_scalar_summary("total_timesteps", update*self.p.N_BATCH,update)
                #self.model.add_scalar_summary("fps", fps,update)
                self.model.add_scalar_summary("policy_entropy", float(policy_entropy),update)
                self.model.add_scalar_summary("value_loss", float(value_loss),update)
                self.model.add_scalar_summary("explained_variance", float(ev),update)
                #Logging in tensorflow ends
                self.model.add_scalar_summary("average_reward",np.average(rewards.flatten()),update)
                self.model.add_scalar_summary("average_collisions",np.sum(rewards<-0.1),update)
                self.model.add_scalar_summary("average_food_eaten",np.sum(rewards>0.5),update)
                self.model.add_scalar_summary("advantage",np.average(np.array(rewards-values,dtype=np.float).flatten()),update)
                # TRAIN DISPLAY
                ##self.env.display(verbose=True,limit = 3)
                ##sleep(1/float(self.p.TRAIN_REFRESH_RATE))
                clear_output(wait=True)
                # PLAY AFTER 30 SECONDS
                print ('TRAINING . . . TIME =',int(time.time()-last_played),'s  , TOTAL TIME : ',int(nseconds) )
            if (time.time()-last_played)>self.p.PLAY_SKIP and self.p.ENABLE_PLAY == True:
                self.play()
                last_played=time.time()
            if update % self.p.SAVE_SKIP == 0 or update == 1:
                self.model.save(save_path)

        self.env.close()
                    
    def play(self):
        tstart = time.time()
        nseconds=0;
        while(nseconds<self.p.PLAY_INTERVAL):
            print ('PLAYING  . . . TIME =',int(nseconds),'s')
            self.step()
            self.env.display(verbose=True,limit = 3)
            sleep(1/float(self.p.PLAY_REFRESH_RATE))
            clear_output(wait=True)
            nseconds = time.time()-tstart

    def step(self,verbose=False):
        """
        Takes only 1 step at a time for  deploying
        """
        actions, values , _ = self.model.step(self.obs,self.states)
        self.obs,rewards,self.dones,self.states = self.env.step(action_list = actions,verbose=verbose)
        return (self.obs,rewards,self.dones)    


    def user_step(self,action,verbose=False):
        """
        Takes only 1 step at a time for  deploying
        Action is ENV1SNAKE1 ENV1SNAKE2 . . .. ENVNSNAKE1 ENVNSNAKE2
        """
        
        actions = np.ones((self.p.N_ENVS*self.p.N_SNAKES,1))*action;
        self.obs,rewards,self.dones,self.states = self.env.step(action_list = actions,verbose=verbose)
        print (self.model.hidden_value(self.obs,self.states))
        return (self.obs,rewards,self.dones)    
    
