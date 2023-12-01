#!/usr/bin/env python3

# Univariate Marginal Distribution Algorithm

import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import time

from ple import PLE

class UMDAc():

    def __init__(self, gen_size, net_size, activation,
                 env, max_steps=None, seed=0, 
                 action_mode='argmax', iterations=1, 
                 display_info=False):
        
        ## Global variables
        self.gen_size = gen_size
        self.net_size = net_size 
        self.activation = activation
        self.iterations = iterations
        
        self.seed = seed
        self.max_steps = max_steps

        ## Detect environment, OpenAI or PLE
        try:
            ## Environment is from OpenAI Gym
            self.state_size = env.observation_space.shape[0]
            self.openai = True
            self.ple = False

            self.env = env ## Environment

            try:
                ## Size of action vector agent can take
                self.action_size = env.action_space.n
            except:
                ## Size of action vector agent can take
                self.action_size = env.action_space.shape[0]
            
        except:
            ## Environment is from PLE
            self.openai = False
            self.ple = True
            
            self.game = env
            ## Init environment
            self.env = PLE(self.game, fps=30, display_screen=True,
                          rng=0)
            ## Allowed actions set
            self.allowed_actions = list(self.env.getActionSet())
            self.action_size = len(self.allowed_actions)
            #self.state_size = len(self.game.getGameState())
            self.state_size = self._ple_get_state().shape[1]

        if display_info:
            ## Print environment info
            print('\n' + '#'*5, ' Environment data: ' ,'#'*5)
            print('Type (Autodected): ', 'Gym' if self.openai else 'PLE')
            print('State size: ', self.state_size)
            print('Action size: ', self.action_size)
            print('')
            print('Iterations: ', self.iterations)
            print('')

        '''
        ACTION MODE:
            Determines how output data from neural network
            will be treated. Three options:
                - raw
                - argmax
                - tanh
        '''
        self.action_mode = action_mode

        self.fitness = {} # Init fitness log
        
        ## Create first generation randomly
        self.gen = {} # Init generation 0

        ## Create random specimens
        for i in range(gen_size):
            ## Generate specimen weights and biases
            specimen = {}
            ## First layer
            specimen['h0'] = np.random.uniform(
                -1, 1, [self.state_size, net_size[0]])
            specimen['b0'] = np.random.uniform(
                -1, 1, [1, net_size[0]])

            ## Intermediate layers 
            h_i = 1
            for layer in net_size[1:]:
                ## Generate hidden layers and biases    
                specimen['h'+str(h_i)] = np.random.uniform(-1,1,
                    [net_size[h_i-1], net_size[h_i]])
                specimen['b'+str(h_i)] =  np.random.uniform(-1,1,
                    [1, net_size[h_i]])

                h_i += 1
            
            ## Last layer 
            specimen['h'+str(h_i)] = np.random.uniform(-1,1,
                    [net_size[h_i-1], self.action_size])
            specimen['b'+str(h_i)] = np.random.uniform(-1,1,
                    [1, self.action_size])

            ## Add specimen to generation
            self.gen['s'+str(i)] = specimen
            ## Add specimen to fitness log, init with fitness
            ## value of 0
            self.fitness['s'+str(i)] = 0.

            ## Create a dictionary to hold new specimens
            self.new = {}
            
            ## First new specimen (reference specimen)
            reference = {}

            reference['h0'] = np.empty([self.state_size, net_size[0]])
            reference['b0'] = np.empty([1, net_size[0]])
            ## Intermediate layers 
            h_i = 1
            for layer in net_size[1:]:
                ## Generate hidden layers and biases    
                reference['h'+str(h_i)] = np.empty([net_size[h_i-1], 
                                                    net_size[h_i]])
                reference['b'+str(h_i)] =  np.empty([1, net_size[h_i]])

                h_i += 1
            
            ## Last layer 
            reference['h'+str(h_i)] = np.empty([net_size[h_i-1], 
                                                self.action_size])
            reference['b'+str(h_i)] = np.empty([1, self.action_size])

            ## Add reference to dict
            self.new['n0'] = reference
            

    def show(self, name, show_weights=False):
        ## For every layer in specimen 
        for l_i in range(int(len(self.gen[name])/2)):
            ## Print info about layer and bias
            print('-'*5, " layer Nº", str(l_i), ' ', '-'*5)
            print(' * Neurons: ', 
                  self.gen[name]['h'+str(l_i)].shape[1],'\n',
                  '* Weights of each neuron: ', 
                  self.gen[name]['h'+str(l_i)].shape[0],'\n',
                  '* Biases: ', self.gen[name]['b'+str(l_i)].shape[1]
                  , '\n')
            
            if show_weights:
                ## Show weight values
                print("* Weights:")
                print(self.gen[name]['h'+str(l_i)])
                print("* Biases:")
                print(self.gen[name]['b'+str(l_i)])
                print('')

    def pass_forward(self, feature, specimen):

        in_data = feature ## Load input data

        for l_i in range(int(len(specimen)/2)):
            ## Pass through weights and sum 
            h_z = np.dot(in_data, specimen['h'+str(l_i)]
                        ) + specimen['b'+str(l_i)]
            ## Activation function 
            h_a = self.activation(h_z)
            ## Pass data to next layer
            in_data = h_a
        ## Return las activation
        return h_a

    def gym_evaluate(self, specimen,  
                    render=False, 
                    time_sleep=.0):

        seed = self.seed ## Initial random seed
        reward_log = [] ## For later use in total reward sum if iterations > 1 
        for iters in range(self.iterations):

            ## Reset environment 
            self.env.seed(seed)
            state = self.env.reset()

            t_reward = 0 ## Reset total reward
            
            if self.max_steps != None:
                ## Finite time steps
                for step in range(self.max_steps):
                    ## Render env
                    if render:
                        self.env.render()

                    ## Pass forward state data 
                    output = self.pass_forward(state, specimen)

                    ## Format output to use it as next action
                    if self.action_mode == 'argmax':
                        action = np.argmax(output[0])

                    elif self.action_mode == 'raw':
                        action = output[0]

                    elif self.action_mode == 'tanh':
                        action = np.tanh(output[0])

                    ## Run new step
                    state, reward, done, _ = self.env.step(action)
                    time.sleep(time_sleep) ## Wait time

                    ## Add current reard to total
                    t_reward += reward
                    
                    if done:
                        break
                ## Used if iterations > 1
                reward_log.append(t_reward)
                ## Update seed to test agent in different scenarios
                seed += 1

            else:
                ## Test agent until game over
                done = False
                while not done:
                    ## Render env
                    if render:
                        self.env.render()

                    ## Pass forward state data 
                    output = self.pass_forward(state, specimen)

                    ## Format output to use it as next action
                    if self.action_mode == 'argmax':
                        action = np.argmax(output[0])

                    elif self.action_mode == 'raw':
                        action = output[0]

                    elif self.action_mode == 'tanh':
                        action = np.tanh(output[0])

                    ## Run new step
                    state, reward, done, _ = self.env.step(action)
                    time.sleep(time_sleep) ## Wait time

                    ## Add current reard to total
                    t_reward += reward
                    ## End game if game over
                    if done:
                        break
                ## Used if iterations > 1
                reward_log.append(t_reward)
                seed += 1 ## Update random seed 
                
        ## Disable random seed
        ''' This prevents the algorithm to generate the
            same random numbers all time.   '''
        np.random.seed(None)
        ## Sum of total rewards in all iterations
        return sum(reward_log)

    def _ple_get_state(self):
        ## Adapt game observation to
        ## useful state vector
        observation = self.game.getGameState()
        state = []
        for item in observation:

            data = observation[item]

            if type(data) is dict:
                for d in data:
                    inf = np.array(data[d]).flatten()
                    for dt in inf:
                        state.append(dt)

            elif type(data) is list:                
                data = np.array(data).flatten()                    
                for val in data:
                    state.append(val)
            else:
                state.append(data)

        return np.array([state])

    def ple_evaluate(self, specimen, time_sleep=.0):
            
        ## Set initial random seed 
        np.random.seed(self.seed)
        
        class MyRandom():
            def __init__(self, seed):
                pass
                #np.random.seed(seed)
                #np.random.seed(0)
                #self.seed = seed
            def random_sample(self, size=None):
                return np.random.random_sample(size) 
            def choice(self, a, size=None, replace=True, p=None):
                return np.random.choice(a, size, replace, p)
            def random_integers(self, rmin, rmax):
                return np.random.randint(rmin, rmax)
            def uniform(self, low=0.0, high=1.0, size=None):
                return np.random.uniform(low, high, size)
            def rand(self):
                return np.random.rand()

        reward_log = [] ## Log of all total rewards

        if self.max_steps != None:

            for i in range(self.iterations):

                ## Initialize game 
                self.game.rng = MyRandom(self.seed)
                self.game.init() ## Reset game
                t_reward = .0 ## Reset total reward 

                for time_step in range(self.max_steps):
                    ## Get state 
                    state = self._ple_get_state()
                    ## Output from specimen for given state 
                    output = self.pass_forward(state, specimen)
                    ## Covert specimen output to action
                    act = self.allowed_actions[np.argmax(output[0])]
                    ## Take action
                    self.env.act(act)
                    ## Wait time useful if render is enabled 
                    time.sleep(time_sleep)
                    ## Update total reward
                    t_reward = self.env.score()
                    ## End game if game over
                    if self.env.game_over():
                        break

                ## Log reward for later sum
                reward_log.append(t_reward)

        else:
            ## Finite number of time 
            for i in range(self.iterations):

                ## Initialize game 
                self.game.rng = MyRandom(self.seed)
                self.game.init()
                t_reward = .0 ## Reset total reward 

                while not self.env.game_over():
                    ## Get state 
                    state = self._ple_get_state()
                    ## Take action
                    output = self.pass_forward(state, specimen)
                    act = self.allowed_actions[np.argmax(output[0])]
                    self.env.act(act)
                    ## Useful if random enabled
                    time.sleep(time_sleep)
                    ## Update total reward 
                    t_reward = self.env.score()
                ## Log all total rewards
                reward_log.append(t_reward)

        ## Disable random seed
        ''' This prevents the algorithm to generate the
            same random numbers all time.   '''
        np.random.seed(None)
        ## Sum all total rewards 
        return sum(reward_log)

    def train(self, n_surv, n_random_surv):
        
        ## Collect data about generation
        survivors = list(self.fitness.keys()) ## Survivors' names
        survivors_fitness = list(self.fitness.values()) ## Survivors's fitnesses

        worsts = [] ## Worst specimens names
        worsts_fitness = [] ## Worst specimens fitness values

        ## Select best fitness survivors
        n_r = len(survivors) - n_surv ## Number of not survivor specimens 
        for n in range(n_r):
            
            ## Select worst specimen
            indx = survivors_fitness.index(min(survivors_fitness))
            ## Save worsts 
            worsts.append(survivors[indx])    
            worsts_fitness.append(survivors_fitness[indx])
            ## Delete worsts from survivors lists
            del survivors[indx]
            del survivors_fitness[indx]

        ## Randomly select bad specimens to survive
        for i in range(n_random_surv):
            ## Random index
            indx = np.random.randint(len(worsts))
            ## Add random specimen to survivors 
            survivors.append(worsts[indx])
            survivors_fitness.append(worsts_fitness[indx])
            ## Update worst specimens' lists
            del worsts[indx]
            del worsts_fitness[indx]
        
        ## Generate new specimens (empty):
        for i in range(len(worsts)):
            self.new['n'+str(i)] = copy.deepcopy(self.gen['s0'])

        for param in self.gen['s0']:
            ## For each parameter
            for i in range(self.gen['s0'][param].shape[0]):
                for j in range(self.gen['s0'][param].shape[1]):
                    ## layer[i][j] weight of each survivor 
                    w = []
                    ## For each survivor
                    for name in survivors:
                        w.append(self.gen[name][param][i][j])

                    ## NOTE: Experimental
                    #n_mut = int(len(w)*.3)
                    #muts = np.random.rand(n_mut)

                    #w = np.array(w)
                    #np.random.shuffle(w)
                    #
                    #w = np.delete(w, range(len(w)-n_mut, len(w)), 0)

                    #w = np.hstack((w, muts))
                    #np.random.shuffle(w)
                    ## END OF NOTE
                    
                    ## Compute weights list's mean 
                    mean = np.mean(w)
                    ## Standard deviation
                    std = np.std(w)
                    
                    ## Get samples
                    samples = np.random.normal(mean, std, 
                                               len(worsts))
                    
                    i_sample = 0 ##  Iterator
                    ## Generate new specimens
                    for name in self.new:
                        ## Update weight  
                        self.new[name][
                            param][i][j] = samples[i_sample]
                        i_sample += 1 
        
        ## After generating a set of new specimens
        new_names = []
        new_fitness = []

        for name in self.new:
            ## Load specimen
            specimen = self.new[name]
            ## Evaluate new specimens
            ## and store data for later comparison
            new_names.append(name)

            if self.openai:
                new_fitness.append(self.gym_evaluate(specimen))

            elif self.ple:
                new_fitness.append(self.ple_evaluate(specimen))

        '''
        Selection. Replace all specimens in the worsts list
        with best specimens of the to_select lists.
        '''
        to_select_names = new_names+worsts
        to_select_fitness = new_fitness+worsts_fitness

        for i in range(len(worsts)):
            indx = np.argmax(to_select_fitness)
            
            ## Add selected specimen to new generation
            if 'n' in to_select_names[indx]:
                ## Replace specimen
                self.gen[worsts[i]] = copy.deepcopy(self.new[
                to_select_names[indx]])

            else:
                ## Replace specimen
                self.gen[worsts[i]] = copy.deepcopy(self.gen[
                to_select_names[indx]])

            ## Update selection lists
            del to_select_names[indx]
            del to_select_fitness[indx]

    def add_neurons(self, layer_name, n_neurons=1):

        ## To all specimens in generation
        for name in self.gen:

            ## Load specimen
            specimen = self.gen[name]

            last_indx = int(len(specimen) / 2) - 1 ## Number of layers
            sel_indx = int(layer_name[1]) ## Selected layer's index

            ## Add neuron to layer
            new_neuron = np.random.rand(
                specimen[layer_name].shape[0], n_neurons)
            specimen[layer_name] = np.hstack(
                (specimen[layer_name], new_neuron))
             
            ## Add new bias 
            new_bias = np.random.rand(1,n_neurons)
            specimen['b'+str(sel_indx)] = np.hstack(
                (specimen['b'+str(sel_indx)], new_bias))

            ## Check if the selected layer is 
            ## the last (output layer) of the net
            if sel_indx != last_indx:
                next_layer = specimen['h'+str(sel_indx+1)]
                ## Selected layer isn't the last 
                ## Generate new weights
                new_w = np.random.rand(n_neurons, next_layer.shape[1])
                ## Add weights to next layer
                specimen['h'+str(sel_indx+1)] = np.vstack(
                    (new_w, next_layer))

    def add_layer(self, n_neurons):
    ## Add one layer to all specimens
    ## The new layer is added before
    ## the output layer
        
        ## Define network's layers
        specimen = self.gen['s0']
        layers = []
        layers_shape = []
        biases = []
        biases_shape = []
        for l in specimen:
            if 'h' in l:
                layers.append(l)               
                layers_shape.append(specimen[l].shape)
            elif 'b' in l:
                biases.append(l)
                biases_shape.append(specimen[l].shape)

        for name in self.gen:
            ## Load specimen
            specimen = self.gen[name]
            ## Reset output layer 
            new_o = np.random.rand(n_neurons, self.action_size)
            ## Reset output layer bias 
            new_o_b = np.random.rand(1, self.action_size)
            ## Create new layer
            new_l = np.random.rand(layers_shape[-2][1], 
                                  n_neurons)
            new_l_b = np.random.rand(1, n_neurons)

            specimen[layers[-1]] = new_l 
            specimen[biases[-1]] = new_l_b
            specimen['h'+str(len(layers))] = new_o
            specimen['b'+str(len(biases))] = new_o_b
            

    def save_specimen(self, specimen, filename='specimen0.txt'):
        ## Open file
        f = open(filename, 'w')
        ## Write layers
        for layer in specimen:
            f.write(layer+'\n')
            f.write(str(specimen[layer].tolist())+'\n')

        f.close() # Close file

    def load_specimen(self, filename):

        import ast

        ## Open file
        f = open(filename, 'r')
        ## Init specimen
        specimen = {}
        ## Read file
        array = False
        for line in f.readlines():
            line = line.split('\n')[0] 
            if array:
                ## Covert string to np array
                layer = np.array(ast.literal_eval(line))
                specimen[layer_name] = layer ## Add layer                
                array = False

            else:
                layer_name = line
                array = True
        f.close() ## Close
        return specimen

if __name__ == '__main__':

    import time
    import gym
    from activation.relu import relu
    
    ### HYPERPARAMETERS ###
    ## CartPole
    LOG_FILENAME = 'log10.txt'
    LOG_NOTES = 'gensize:30, nsurv:5, nrandsurv:10'

    NET_SIZE = [1] ## One hidden layer, with one neuron, plus output layer
    ACTIVATION = relu

    GENERATIONS = 50
    GEN_SIZE = 30
    N_SURV = 10
    N_RAND_SURV = 5

    ENV_NAME = 'CartPole-v0'
   
    ITERATIONS = 2
    MAX_STEPS = None
     
    ## Environment initialization
    env = gym.make(ENV_NAME)

    ## Initialize UMDAc
    umdac = UMDAc(GEN_SIZE, NET_SIZE, ACTIVATION, 
                  env, 
                  max_steps=MAX_STEPS,
                  iterations=ITERATIONS, 
                  action_mode='argmax',
                  display_info=True)

    ## Reset training data loggers    
    avg_reward_log = []
    max_rewards = []
    min_rewards = []
    last_avg_reward = 0

    for i in range(GENERATIONS):    
        ## Reset reward logger
        reward_log = []
        for name in umdac.gen:
            ## Load specimen
            specimen = umdac.gen[name]
            ## Tests specimen in environment
            t_reward = umdac.gym_evaluate(specimen, render=False)
            
            reward_log.append(t_reward)
            
            ## Update fitness value
            umdac.fitness[name] = t_reward
        
        ## Train, create new generation
        umdac.train(N_SURV, N_RAND_SURV)    

        ## Calculate and log average reward
        avg_reward = sum(reward_log) / len(reward_log)
        avg_reward_log.append(avg_reward)
        ## Log max reward
        max_rewards.append(max(reward_log))
        min_rewards.append(min(reward_log))

        ## Plot training info online
        plt.clf()

        plt.plot(range(len(min_rewards)), min_rewards,
                 label='Minimum')
        plt.plot(range(len(max_rewards)), max_rewards,
                 label='Maximum')
        plt.plot(range(len(avg_reward_log)), avg_reward_log, 
                 label='Average')
        plt.grid(b=True, which='major', color='#DDDDDD', 
                 linestyle='-')
        plt.legend()
        plt.xlabel('Generation')
        plt.ylabel('Reward')

        plt.draw()
        plt.pause(.00001)

        ## Print some data during training
        print('generation ', i, '/', GENERATIONS,
              ', average reward: ', avg_reward)

    umdac.env.close() ## Close environment 

    ## Save training data 
    f = open(LOG_FILENAME, 'w')
    f.write('Data order: avg, max, min. Notes: '+LOG_NOTES+'\n')
    f.write(str(avg_reward_log)+'\n')
    f.write(str((max_rewards))+'\n')
    f.write(str((min_rewards))+'\n')
    f.close()

    print('Training finished!')
    
    ## Plot training data
    plt.show()

    ## Select best specimen 
    best = list(umdac.fitness.keys())[
    list(umdac.fitness.values()).index(max(
    umdac.fitness.values()))]

    ## Render best speciemens
    print('')
    print('-'*5, ' Rendering best specimen ', '-'*5)

    umdac.iterations = 1
    
    while 1:
        ## For each specimen
        specimen = umdac.gen[best]
        ## Tests specimen in environment
        t_reward = umdac.gym_evaluate(specimen,
                                     render=True)
        print('Total reward: ', t_reward) 
        ## Set random seed to random value
        umdac.seed = np.random.randint(254)
