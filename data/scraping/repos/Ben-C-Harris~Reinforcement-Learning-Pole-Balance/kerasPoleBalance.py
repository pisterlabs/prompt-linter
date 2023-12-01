import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from PIL import Image

from loggingFunctionality import LoggerOutputs # Import ScoreLogger Class from /scores/score_logger

# Define which openAI Gym environment to use
ENV_NAME = "CartPole-v1"

# User options
DEBUG = False
LOAD_PRIOR_MODEL = False
PRIOR_MODEL_NAME = "kerasModelWeights.h5"
EXPORT_MODEL = False
SAVE_GIFS = False

# Define RL variables
GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 10

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995


class DeepQLearningSolver(LoggerOutputs): # Neural Net for Deep Q-Learning - reinforcement learning algo class

    '''
    init required to init variables but also to init openAI gym space and Keras Model
    
    Option for loading in prior model weights should you wish - define load variables above
       
    Observation: 
        Type: Box(4,)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24°           24°
        3	Pole Velocity At Tip      -Inf            Inf
        
    Action:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right
    '''
    # Class initialization method. ENV_NAME for inherited Class, observation and action space for this class
    def __init__(self, ENV_NAME, observation_space, action_space): 
        
        # init inherited class
        super().__init__(ENV_NAME)
        
        # Set method attributes
        self.exploration_rate = EXPLORATION_MAX

        # Set openAI gym space
        self.action_space = action_space
        
        # Memory managment for RL
        self.memory = deque(maxlen=MEMORY_SIZE)

        # Build sequential keras tensorflow model (Sequential is a linear stack of layers)
        self.model = Sequential() # Define model type
        
        # Build first layer. Input shape is of openAI gym observation space i.e. 4 inputs
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu")) # input shape of four due to four observations
        self.model.add(Dense(24, activation="relu")) 
        self.model.add(Dense(self.action_space, activation="linear")) # output two as only push cart to right or left are required
        
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))
        
        if LOAD_PRIOR_MODEL: # If picking up an earlier saved keras model of type sequential
            self.loadTrainedModelWeights(PRIOR_MODEL_NAME)    
        if DEBUG:
            print("DQNSolver __init__ successful...")
        
    '''
    Remember prior time step variables and what to do next.
    Append them into batch memory which is later used for RL fit/predict.
    '''
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if DEBUG:
            print("Appended to memory batch...")
        

    '''
    act method returns 0 or 1.
       
    1-exploration_rate of the time itll be random, else returns 1...
    
    Starts more random but as time goes on it learns more and more from prior results.
    i.e. weighting on learning gets higher as you go longer as opposed to earlier where
    it is just trying to be random and see what happens
    '''
    def act(self, state):
        if np.random.rand() < self.exploration_rate: # X% of the time NOTE: Exploration Rate changes with steps due to exploration decay
            if DEBUG:
                print("Act method invoked a random left/right motion of cart due to exploration rate")
            return random.randrange(self.action_space) # randomly returns 0 or 1  (i.e. left/right)
        else:
            q_values = self.model.predict(state) # Predict quality value - predicts the confidence of using left or right movement of cart
            if DEBUG:
                print("Act method invoked predicted cart motion: " + str(np.argmax(q_values[0])))
            return np.argmax(q_values[0])

        
    '''
    If enough simulation data currently remembered in self.memory, this method will
    iterate through the batch and using GAMMA (user defined above) will deliver new
    model predict and fit. The exploration rate decay is then applied.
    '''
    def experienceReplay(self):
        if len(self.memory) < BATCH_SIZE: # Only use the previous X amount of runs to influence and train the model on.
            if DEBUG:
                print("Learning not taking place as not enough prior simulations/steps yet...")
            return
        batch = random.sample(self.memory, BATCH_SIZE) # Define batch learning size
        for state, action, reward, state_next, terminal in batch:
            q_update = reward # Updated quality score
            if not terminal: # i.e. reward not negative
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0])) # Gamma = discount factor & max predicted quality score for next step
            q_values = self.model.predict(state) # q_values are the confidence/quality values over whether the cart needs to go left or right in next frame
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY # Apply exploration decay
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate) # Ensure doesnt go below specified minimum value
        
    
    '''
    Export trained Keras model and weights to local dir in multiple formats - uncomment as required
    '''
    def exportModel(self):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open("kerasModelWeights.json", "w") as json_file:
            json_file.write(model_json)
        
        # serialize weights to HDF5
        self.model.save_weights("kerasModelWeights.h5")    
        # save full model
        self.model.save("fullKerasModel.h5")
        
        
    '''
    Load previously trained RL model weights into __init__ of sequential model
    '''    
    def loadTrainedModelWeights(self, modelWeightFile):
       self.model.load_weights(modelWeightFile)
       self.exploration_rate = EXPLORATION_MIN # Already a mature model so exploration not required
       if DEBUG:
           print("Previous Model Weights Loaded...")


    '''
    Export a gif of all relevant frames parsed in
    '''
    def exportGIF(self, GIFname, framesList):
        with open(GIFname, 'wb') as f:
            im = Image.new('RGB', framesList[0].size)
            im.save(f, save_all=True, append_images=framesList) 
        if DEBUG:
            print("GIF Created...")
    

'''
Main method for solving the pole balance cart problem from OpenAI gym.
Initiates multiple runs containing multiple iterations to solve.
Between runs it resets the gym environment.
Results are logged and dependent on user settings creates GIFs of render
and exports/saves Keras model.
'''
def poleBalance():
    env = gym.make(ENV_NAME) # New OpenAI Gym Env
    observation_space = env.observation_space.shape[0] # four observations cart pos&vel & pole angle&vel
    action_space = env.action_space.n # two actions cart goes left or right
    
    # DeepQLearningSolver class object with observation and action space attributes, and inherited class attribute
    dqnObj = DeepQLearningSolver(ENV_NAME, observation_space, action_space)
       
    frames = [] # Empty list to be filled with frames for GIF export
    run = 0 # Start run calculations from zero
    while True: # Each actual simulation run instance
        if DEBUG:
            print("Starting New Simulation...")
        run += 1 # Started a new simulation run so add one
        state = env.reset() # Reset Gym environment
        state = np.reshape(state, [1, observation_space]) # random new state for this run instance
        step = 0 # init starting simulation step at zero
        while True: # Each timestep within said simulation...
            if DEBUG:
                print("Starting New Time Step...")
            step += 1 # New timestep
            env.render() # Render image of current gym simulation
            
            # Call object method act which delivers a 0 or 1 based upon exploration rate/decay (1 == Right & 0 == Left)
            action = dqnObj.act(state) # state will be a first iteration new state, or will be the last time steps state
            if DEBUG:
                if action == 0:
                    print("Cart push Left...")
                elif action == 1:
                    print("Cart push Right...")
            
            '''
            time step forward using random action but as exploration decays use 0 less and 1 more.
            .step creates an Observation(state_next = object), reward(float), terminal(done = bool), info(dict) for each time step
            state_next is essentially what is going on in the gym, rotations velocities etc.
            '''
            # Using action against the current state what has happened - step forward to find out...          
            state_next, reward, terminal, info = env.step(action) 
           
            # If the simulation has not terminated (i.e. failed criteria) then it gets a positive reward.
            # If it has terminated, i.e. the pole has fallen over/fail criteria met then it gets a negative reward
            reward = reward if not terminal else -reward
                        
            state_next = np.reshape(state_next, [1, observation_space])
            
            # Activly rememeber what state you were in, what action you took, 
            # whether that was "rewarding" and what the next state was and then whether it terminated or not.
            dqnObj.remember(state, action, reward, state_next, terminal)
            
            state = state_next # Define state as that of your prior attempt - i.e. previous step influences this step i.e. learninggggggg
            
            frames.append(Image.fromarray(env.render(mode='rgb_array')))  # save each frames

            if terminal:
                print ("Run: " + str(run) + ", exploration: " + str(dqnObj.exploration_rate) + ", score: " + str(step))
                dqnObj.addScore(step, run) # Call inherited method              
                if run % 5 == 0: # Export every 5th run                       
                    if EXPORT_MODEL:
                        if DEBUG:
                            print("Exporting Model")
                        dqnObj.exportModel()
                    if run % 20 == 0: # Export GIF of latest 5 runs every 20 runs
                        gifPath = "./GIFs/cart_R" + str(run) + ".gif"
                        if DEBUG:
                            print("Creating GIF: " + gifPath)  
                        if SAVE_GIFS:
                            dqnObj.exportGIF(gifPath, frames)      
                    frames = [] # Reset

                break            
            dqnObj.experienceReplay() # Actual reinforcement...
    

if __name__ == "__main__":
    poleBalance() # call poleBalance() method
