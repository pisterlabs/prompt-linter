"""
Actor and critic classes adapted from
https://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html

"""
import numpy as np
import scipy.linalg as spla
import random
from collections import deque
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import spin_simulation as ss

class Action:
    
    def __init__(self, action, type='discrete'):
        """
        Arguments:
            action: If it's discrete, then the action encoding should be an
                array of size 1*numActions. If it's continuous, then action
                should be a tuple (phi, rot, time).
            type: Either 'discrete' or 'continuous'.
        """
        self.action = action
        self.type = type
    
    def __repr__(self):
        return f'phi: {self.getPhi()/np.pi:.02f}pi,'+\
            f'rot: {self.getRot()/np.pi:.02f}pi,'+\
            f'dt: {self.getTime()/1e-6:.02f} microsec'
    
    def getPhi(self):
        """Get the angle phi that specifies the axis of rotation in the
        xy-plane. Should be a value in [0,2*pi].
        
        """
        if self.type == 'discrete':
            ind = np.nonzero(self.action)[0]
            if ind.size > 0:
                ind = ind[0]
            else: # the action is null
                return 0.
            if ind in [0,1]: # X, Xbar
                return 0.
            elif ind in [2,3]: # Y, Ybar
                return np.pi/2
            elif ind == 4: # nothing
                return 0.
        elif self.type == 'continuous':
            return np.mod(self.action[0] * np.pi/2, 2*np.pi)
    
    def getRot(self):
        """Get the rotation angle from the action. Can be positive or negative.
        """
        if self.type == 'discrete':
            ind = np.nonzero(self.action)[0]
            if ind.size > 0:
                ind = ind[0]
            else: # the action is null
                return 0.
            if ind in [0,2]:
                return np.pi/2
            elif ind in [1,3]:
                return -np.pi/2
            elif ind == 4:
                return 0.
        elif self.type == 'continuous':
            return self.action[1] * 2*np.pi
    
    def getTime(self):
        """Get the time (in seconds) from the action encoding.
        
        Ideally want action-time mappings to be 0 -> 0, 1 -> 5e-6.
        """
        if self.type == 'discrete':
            ind = np.nonzero(self.action)[0]
            if ind.size > 0:
                ind = ind[0]
            else: # the action is null
                return 0.
            if ind in [0,1,2,3]:
                return 0.
            elif ind == 4:
                return 5e-6
        elif self.type == 'continuous':
            # return 10.0**((a[..., 2])*1.70757 - 7) -1e-7
            return 10.0**((self.action[2]+1)*0.853785 - 7) -1e-7
    
    def format(self):
        if self.type == 'discrete':
            ind = np.nonzero(self.action)[0]
            if ind.size > 0:
                ind = ind[0]
            else:
                return ''
            return ['X', 'Xbar', 'Y', 'Ybar', 'delay'][ind]
        else:
            if self.getRot() != 0:
                # non-zero rotation
                return f"phi={self.getPhi()/np.pi:.02f}pi, " + \
                    f" rot={self.getRot()/np.pi:.02f}pi, " + \
                    f"t={self.getTime()*1e6:.02f} microsec"
            else:
                # no rotation -> delay
                if self.getTime() != 0:
                    return f'delay, t={self.getTime()*1e6:.02f} microsec'
                else:
                    # no rotation, no time
                    return ''
    
    def clip(self):
        """Clip the action to give physically meaningful information.
        """
        if self.type == 'continuous':
            self.action = np.array([np.clip(self.action[0], -1, 1), \
                          np.clip(self.action[1], -1, 1), \
                          np.clip(self.action[2], -1, 1)])
    
    def print(self):
        print(self.format())
        
    def get_propagator(self, N, dim, H, discretePropagators=None):
        """Convert an action a into the RF Hamiltonian H.
        
        TODO: change the action encoding to (phi, strength, t) to more easily
            constrain relevant parameters (minimum time, maximum strength)
        
        Arguments:
            a: Action performed on the system. The action is a 1x3 array
                containing the relevant information for a rotation over some
                time.
            H: Time-independent Hamiltonian.
        
        Returns:
            The propagator U corresponding to the time-independent Hamiltonian and
            the RF pulse
        """
        # TODO make getting propagator easier for discrete actions
        if self.type == 'discrete':
            ind = np.nonzero(self.action)[0][0]
            return discretePropagators[ind]
        elif self.type == 'continuous':
            J = ss.getAngMom(np.pi/2, self.getPhi(), N, dim)
            rot = self.getRot()
            time = self.getTime()
            return spla.expm(-1j*(H*time + J*rot))
    

def formatActions(actions, type='discrete'):
    """Format a list of actions nicely
    """
    str = ''
    i=0
    for a in actions:
        strA = Action(a, type=type).format()
        if strA != '':
            str += f'{i}: ' + strA + '\n'
        i += 1
    return str

class Environment(object):
    
    def __init__(self, N, dim, coupling, delta, sDim, Htarget, X, Y,\
            type='discrete', delay=5e-6, delayAfter=False):
        """Initialize a new Environment object
        
        Arguments:
            delayAfter: Should there be a delay after every pulse/delay?
        """
        self.N = N
        self.dim = dim
        self.coupling = coupling
        self.delta = delta
        self.Htarget = Htarget
        self.X = X
        self.Y = Y
        
        self.sDim = sDim
        self.type = type
        self.delay = delay
        self.delayAfter = delayAfter
        
        self.reset()
    
    def makeDiscretePropagators(self):
        """Make a discrete number of propagators so that I'm not re-calculating
        the propagators over and over again.
        
        To simplify calculations, define each action as a pulse (or no pulse)
        followed by a delay
        """
        Udelay = spla.expm(-1j*(self.Hint*self.delay))
        Ux = spla.expm(-1j*(self.X*np.pi/2))
        Uxbar = spla.expm(-1j*(self.X*-np.pi/2))
        Uy = spla.expm(-1j*(self.Y*np.pi/2))
        Uybar = spla.expm(-1j*(self.Y*-np.pi/2))
        if self.delayAfter:
            Ux = Udelay @ Ux
            Uxbar = Udelay @ Uxbar
            Uy = Udelay @ Uy
            Uybar = Udelay @ Uybar
        self.discretePropagators = [Ux, Uxbar, Uy, Uybar, Udelay]
    
    def reset(self, randomize=True):
        """Resets the environment by setting all propagators to the identity
        and setting t=0
        """
        # randomize dipolar couplings and get Hint
        if randomize:
            _, self.Hint = ss.get_H(self.N, self.dim, \
                self.coupling, self.delta)
        # initialize propagators to delay
        if self.delayAfter:
            self.Uexp = ss.get_propagator(self.Hint, self.delay)
            self.Utarget = ss.get_propagator(self.Htarget, self.delay)
        else:
            self.Uexp = np.eye(self.dim, dtype="complex128")
            self.Utarget = np.copy(self.Uexp)
        # initialize time
        self.t = 0
        if self.delayAfter:
            self.t += self.delay
        # for network training, define the "state" (sequence of actions)
        self.state = np.zeros((32, self.sDim), dtype="float32")
        # depending on time encoding, need to set this so that t=0
        if self.type == 'continuous':
            self.state[:,2] = -1
        self.tInd = 0 # keep track of time index in state
        
        # and recalculate propagators if discrete
        if self.type == 'discrete':
            self.makeDiscretePropagators()
    
    def copy(self):
        """Return a copy of the environment
        """
        return Environment(self.N, self.dim, self.coupling, self.delta, \
            self.sDim, self.Htarget, self.X, self.Y, type=self.type, \
            delay=self.delay, delayAfter=self.delayAfter)
        
    
    def getState(self):
        return np.copy(self.state)
    
    def act(self, action):
        """Evolve the environment corresponding to an action and the
        time-independent Hamiltonian
        
        Arguments:
            action: An instance of Action class.
        """
        # TODO change below when doing finite pulse widths/errors
        if self.delayAfter:
            dt  = self.delay
        else:
            dt = action.getTime()
        if self.tInd < np.size(self.state, 0):
            if self.type == 'discrete':
                self.Uexp = action.get_propagator(self.N, self.dim, self.Hint, \
                    self.discretePropagators) @ self.Uexp
            else:
                self.Uexp = action.get_propagator(self.N, self.dim, self.Hint) \
                    @ self.Uexp
            if dt > 0:
                self.Utarget = ss.get_propagator(self.Htarget, dt) @ \
                                self.Utarget
                self.t += dt
            self.state[self.tInd,:] = action.action
            self.tInd += 1
        else:
            print('ran out of room in state array, not evolving state')
            print(self.state)
            print(f'tInd: {self.tInd}, t: {self.t}')
    
    def reward(self):
        return -1.0 * (self.t > 1e-6) * \
            np.log10((1-ss.fidelity(self.Utarget,self.Uexp))+1e-100)

        # isTimeGood = 1/(1 + np.exp((15e-6-self.t)/2e-6))
        # return -1.0 * isTimeGood * np.log10((1 - \
        #     np.power(ss.fidelity(self.Utarget, self.Uexp), 20e-6/self.t)) + \
        #     1e-100)
        #
        # isTimeGood = self.t >= 15e-6
        # isDelay = 1-self.state[self.tInd-1,1] # if there's no rotation
        # return -1.0 * isTimeGood * isDelay * np.log10((1 - \
        #     np.power(ss.fidelity(self.Utarget, self.Uexp), 20e-6/self.t)) + \
        #     1e-100)
    
    def isDone(self):
        """Returns true if the environment has reached a certain time point
        or once the number of state variable has been filled
        TODO modify this when I move on from constrained (4-pulse) sequences
        """
        return self.tInd >= np.size(self.state, 0)

class NoiseProcess(object):
    """A noise process that can have temporal autocorrelation
    
    Scale should be a number between 0 and 1.
    
    TODO need to add more sophisticated noise here...
    """
    
    def __init__(self, scale):
        self.scale = scale
    
    def copy(self):
        """Copy noise process
        """
        return NoiseProcess(self.scale)
    
    def getNoise(self):
        return np.array( \
            [np.random.normal(loc=0, scale=.05*self.scale) + \
                np.random.choice([-1,-.5,.5,1,0], \
                p=[self.scale/4,self.scale/4,self.scale/4,self.scale/4,\
                    1-self.scale]), \
             np.random.normal(loc=0, scale=.05*self.scale) + \
                np.random.choice([-1,-.5,.5,1,0], \
                p=[self.scale/4,self.scale/4,self.scale/4,self.scale/4,\
                    1-self.scale]), \
             np.random.normal(loc=0, scale=.05*self.scale) + \
                np.random.choice([-.5,.5,0], \
                p=[self.scale/2,self.scale/2,1-self.scale])])

class ReplayBuffer(object):
    """Define a ReplayBuffer object to store experiences for training
    """
    
    def __init__(self, bufferSize):
        self.bufferSize = bufferSize
        self.size = 0
        self.buffer = deque()
        
    def add(self, s, a, r, s1, d):
        """Add an experience to the buffer.
        If the buffer is full, pop an old experience and
        add the new experience.
        
        Arguments:
            s: Old environment state on which action was performed
            a: Action performed
            r: Reward from state-action pair
            s1: New environment state from state-action pair
            d: Is s1 terminal (if so, end the episode)
        """
        exp = (s,a,r,s1,d)
        if self.size < self.bufferSize:
            self.buffer.append(exp)
            self.size += 1
        else:
            self.buffer.popleft()
            self.buffer.append(exp)
    
    def getSampleBatch(self, batchSize, powerOfTwo=True):
        """Get a sample batch from the replayBuffer
        
        Arguments:
            batchSize: Size of the sample batch to return. If the replay buffer
                doesn't have batchSize elements, return the entire buffer
            powerOfTwo: Boolean, whether to return a sample batch of size 2^n.
        
        Returns:
            A tuple of arrays (states, actions, rewards, new states, and d)
        """
        batch = []
        
        size = np.minimum(self.size, batchSize)
        
        if powerOfTwo:
            size = int(2**(np.floor(np.log2(size))))
        
        batch = random.sample(self.buffer, size)
        sBatch = np.array([_[0] for _ in batch])
        aBatch = np.array([_[1] for _ in batch])
        rBatch = np.array([_[2] for _ in batch])
        s1Batch = np.array([_[3] for _ in batch])
        dBatch = np.array([_[4] for _ in batch])
        
        return sBatch, aBatch, rBatch, s1Batch, dBatch
    
    def clear(self):
        self.buffer.clear()
        self.size = 0

def mutateMat(mat, mutateStrength=1, mutateFrac=.1, \
        superMutateProb=.01, resetProb=.01):
    """Method to perform mutations on a given nd-array
    
    Arguments:
        mat: Matrix on which to perform mutations.
        mutateStrength: Strength of mutation. Corresponds to the variance of
            the random number that the weight is multiplied by.
        superMutateProb: Probability of a "super-mutation." This is the
            probability _given_ that a mutation occurs.
        resetProb: Probability of resetting the weight. This is the
            probability _given_ that a mutation occurs.
    """
    # choose which elements to mutate
    mutateInd = np.random.choice(mat.size, \
            int(mat.size * mutateFrac), replace=False)
    superMutateInd = mutateInd[0:int(mat.size*mutateFrac*superMutateProb)]
    resetInd = mutateInd[int(mat.size*mutateFrac*superMutateProb):\
        int(mat.size*mutateFrac*(superMutateProb + resetProb))]
    mutateInd = mutateInd[int(mat.size*mutateFrac*(superMutateProb+resetProb)):]
    
    # perform mutations on mat
    mat[np.unravel_index(superMutateInd, mat.shape)] *= \
        np.random.normal(scale=100*mutateStrength, size=superMutateInd.size)
    mat[np.unravel_index(resetInd, mat.shape)] = \
        np.random.normal(size=resetInd.size)
    mat[np.unravel_index(mutateInd, mat.shape)] *= \
        np.random.normal(scale=mutateStrength, size=mutateInd.size)



class Actor(object):
    """Define an Actor object that learns the deterministic policy function
    pi(s): state space -> action space
    """
    
    def __init__(self, sDim=3, aDim=3, learningRate=1e-3, type='discrete'):
        """Initialize a new Actor object
        
        Arguments:
            sDim: Dimension of state space.
            aDim: Dimension of action space. If discrete, it's the number of
                actions that can be performed. If continuous, it's the degrees
                of freedom for an action.
            learningRate: Learning rate for optimizer.
            type: The type of actor, either 'discrete' or 'continuous'. If
                'discrete', then the actor learns a stochastic policy which
                gives the propensity of performing a discrete number of
                actions. If 'continuous', then the actor learns a deterministic
                policy.
        """
        self.sDim = sDim
        self.aDim = aDim
        self.learningRate = learningRate
        self.type = type
        self.model = None
        
        self.optimizer = keras.optimizers.SGD(learning_rate=learningRate)
    
    def createNetwork(self, lstmLayers, denseLayers, lstmUnits, denseUnits,\
        normalizationType='layer'):
        """Create the network
        
        Arguments:
            lstmLayers: The number of LSTM layers to process state input
            denseLayers: The number of fully connected layers
            normalizationType: 'layer' uses layer normalization, 'batch' uses
                batch normalization.
        """
        self.model = keras.Sequential()
        # add LSTM layers
        if lstmLayers == 1:
            self.model.add(layers.LSTM(lstmUnits,\
                input_shape=(None,self.sDim,), \
                # bias_initializer=tf.random_normal_initializer(stddev=.05), \
                # unit_forget_bias=True, \
                ))
        elif lstmLayers == 2:
            self.model.add(layers.LSTM(lstmUnits, \
                input_shape=(None,self.sDim,), \
                # bias_initializer=tf.random_normal_initializer(stddev=.05), \
                # unit_forget_bias=True, \
                return_sequences=True, \
                ))
            self.model.add(layers.LSTM(lstmUnits))
        elif lstmLayers > 2:
            self.model.add(layers.LSTM(lstmUnits, \
                input_shape=(None, self.sDim,), \
                # bias_initializer=tf.random_normal_initializer(stddev=.05), \
                # unit_forget_bias=True, \
                return_sequences=True, \
                ))
            for i in range(lstmLayers-2):
                self.model.add(layers.LSTM(lstmUnits, \
                    # bias_initializer=tf.random_normal_initializer(stddev=.05), \
                    # unit_forget_bias=True, \
                    return_sequences=True))
            self.model.add(layers.LSTM(lstmUnits, \
                # bias_initializer=tf.random_normal_initializer(stddev=.05), \
                # unit_forget_bias=True,\
                ))
        else:
            raise("Problem making the network...")
        
        # add dense layers
        for i in range(denseLayers):
            if normalizationType == 'layer':
                self.model.add(layers.LayerNormalization())
            elif normalizationType == 'batch':
                self.model.add(layers.BatchNormalization())
            else:
                raise('Problem adding normalization layer')
            self.model.add(layers.Dense(denseUnits, activation="elu"))
        # add output layer
        # depends on whether the actor is discrete or continuous
        if self.type == 'discrete':
            self.model.add(layers.Dense(self.aDim, activation='softmax'))
        elif self.type == 'continuous':
            self.model.add(layers.Dense(self.aDim, activation="elu", \
            kernel_initializer=\
                tf.random_uniform_initializer(minval=-1e-3,maxval=1e-3), \
            bias_initializer=\
                tf.random_uniform_initializer(minval=-1e-3,maxval=1e-3), \
            ))
        else:
            raise('problem creating output layer for actor')
    
    def predict(self, states, training=False):
        """
        Predict policy values from given states
        
        Arguments:
            states: A batchSize*timesteps*sDim array.
        """
        if len(np.shape(states)) == 3:
            # predicting on a batch of states
            return self.model(states, training=training)
        elif len(np.shape(states)) == 2:
            # predicting on a single state
            return self.model(np.expand_dims(states,0), training=training)[0]
    
    #@tf.function
    def trainStep(self, batch, critic):
        """Trains the actor's policy network one step
        using the gradient specified by the DDPG algorithm (if continuous)
        or using REINFORCE with baseline (if discrete)
        
        Arguments:
            batch: A batch of experiences from the replayBuffer. `batch` is
                a tuple: (state, action, reward, new state, is terminal?).
            critic: A critic to estimate the Q-function
        """
        batchSize = len(batch[0])
        # calculate gradient
        if self.type == 'continuous':
            with tf.GradientTape() as g:
                Qsum = tf.math.reduce_sum( \
                        critic.predict(batch[0], \
                                       self.predict(batch[0], training=True)))
                # scale gradient by batch size and negate to do gradient ascent
                Qsum = tf.math.multiply(Qsum, -1.0 / batchSize)
            gradients = g.gradient(Qsum, self.model.trainable_variables)
            self.optimizer.apply_gradients( \
                zip(gradients, self.model.trainable_variables))
        elif self.type == 'discrete':
            # perform gradient ascent for actor-critic
            with tf.GradientTape() as g:
                # TODO include gamma factors? Ignoring for now...
                # N*1 tensor of delta values
                delta = batch[2] + tf.math.multiply(1-batch[4],\
                    critic.predict(batch[3])) - critic.predict(batch[0])
                # N*1 tensor of policy values
                policies = tf.math.multiply(self.predict(batch[0]), batch[1])
                policies = tf.math.reduce_sum(policies, axis=1)
                loss = tf.math.multiply(-1.0/batchSize, tf.math.reduce_sum( \
                    tf.math.multiply(delta, tf.math.log(policies))
                ))
            gradients = g.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients( \
                zip(gradients, self.model.trainable_variables))
    
    def save_weights(self, filepath):
        """Save model weights in ckpt format
        """
        self.model.save_weights(filepath)
    
    def load_weights(self, filepath):
        self.model.load_weights(filepath)
    
    def getParams(self):
        return self.model.get_weights()
    
    def setParams(self, params):
        return self.model.set_weights(params)
    
    def copyParams(self, actor, polyak=0):
        """Update the network parameters from another actor, using
        polyak averaging, so that
        theta_self = (1-polyak) * theta_self + polyak * theta_a
        
        Arguments:
            
            polyak: Polyak averaging parameter between 0 and 1
        """
        params = self.getParams()
        aParams = actor.getParams()
        copyParams = [params[i] * (1-polyak) + aParams[i] * polyak \
                           for i in range(len(params))]
        self.setParams(copyParams)
    
    def copy(self):
        """Copy the actor and return a new actor with same model
        and model parameters.
        """
        copy = Actor(self.sDim, self.aDim, self.learningRate, type=self.type)
        copy.model = keras.models.clone_model(self.model)
        copy.setParams(self.getParams())
        return copy
    
    def paramDiff(self, a):
        """Calculate the Frobenius norm for network parameters between network
        and another network.
        """
        diff = [np.mean((_[0] - _[1])**2) for _ in \
                        zip(self.getParams(), a.getParams())]
        # diff = np.linalg.norm(diff)
        return diff
    
    def getAction(self, state, noiseProcess=None):
        """Get action from policy.
        """
        a = self.predict(state)
        if self.type == 'continuous':
            if noiseProcess is not None:
                a += noiseProcess.getNoise()
            a = Action(a, type=self.type)
            a.clip()
        if self.type == 'discrete':
            # pick an action according to probability distribution
            p = np.array(a, dtype='float32')
            p = p/p.sum(0)
            ind = np.random.choice(self.aDim, p=p)
            a = np.zeros((self.aDim), dtype='float32')
            a[ind] = 1
            a = Action(a, type=self.type)
        return a
        
    
    def evaluate(self, env, replayBuffer=None, noiseProcess=None, numEval=1,\
            candidatesFile=None):
        """Perform a complete play-through of an episode, and
        return the total rewards from the episode.
        """
        fTot = 0.
        # delay = Action(np.array([0,0,0,0,1]), type='discrete')
        for i in range(numEval):
            f=0.
            env.reset()
            # env.act(delay) # start with delay
            s = env.getState()
            done = False
            ind = 0
            while not done:
                a = self.getAction(s, noiseProcess)
                env.act(a)
                # env.act(delay) # add delay
                r = env.reward()
                s1 = env.getState()
                done = env.isDone()
                if replayBuffer is not None:
                    replayBuffer.add(s,a.action,r,s1, done)
                s = s1
                f = np.maximum(f, r)
                if f == r:
                    fInd = ind
                ind += 1
            if f > 5 and candidatesFile is not None:
                candidatesFile.write('Candidate pulse sequence identified:\n'+\
                    formatActions(s, type=self.type) + '\n\n')
                candidatesFile.write(f'Fitness:\t{f:.02f} (fInd: {fInd})\n')
            fTot += f
        return fTot/numEval, fInd
    
    def test(self, env, critic=None):
        """Test the actor's ability without noise. Return the actions it
        performs and the rewards it gets through the episode
        
        Arguments:
            env: Environment instance
            critic: Critic. If it's passed, then evaluate the state-action or
                state values as predicted by the critic. Return as the third
                element of the tuple.
        """
        rMat = []
        criticMat = []
        env.reset()
        # delay = Action(np.array([0,0,0,0,1]), type='discrete')
        # env.act(delay) # add delay
        s = env.getState()
        done = False
        while not done:
            if critic is not None:
                criticMat.append(critic.predict(s))
            a = self.getAction(s)
            env.act(a)
            # env.act(delay) # add delay
            rMat.append(env.reward())
            s = env.getState()
            done = env.isDone()
        if critic is None:
            return s, rMat
        else:
            return s, rMat, criticMat
    
    def crossover(self, p1, p2, weight=0.5):
        """Perform evolutionary crossover with two parent actors. Using
        both parents' parameters, copies their "genes" to this actor.
        
        Many choices for crossover methods exist. This implements the simplest
        uniform crossover, which picks "genes" from either parent with
        probabilities weighted by
        
        Arguments:
            p1, p2: Actors whose parameters are crossed, then copied to self.
            weight: Probability of selecting p1's genes to pass to child.
                Should probably be dependent on the relative fitness of the
                two parents.
        """
        childParams = self.getParams()
        p1Params = p1.getParams()
        p2Params = p2.getParams()
        for i in range(len(p1Params)):
            if np.random.rand() < weight:
                childParams[i] = p1Params[i]
            else:
                childParams[i] = p2Params[i]
        self.setParams(childParams)
    
    def mutate(self, mutateStrength=1, mutateFrac=.1, \
    superMutateProb=.01, resetProb=.01):
        """Mutate the parameters for the neural network.
        
        Arguments:
            mutateFrac: Fraction of weights that will be mutated.
            superMutateProb: Probability that a "super mutation" occurs
                (i.e. the weight is multiplied by a higher-variance random
                number).
            resetProb: Probability that the weight is reset to a random value.
        """
        params = self.getParams()
        for i in range(len(params)):
            mutateMat(params[i], mutateStrength, mutateFrac, superMutateProb, \
                resetProb)
    


class Critic(object):
    """Define a Critic that learns the Q-function or value function for
    associated policy.
    
    Q: state space * action space -> R
    which gives the total expected return by performing action a in state s
    then following policy
    
    V: state space -> total expected rewards
    """
    
    def __init__(self, sDim=3, aDim=3, gamma=.99, learningRate=1e-3, type='V'):
        """Initialize a new Actor object
        
        Arguments:
            sDim: Dimension of state space
            aDim: Dimension of action space
            gamma: discount rate for future rewards
            learningRate: Learning rate for optimizer.
            type: Q function ('Q') or value function ('V').
        """
        self.sDim = sDim
        self.aDim = aDim
        self.gamma = gamma
        self.learningRate = learningRate
        self.type = type
        
        self.model = None
        self.optimizer = keras.optimizers.SGD(learning_rate=learningRate)
        self.loss = keras.losses.MeanSquaredError()
    
    def createNetwork(self, lstmLayers, denseLayers, lstmUnits, denseUnits,\
        normalizationType='layer'):
        """Create the network, either a Q-network or value network depending
        on the critic type.
        
        Arguments:
            lstmLayers: The number of LSTM layers to process state input
            denseLayers: The number of fully connected layers
            normalizationType: Same as for actor... TODO update
        """
        stateInput = layers.Input(shape=(None, self.sDim,), name="stateInput")
        if self.type == 'Q':
            actionInput = layers.Input(shape=(self.aDim,), name="actionInput")
        # add LSTM layers
        if lstmLayers == 1:
            stateLSTM = layers.LSTM(lstmUnits, \
                # bias_initializer=tf.random_normal_initializer(stddev=.05), \
                # unit_forget_bias=True,
                )(stateInput)
        elif lstmLayers == 2:
            stateLSTM = layers.LSTM(lstmUnits, \
                # bias_initializer=tf.random_normal_initializer(stddev=.05), \
                # unit_forget_bias=True, \
                return_sequences=True)(stateInput)
            stateLSTM = layers.LSTM(lstmUnits)(stateLSTM)
        elif lstmLayers > 2:
            stateLSTM = layers.LSTM(lstmUnits, \
                # bias_initializer=tf.random_normal_initializer(stddev=.05), \
                # unit_forget_bias=True, \
                return_sequences=True)(stateInput)
            for i in range(lstmLayers-2):
                stateLSTM=layers.LSTM(lstmUnits, \
                    # bias_initializer=tf.random_normal_initializer(stddev=.05), \
                    # unit_forget_bias=True, \
                    return_sequences=True)(stateLSTM)
            stateLSTM = layers.LSTM(lstmUnits, \
                # bias_initializer=tf.random_normal_initializer(stddev=.05), \
                # unit_forget_bias=True, \
                )(stateLSTM)
        else:
            print("Problem making the network...")
            raise
        if self.type == 'Q':
            # stateHidden = layers.Dense(int(denseUnits/2))(stateLSTM)
            stateHidden = stateLSTM
            # actionHidden = layers.Dense(int(denseUnits/2))(actionInput)
            actionHidden = layers.Dense(denseUnits)(actionInput)
            # concatenate state, action inputs
            x = layers.concatenate([stateHidden, actionHidden])
        else:
            # creating value function, state input only
            # x = layers.Dense(denseUnits)(stateLSTM)
            x = stateLSTM
        
        # add fully connected layers
        for i in range(denseLayers):
            if normalizationType == 'layer':
                x = layers.LayerNormalization()(x)
            elif normalizationType == 'batch':
                x = layers.BatchNormalization()(x)
            else:
                raise('Problem adding normalization layer')
            x = layers.Dense(denseUnits, activation="elu")(x)
        output = layers.Dense(1, name="output", \
            kernel_initializer=\
                tf.random_uniform_initializer(minval=-1e-3,maxval=1e-3), \
            bias_initializer=\
                tf.random_uniform_initializer(minval=-1e-3,maxval=1e-3), \
            )(x)
        if self.type == 'Q':
            self.model = keras.Model(inputs=[stateInput, actionInput], \
                outputs=[output])
        elif self.type == 'V':
            self.model = keras.Model(inputs=[stateInput], outputs=[output])
        else:
            raise('Whoops, problem making critic network')
    
    def predict(self, states, actions=None, training=False):
        """
        Predict Q-values or state values for given inputs
        """
        if len(np.shape(states)) == 3:
            # predicting on a batch of states/actions
            if self.type == 'Q':
                return self.model({"stateInput": states,\
                        "actionInput": actions}, \
                    training=training)
            else:
                return self.model({"stateInput": states}, training=training)
                
        elif len(np.shape(states)) == 2:
            # predicting on a single state/action
            if self.type == 'Q':
                return self.model({"stateInput": np.expand_dims(states,0), \
                                   "actionInput": np.expand_dims(actions,0)}, \
                    training=training)[0][0]
            else:
                return self.model({"stateInput": np.expand_dims(states,0)}, \
                    training=training)[0][0]
    
    #@tf.function
    def trainStep(self, batch, actorTarget=None, criticTarget=None):
        """Trains the critic's Q/value function one step
        using the gradient specified by the DDPG algorithm
        
        Arguments:
            batch: A batch of experiences from the replayBuffer
            actorTarget: Target actor
            criticTarget: Target critic
        """
        batchSize = len(batch[0])
        if self.type == 'Q':
            # learn Q function, based on DDPG
            targets = batch[2] + self.gamma * (1-batch[4]) * \
                criticTarget.predict(batch[3], actorTarget.predict(batch[3]))
            # calculate gradient according to DDPG algorithm
            with tf.GradientTape() as g:
                predictions = self.predict(batch[0], batch[1], training=True)
                predLoss = self.loss(predictions, targets)
                predLoss = tf.math.multiply(predLoss, 1.0 / batchSize)
            gradients = g.gradient(predLoss, self.model.trainable_variables)
            self.optimizer.apply_gradients( \
                    zip(gradients, self.model.trainable_variables))
        else:
            # learn value function
            # implementation from Barto and Sutton
            # delta = batch[2] + \
            #     self.gamma * tf.math.multiply(1-batch[4], \
            #         self.predict(batch[3])) -\
            #     self.predict(batch[0])
            # with tf.GradientTape() as g:
            #     values = self.predict(batch[0], training=True)
            #     loss = tf.math.multiply(-1.0/batchSize, \
            #         tf.math.reduce_sum(tf.multiply(delta, values)))
            
            # implementation from OpenAI and others
            # calculate target values using reward and discounted future value
            targets = batch[2] + \
                self.gamma * tf.math.multiply(1-batch[4], \
                    self.predict(batch[3]))
            with tf.GradientTape() as g:
                values = self.predict(batch[0], training=True)
                loss = self.loss(values, targets)
            gradients = g.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients( \
                    zip(gradients, self.model.trainable_variables))
    
    def save_weights(self, filepath):
        """Save model weights in ckpt format
        """
        self.model.save_weights(filepath)
    
    def load_weights(self, filepath):
        self.model.load_weights(filepath)
    
    def getParams(self):
        return self.model.get_weights()
    
    def setParams(self, params):
        return self.model.set_weights(params)
    
    def copyParams(self, a, polyak=0):
        """Update the network parameters from another network, using
        polyak averaging, so that
        theta_self = (1-polyak) * theta_self + polyak * theta_a
        
        Arguments:
            polyak: Polyak averaging parameter between 0 and 1
        """
        params = self.getParams()
        aParams = a.getParams()
        copyParams = [params[i] * (1-polyak) + aParams[i] * polyak \
                           for i in range(len(params))]
        self.setParams(copyParams)
    
    def copy(self):
        """Copy the critic and return a new critic with same model
        and model parameters.
        """
        copy = Critic(self.sDim, self.aDim, self.gamma, self.learningRate,\
            type=self.type)
        copy.model = keras.models.clone_model(self.model)
        copy.setParams(self.getParams())
        return copy
    
    def paramDiff(self, c):
        """Calculate the Frobenius norm for network parameters between network
        and another network.
        
        Returns:
            A list of scalars, where each element is the norm for a particular
            layer in the neural network.
        """
        diff = [np.mean((_[0] - _[1])**2) for _ in \
                        zip(self.getParams(), c.getParams())]
        # diff = np.linalg.norm(diff)
        return diff

    

class Population(object):
    """Population of actors, for evolutionary reinforcement learning.
    
    """
    
    def __init__(self, size=10):
        self.size = size
        self.fitnesses = np.full((self.size,), -1e100, dtype=float)
        self.fitnessInds = np.zeros((self.size), dtype=int)
        self.pop = np.full((self.size,), None, dtype=object)
        # when actor was synced to population from gradient actor
        self.synced = np.zeros((self.size), dtype=int)
        # when actor was last mutated
        self.mutated = np.zeros((self.size), dtype=int)
    
    def startPopulation(self, sDim, aDim, learningRate, type='discrete', \
        lstmLayers=1, denseLayers=4, lstmUnits=64, denseUnits=32,\
        normalizationType='layer'):
        for i in range(self.size):
            self.pop[i] = Actor(sDim, aDim, learningRate, type=type)
            self.pop[i].createNetwork(lstmLayers,denseLayers,\
                lstmUnits, denseUnits, normalizationType)
    
    def evaluate(self, env, replayBuffer, noiseProcess=None, numEval=1,\
            candidatesFile=None):
        """Evaluate the fitnesses of each member of the population.
        
        Arguments:
            candidatesFile: If not none, pass to each individual when it's
                evaluated and save high-reward pulse sequences.
        """
        for i in range(self.size):
            print(f'evaluating individual {i+1}/{self.size},\t', end='')
            self.fitnesses[i], self.fitnessInds[i] = self.pop[i].evaluate(env,\
                replayBuffer, noiseProcess, numEval,\
                candidatesFile=candidatesFile)
            print(f'fitness is {self.fitnesses[i]:.02f}')
    
    def iterate(self, eliteFrac=0.1, tourneyFrac=.2, crossoverProb=.25, \
        mutateProb = .25, mutateStrength=1, mutateFrac=.1, \
        superMutateProb=.01, resetProb=.01, generation=0):
        """Iterate the population to create the next generation
        of members.
        
        Arguments:
            eliteFrac: Fraction of total population that will be marked as
                "elites".
            tourneyFrac: Fraction of population to include in each tournament
                ("tourney").
        """
        # TODO maybe put a bunch of prints in here
        # to make sure it's working as expected
        
        # sort population by fitness
        indSorted = np.argsort(self.fitnesses)
        numElite = int(np.ceil(self.size * eliteFrac))
        indElite = indSorted[-numElite:]
        indOther = indSorted[:-numElite]
        elites = self.pop[indElite]
        eliteFitness = self.fitnesses[indElite]
        print('selected elites')
        # perform tournament selection to get rest of population
        selected = np.full((self.size-numElite), None, dtype=object)
        selectedSynced = np.zeros((self.size-numElite), dtype=int)
        selectedMutated = np.copy(selectedSynced)
        tourneySize = int(np.ceil(self.size * tourneyFrac))
        for i in range(self.size-numElite):
            # pick random subset of population for tourney
            ind = np.random.choice(self.size, tourneySize, replace=False)
            # pick the winner according to highest fitness
            indWinner = ind[np.argmax(self.fitnesses[ind])]
            winner = self.pop[indWinner]
            selectedSynced[i] = self.synced[indWinner]
            selectedMutated[i] = self.mutated[indWinner]
            if winner not in selected and winner not in elites:
                selected[i] = winner
            else:
                selected[i] = winner.copy()
        print('selected rest of population')
        
        # do crossover/mutations with individuals in selected
        for sInd, s in enumerate(selected):
            if np.random.rand() < crossoverProb:
                selectedMutated[sInd] = generation
                eInd = np.random.choice(numElite)
                e = elites[eInd]
                s.crossover(e, s)
            if np.random.rand() < mutateProb:
                selectedMutated[sInd] = generation
                s.mutate(mutateStrength, mutateFrac, \
                superMutateProb, resetProb)
        print('mutated non-elite individuals')
        
        # then reassign them to the population
        self.pop[indElite] = elites
        self.pop[indOther] = selected
        self.synced[indOther] = selectedSynced
        self.mutated[indOther] = selectedMutated
        
        # reset fitnesses
        self.fitnesses = np.full((self.size,), -1e100, dtype=float)
    
    def sync(self, newMember, generation=0):
        """Replace the weakest (lowest-fitness) member with a new member.
        
        """
        ind = np.argmin(self.fitnesses)
        self.pop[ind] = newMember.copy()
        self.synced[ind] = generation
        self.mutated[ind] = generation
        self.fitnesses[ind] = -1e100
