'''
Bryon Kucharski
CMPSCI687 - Reinforcement Learning
Fall 2018

Collection of different RL enviornments

'''



import numpy as np
import time


class gridworld687:

    def __init__(self,num_states,actions,gamma):
        self.actions = actions
        self.num_actions = len(self.actions)
        self.num_states = num_states
        self.gamma = gamma
        self.state = None

    def reset(self):
        """
        Go back to d_0
        """
        self.state = (0,0)
        return self.convert_tuple_state_to_single_state(self.state)
    
    def get_random_action(self):
        """
        Choose a random action to take
        """
        return np.random.choice(self.actions)

    def take_action(self,a):
        """
        returns new state given an action and current state
        """
        s = self.state
        if a == "AL":
            if s[0] != 0 and s != (3,2) and s != (3,3) :
                return (s[0]-1, s[1])
        elif a == "AR":
            if s[0] != 4 and s != (1,2) and s != (1,3) :
                return (s[0]+1, s[1])
        elif a == "AU":
            if s[1] != 0 and s != (2,4):
                return (s[0], s[1]-1)
        elif a == "AD":
            if s[1] != 4 and s != (2,1):
                return (s[0], s[1]+1)
        return s

    def veer(self,attempt,veer_type):
        """
        #veer_type 0 = veer left
        #veer_type 1 = veer right
        attempt is the action it is trying to take
        a new action after the veer is returned
        This is called once the enviornment has already decided to veer
        """
        if attempt == "AL":
            if veer_type == 0:
                a = "AD" # go down
            else:
                a = "AU" #go up
        elif attempt == "AR":
            if veer_type == 0:
                a = "AU"
            else:
                a = "AD"
        elif attempt == "AU":
            if veer_type == 0:
                a = "AL"
            else:
                a = "AR"
        elif attempt == "AD":
            if veer_type == 0:
                a = "AR"
            else:
                a = "AL"
        return a

    def get_state_prime(self,a):
        """
        given state, perform an action. Action may be changed based on the probability of veering/staying
        returns the new state action the action is taken
        """
        a = self.actions[a]
        s = self.state
        prob = np.random.rand()
        new_a = a
        if prob <= 0.8: #80% of time
            s_ = self.take_action(a)
        elif prob > 0.8 and prob <= 0.9: #10% of time
            new_a = "-"
            s_ = s # do nothing
        elif prob > 0.9 and prob <= 0.95: #5% of time
            #veer right
            new_a = self.veer(a,1)#changes depending on where robot is trying to go
            s_ = self.take_action(new_a)
        elif prob > 0.95 and prob <= 1.0: #5% of time
            #veer left!
            new_a = self.veer(a,0)
            s_ = self.take_action(new_a)
        return s_, new_a

    def get_reward(self,s_):
        """
        return 10 if in goal state, -10 if in water state, 0 else
        """
        if s_ == (4,4):
            return 10
        if s_ == (2,4):
            return -10
        return 0

    def is_terminal_state(self):
        """
        Goal state?
        """
        return (self.state == (4,4))

    def draw_world(self):
        """
        Draws a 4x4 text based representation of the current state
        """
        s = self.state
        for j in range(5):
            for i in range(5):
                if (i,j) == s:
                    print('R', end='')
                elif (i,j) == (2,4):
                    print('W', end='')
                elif (i,j) == (4,4):
                    print('G')
                elif (i,j) == (2,3) or (i,j) == (2,2):
                    print('X', end='')
                else:
                    print('-', end='')
                print(" ", end = '')
            print(" ")
        
    def convert_tuple_state_to_single_state(self,tuple_state):
       return int(tuple_state[0] + np.sqrt(self.num_states)*tuple_state[1]) #assumes square gridworld

        
    def step(self,action):
        
        s = self.state
        s_, actual_a = self.get_state_prime(action)
        r = self.get_reward(s_)

        self.state = s_
        if self.is_terminal_state():
            done = True
        else:
            done = False
        return self.convert_tuple_state_to_single_state(s_),r,done
    
    def softmax_action_selection(self,s,theta, sigma):
        
        state_index = self.convert_tuple_state_to_single_state(s) #ex: converts (0,1) to 1, converts (4,4) to 24
        action_weights = np.array_split(theta,self.num_states)[state_index] #split theta into chunks, return the actions that correspond to current s
        #subtract max for numerical stability 
        action_weights = action_weights - action_weights.max()

        probs = []
        for a in range(self.num_actions):
            prob = np.exp(sigma * action_weights[a]) / np.sum(np.exp(sigma * action_weights))
            probs.append(prob)
        
    
        #assert np.sum(probs) == 1.0 #make sure this is a valid probability distribution
        action = np.random.choice( self.num_actions,1,p = probs) #selects from the list of actions based on probabilities of each action
        #print(state_index,action_weights,probs,action)
        #print(probs)
        return int(action)
    
    def simulate(self,N,action_selection = 'softmax',sigma = 0.0 ,theta = None, draw = False):
        returns = []
        st = time.time()
        for i in range(N):
            rewards = []
            s = self.reset()
            j = 0
            
            while True:
                if action_selection == 'softmax':
                    action_index = self.softmax_action_selection(s,theta,sigma)
                    action = self.actions[action_index]
                elif action_selection == 'optimal':
                    action = self.get_optimal_action(s)
                elif action_selection == 'random':
                    action = self.get_random_action() 
                
                
                s_, reward, done = self.step(s,action)
                rewards.append(reward)
               
                if done or j > 1000:
                    rt = self.get_return(rewards)
                    returns.append(rt)
                    break
                else:
                    s = s_
                j+=1 
       
        return (np.sum(returns) / N)
        

class cartpole():
    

    def __init__(self,x_inc= None,x_dot_inc = None,theta_inc = None,theta_dot_inc = None,gamma=1.0):
        self.motor_force = 10.0 # position of neg is based on action
        self.gravity = 9.8 #m/s
        self.cart_mass = 1.0
        self.pole_mass = 0.1
        self.pole_half_length = 0.5
        self.delta_t = 0.02
        self.max_time = 20.2 #seconds

        self.x_inc = x_inc
        self.x_dot_inc = x_dot_inc
        self.theta_inc = theta_inc
        self.theta_dot_inc = theta_dot_inc
        
        self.max_x_distance = 3.0
        self.fail_angle = np.pi / 2.0

        self.num_actions = 2
        self.actions = [0,1]

        self.gamma = gamma

        #self.x_states = np.around(np.arange(-self.max_x_distance,self.max_x_distance+ self.x_inc,self.x_inc),4)
        self.x_states = np.array([-3,0,-3])
        #print(self.x_states)
        #self.x_dot_states = np.around(np.arange(-10,10+self.x_dot_inc,self.x_dot_inc),4)
        self.x_dot_states = np.array([-5,0,-5])
        #print(self.x_dot_states)
        
        #self.theta_states = np.around(np.arange(-self.fail_angle,self.fail_angle ,self.theta_inc),4)
        self.theta_states = np.array([-0.0174532, -0.1047192,0,0.1047192,0.0174532])
        #print(self.theta_states)
        #self.theta_dot_states = np.around(np.arange(-np.pi,np.pi,self.theta_dot_inc),4)
        self.theta_dot_states = np.array([-0.87266,0,0.87266])
        #print(self.theta_dot_states)

        self.num_states = len(self.x_states) * len(self.x_dot_states) * len(self.theta_states) * len(self.theta_dot_states)
        #print(self.num_states)

        self.min_state = np.array([-3,-10,-self.fail_angle,-np.pi])
        self.max_state = np.array([ 3, 10, self.fail_angle, np.pi])

        self.viewer = None
        self.state = None
        self.elapsed_time = 0.0

    def reset(self):
        self.elapsed_time = 0.0
        self.state = (0,0,0,0)
        return self.state
    
    def set_state(self,s):
        self.state = s
        return self.state

    def get_state(self):
        return self.state
    
    
    def increment_time(self):
        self.elapsed_time += self.delta_t
        if self.elapsed_time > self.max_time:
            return True
        return False

    def discrestize_state(self,s):
        #x_prime =           round(round(s[0]/self.x_inc)*self.x_inc,4)
        #x_dot_prime =       round(round(s[1]/self.x_dot_inc)*self.x_dot_inc,4)
        #theta_prime =       round(round(s[2]/self.theta_inc)*self.theta_inc,4)
        #theta_dot_prime =   round(round(s[3]/self.theta_dot_inc)*self.theta_dot_inc,4)

        x_prime = (np.abs(self.x_states-s[0])).argmin()
        x_dot_prime = (np.abs(self.x_dot_states-s[1])).argmin()
        theta_prime = (np.abs(self.theta_states-s[2])).argmin()
        theta_dot_prime = (np.abs(self.theta_dot_states-s[3])).argmin()

        return (x_prime, x_dot_prime, theta_prime,theta_dot_prime )
    def normalize_state(self,state):
        return (state-self.min_state)/(self.max_state-self.min_state)       
    
    def step(self, action):
        x,x_dot,theta,theta_dot = self.state
        force = 0
        if action == 1:
            force = self.motor_force
        elif action == 0:
            force = self.motor_force * -1

        #use equations from paper

        masses = self.cart_mass + self.pole_mass

        theta_double_dot = (self.gravity*np.sin(theta) + np.cos(theta)*((-force - self.pole_mass*self.pole_half_length*(theta_dot*theta_dot) * np.sin(theta) ) / (masses))) / (self.pole_half_length * ((4.0/3.0)  - (self.pole_mass*(np.cos(theta) * np.cos(theta)) / masses ) ))


        x_double_dot = (force + self.pole_mass*self.pole_half_length*((theta_dot*theta_dot) * np.sin(theta) - theta_double_dot*np.cos(theta))) / masses
    
        #update to new state
        x_prime = x + self.delta_t * x_dot
        x_dot_prime = x_dot + self.delta_t * x_double_dot
        theta_prime = theta + self.delta_t*theta_dot
        theta_dot_prime = theta_dot + self.delta_t*theta_double_dot

        #max cap
        if theta_dot_prime > np.pi:
            theta_dot_prime = np.pi
        elif theta_dot_prime < -np.pi:
            theta_dot_prime = -np.pi

        if x_dot_prime > 10:
            x_dot_prime = 10
        elif x_dot_prime < -10:
            x_dot_prime = 10

        state_prime = (x_prime,x_dot_prime,theta_prime,theta_dot_prime)
        
        cond1 = (x_prime > self.max_x_distance or x_prime < -self.max_x_distance)
        cond2 = (theta_prime > self.fail_angle) or (theta_prime < -self.fail_angle)
        cond3 = self.increment_time()
   
        done = cond1 or cond2 or cond3

        if not done:
            reward =  1.0
        else:
            reward =  0.0

        self.state = state_prime

        return state_prime, reward, done

    def softmax_action_selection(self,theta, sigma):
        
        x_index, x_dot_index, theta_index, theta_dot_index = self.discrestize_state(self.state)
        #print(self.discrestize_state(self.state))
        #x_index =           int(np.where(self.x_states==discrete_state[0])[0])
        #x_dot_index =       int(np.where(self.x_dot_states==discrete_state[1])[0])
        #theta_index =       int(np.where(self.theta_states==discrete_state[2])[0])
        #theta_dot_index =   int(np.where(self.theta_dot_states==discrete_state[3])[0])
  
        state_index =   (len(self.x_states)-x_index) \
                        *(len(self.x_dot_states) - x_dot_index) \
                        *(len(self.theta_states) - theta_index) \
                        * (len(self.theta_dot_states)-theta_dot_index) -1

        #print(state_index)
    

        action_weights = np.array_split(theta,self.num_states)[state_index] #split theta into chunks, return the actions that correspond to current s
        #subtract max for numerical stability 
        action_weights = action_weights - action_weights.max()

        probs = []
        for a in range(self.num_actions):
            prob = np.exp(sigma * action_weights[a]) / np.sum(np.exp(sigma * action_weights))
            probs.append(prob)
        
        #assert np.sum(probs) == 1.0 #make sure this is a valid probability distribution
        action = np.random.choice( self.num_actions,1,p = probs) #selects from the list of actions based on probabilities of each action
        #print(state_index,action_weights,probs,action)
        #print(probs)
        return int(action)

    def get_random_action(self):
        return np.random.choice(self.num_actions)

    def simulate(self,N,action_selection = 'softmax',sigma = 0.0 ,theta = None, draw = False):
        returns = []
        for i in range(N):
            rewards = []
            self.reset()
            s = self.state
            j = 0
            
            while True:
                if action_selection == 'softmax':
                    action_index = self.softmax_action_selection(theta,sigma)
                    action = self.actions[action_index]

                s_, reward, done = self.step(action)
                rewards.append(reward)
               
                if done:
                    rt = self.get_return(rewards)
                    returns.append(rt)
                    break
      
        return (np.sum(returns) / N)

    def get_return(self,trajectory):
        """
        Calcualte discounted future rewards base on the trajectory of an entire episode
        """
        r = 0.0
        for i in range(len(trajectory)):
            r += self.gamma**i * trajectory[i]
       
        return r

    def render(self, mode='human'):
        '''
        This code was taken directly from openai implementation of cartpole just to test my equations in the step function. This was the *only* code taken from openai
        '''
        screen_width = 600
        screen_height = 400

        world_width = self.max_x_distance*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

class mountain_car():
    def __init__(self):
        self.action_dim = 3
        self.action = [0,1,2]
        self.state_dim = 2

        self.gamma = 1.0

        self.terminal_x = 0.5

        self.min_state = np.array([-1.2,-0.07])
        self.max_state = np.array([0.5,0.07])
        self.viewer = None
        self.reset()

    def reset(self):
        self.state = np.array([-0.5,0])
        return self.state

    def normalize_state(self,state):
        return (state-self.min_state)/(self.max_state-self.min_state)

    def step(self, action):

        x, velocity = self.state

        velocity_ = velocity + (0.001*(action-1)) - 0.0025*np.cos(3*x)
        x_ = x + velocity_

        if x_ < -1.2:
            x_ = -1.2
            velocity_ = 0
        if x_ > 0.5:
            x_ = 0.5
            velocity_ = 0

        state_prime = np.array([x_,velocity_])

        if x_ >= 0.5:
            done = True
            reward = 0
        else:
            done = False
            reward = -1
        self.state = state_prime
        return state_prime, reward, done

    def get_return(self, trajectory):
        """
        Calcualte discounted future rewards base on the trajectory of an entire episode
        """
        r = 0.0
        for i in range(len(trajectory)):
            r += self.gamma ** i * trajectory[i]

        return r

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def _height(self, xs):
        return np.sin(3 * xs) * .45 + .55

    def render(self, mode='human'):
        '''
        This code was taken directly from openai implementation of mountain car just to test my equations in the step function. This was the *only* code taken from openai
        '''
        screen_width = 600
        screen_height = 400

        world_width = self.max_state[0] - self.min_state[0]
        scale = screen_width / world_width
        carwidth = 40
        carheight = 20

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_state[0], self.max_state[0], 100)
            ys = self._height(xs)
            xys = list(zip((xs - self.min_state[0]) * scale, ys * scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight / 2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth / 4, clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight / 2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth / 4, clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (0.5 - self.min_state[0]) * scale
            flagy1 = self._height(0.5) * scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)])
            flag.set_color(.8, .8, 0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation((pos - self.min_state[0]) * scale, self._height(pos) * scale)
        self.cartrans.set_rotation(np.cos(3 * pos))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

