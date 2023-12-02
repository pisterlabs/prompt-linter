import gym
from gym import spaces
import numpy as np
# from os import path
import snakeoil3_gym as snakeoil3
import numpy as np
import copy
import collections as col
import os
import subprocess
import time
import signal
from tensorforce.contrib.openai_gym import OpenAIGym

class TorcsEnv:
    terminal_judge_start = 10  # Speed limit is applied after this step
    termination_limit_progress = 10  # [km/h], episode terminates if car is running slower than this limit
    default_speed = 50

    initial_reset = True

    def start_torcs_process(self):
        if self.torcs_proc is not None:
            os.killpg(os.getpgid(self.torcs_proc.pid), signal.SIGKILL)
            time.sleep(0.5)
            self.torcs_proc = None
        window_title = str(self.port)
 
        command =['/usr/local/bin/torcs',' -nofuel -nodamage -nolaptime -title {} -p {}'.format(window_title, self.port) ] 
        if self.vision is True:
            command += ' -vision'
        time.sleep(np.random.ranf([1])*3)
        self.torcs_proc = subprocess.Popen(command, shell=False, preexec_fn=os.setsid)
        time.sleep(1.5)
        os.system('sh autostart.sh {}'.format(window_title))
        time.sleep(0.5)

    def __init__(self, vision=False, throttle=False, gear_change=False, port=3101):
       #print("Init")
        self.vision = vision
        self.throttle = throttle
        self.gear_change = gear_change
        self.port = port
        self.torcs_proc = None

        self.initial_run = True

        ##print("launch torcs")
        time.sleep(0.5)
        self.start_torcs_process()
        time.sleep(0.5)

        """
        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=3101, vision=self.vision)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        """
        if throttle is False:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,))
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,))

        if vision is False:
            high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf])
            low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf])
            self.observation_space = spaces.Box(low=low, high=high)
        else:
            high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf, 255])
            low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf, 0])
            self.observation_space = spaces.Box(low=low, high=high)

    def step(self, u):
       #print("Step")
        # convert thisAction to the actual torcs actionstr
        client = self.client

        this_action = self.agent_to_torcs(u)

        # Apply Action
        action_torcs = client.R.d

        # Steering
        action_torcs['steer'] = this_action['steer']  # in [-1, 1]

        #  Simple Autnmatic Throttle Control by Snakeoil
        if self.throttle is False:
            target_speed = self.default_speed
            if client.S.d['speedX'] < target_speed - (client.R.d['steer']*50):
                client.R.d['accel'] += .01
            else:
                client.R.d['accel'] -= .01

            if client.R.d['accel'] > 0.2:
                client.R.d['accel'] = 0.2

            if client.S.d['speedX'] < 10:
                client.R.d['accel'] += 1/(client.S.d['speedX']+.1)

            # Traction Control System
            if ((client.S.d['wheelSpinVel'][2]+client.S.d['wheelSpinVel'][3]) -
               (client.S.d['wheelSpinVel'][0]+client.S.d['wheelSpinVel'][1]) > 5):
                action_torcs['accel'] -= .2
        else:
            action_torcs['accel'] = this_action['accel']
            #action_torcs['brake'] = this_action['brake']

        #  Automatic Gear Change by Snakeoil
        if self.gear_change is True:
            action_torcs['gear'] = this_action['gear']
        else:
            #  Automatic Gear Change by Snakeoil is possible
            action_torcs['gear'] = 1
            if self.throttle:
                if client.S.d['speedX'] > 50:
                    action_torcs['gear'] = 2
                if client.S.d['speedX'] > 80:
                    action_torcs['gear'] = 3
                if client.S.d['speedX'] > 110:
                    action_torcs['gear'] = 4
                if client.S.d['speedX'] > 140:
                    action_torcs['gear'] = 5
                if client.S.d['speedX'] > 170:
                    action_torcs['gear'] = 6
        # Save the privious full-obs from torcs for the reward calculation
        obs_pre = copy.deepcopy(client.S.d)

        # One-Step Dynamics Update #################################
        # Apply the Agent's action into torcs
        client.respond_to_server()
        # Get the response of TORCS
        client.get_servers_input()

        # Get the current full-observation from torcs
        obs = client.S.d

        # Make an obsevation from a raw observation vector from TORCS
        self.observation = self.make_observaton(obs)

        # Reward setting Here #######################################
        # direction-dependent positive reward
        track = np.array(obs['track'])

        trackPos = np.array(obs['trackPos'])
        sp = np.array(obs['speedX'])
        damage = np.array(obs['damage'])
        rpm = np.array(obs['rpm'])

        progress = sp*np.cos(obs['angle']) #- np.abs(sp*np.sin(obs['angle'])) - sp * np.abs(obs['trackPos'])
        reward = progress

        if self.time_step % 5 == 0 :
            print(progress) 

        # collision detection
        if obs['damage'] - obs_pre['damage'] > 0:
            reward = -1

        # Termination judgement #########################
        episode_terminate = False
        #if (abs(track.any()) > 1 or abs(trackPos) > 1):  # Episode is terminated if the car is out of track
        #    reward = -200
        #    episode_terminate = True
        #    client.R.d['meta'] = True

        #if self.terminal_judge_start < self.time_step: # Episode terminates if the progress of agent is small
        #    if progress < self.termination_limit_progress:
        #        print("No progress")
        #        episode_terminate = True
        #        client.R.d['meta'] = True

        if self.terminal_judge_start < self.time_step: # Episode terminates if the progress of agent is small
           if progress < self.termination_limit_progress:
               #if self.time_step >  20 :
                print("--- No progress restart : reward: {},x:{},angle:{},trackPos:{}".format(progress,sp,'nouse','nouse'))
                print(self.time_step)
                episode_terminate = True
                client.R.d['meta'] = True

        if np.cos(obs['angle']) < 0:  # Episode is terminated if the agent runs backward
            if self.time_step >  20 :
                print("--- backward restart : reward: {},x:{},angle:{},trackPos:{}".format( progress, sp, obs['angle'], obs['trackPos']))
                print(self.time_step)
                episode_terminate = True
                client.R.d['meta'] = True



        # if np.cos(obs['angle']) < 0: # Episode is terminated if the agent runs backward
        #     episode_terminate = True
        #     client.R.d['meta'] = True


        if client.R.d['meta'] is True: # Send a reset signal
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1

        return self.get_obs(), reward, client.R.d['meta'], {}

    def reset(self, relaunch=False):
        print("Reset")

        self.time_step = 0

        if self.initial_reset is not True:
            self.client.R.d['meta'] = True
            self.client.respond_to_server()

            ## TENTATIVE. Restarting TORCS every episode suffers the memory leak bug!
            if relaunch is True:
                self.reset_torcs()
                print("### TORCS is RELAUNCHED ###")

        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(self.start_torcs_process, p=self.port)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        self.observation = self.make_observaton(obs)

        self.last_u = None

        self.initial_reset = False

        ob=self.get_obs()
        states= np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

        return states

    def end(self):
        os.killpg(os.getpgid(self.torcs_proc.pid), signal.SIGKILL)
        time.sleep(0.5)
        os.killpg(os.getpgid(self.torcs_proc.pid), signal.SIGKILL)

    def get_obs(self):
        return self.observation

    def reset_torcs(self):
       #print("relaunch torcs")
        self.torcs_proc.terminate()
        time.sleep(0.5)
        self.start_torcs_process()
        time.sleep(0.5)

    # def agent_to_torcs(self, u):
    #     torcs_action = {'steer': u[0]}

    #     if self.throttle is True:  # throttle action is enabled
    #         torcs_action.update({'accel': u[1]})
    #         #torcs_action.update({'brake': u[2]})

    #     if self.gear_change is True: # gear change action is enabled
    #         torcs_action.update({'gear': int(u[3])})

    #     return torcs_action

    def agent_to_torcs(self, u):
        accel = 0
        brake = 0

        #if len(u)==2:
        #    if u[1] >= 0:
        #        accel = u[1]
        #    else:
        #        brake = u[1]
        if len(u)==3:
            accel = np.abs(u[1])
            brake = np.abs(u[2])

        torcs_action = {'steer': u[0], 'accel': accel, 'brake': brake}
                
        if self.time_step % 5 == 0 :
            print('------------------------------------------------------------')
            print(self.time_step)
            print(torcs_action) 
            

        return torcs_action


    def obs_vision_to_image_rgb(self, obs_image_vec):
        image_vec =  obs_image_vec
        r = image_vec[0:len(image_vec):3]
        g = image_vec[1:len(image_vec):3]
        b = image_vec[2:len(image_vec):3]

        sz = (64, 64)
        r = np.array(r).reshape(sz)
        g = np.array(g).reshape(sz)
        b = np.array(b).reshape(sz)
        return np.array([r, g, b], dtype=np.uint8)

    def make_observaton(self, raw_obs):
        if self.vision is False:
            names = ['focus',
                     'speedX', 'speedY', 'speedZ', 'angle', 'damage',
                     'opponents',
                     'rpm',
                     'track', 
                     'trackPos',
                     'wheelSpinVel']
            Observation = col.namedtuple('Observaion', names)
            return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                               speedX=np.array(raw_obs['speedX'], dtype=np.float32)/300.0,
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32)/300.0,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/300.0,
                               angle=np.array(raw_obs['angle'], dtype=np.float32)/3.1416,
                               damage=np.array(raw_obs['damage'], dtype=np.float32),
                               opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32)/10000,
                               track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                               trackPos=np.array(raw_obs['trackPos'], dtype=np.float32)/1.,
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32))


        else:
            names = ['focus',
                     'speedX', 'speedY', 'speedZ', 'angle',
                     'opponents',
                     'rpm',
                     'track',
                     'trackPos',
                     'wheelSpinVel',
                     'img']
            Observation = col.namedtuple('Observaion', names)

            # Get RGB from observation
            image_rgb = self.obs_vision_to_image_rgb(raw_obs[names[8]])

            return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                               speedX=np.array(raw_obs['speedX'], dtype=np.float32)/self.default_speed,
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32)/self.default_speed,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/self.default_speed,
                               opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32),
                               track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                               trackPos=np.array(raw_obs['trackPos'], dtype=np.float32)/1.,
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
                               img=image_rgb)



class GymTorcsEnv(TorcsEnv):
    def __init__(self, vision=False, throttle=False, gear_change=False, port=3101):
        super(GymTorcsEnv,self).__init__(vision,throttle,gear_change,port )
        #self.states={'num_actions':29,'type':'float','shape':(29,)}
        #self.actions={ 'continuous':True,'shape':(3,),'type':'float'}

    def execute(self,action):
        _,reward,_,_=self.step(action)
        ob=self.get_obs()
        states= np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
        return  states, reward, self.client.R.d['meta']

    @property
    def states(self):
        return {'type':'float','shape':(29,)}

    # @property
    # def actions(self):
    #     return OpenAIGym.action_from_space(space=self.action_space)


    @property
    def actions(self):
        return GymTorcsEnv.action_from_space(space=self.action_space)

    @staticmethod
    def action_from_space(space):
        if isinstance(space, gym.spaces.Discrete):
            return dict(continuous=False, num_actions=space.n)
        elif isinstance(space, gym.spaces.MultiBinary):
            return dict(continuous=False, num_actions=2, shape=space.n)
        elif isinstance(space, gym.spaces.MultiDiscrete):
            if (space.low == space.low[0]).all() and (space.high == space.high[0]).all():
                return dict(continuous=False, num_actions=(space.high[0] - space.low[0]), shape=space.num_discrete_space)
            else:
                actions = dict()
                for n in range(space.num_discrete_space):
                    actions['action{}'.format(n)] = dict(continuous=False, num_actions=(space.high[n] - space.low[n]))
                return actions
        elif isinstance(space, gym.spaces.Box):
            if (space.low == space.low[0]).all() and (space.high == space.high[0]).all():
                return dict(continuous=True, shape=space.low.shape, min_value=space.low[0], max_value=space.high[0])
            else:
                actions = dict()
                low = space.low.flatten()
                high = space.high.flatten()
                for n in range(low.shape[0]):
                    actions['action{}'.format(n)] = dict(continuous=True, min_value=low[n], max_value=high[n])
                return actions
        elif isinstance(space, gym.spaces.Tuple):
            actions = dict()
            n = 0
            for space in space.spaces:
                action = OpenAIGym.action_from_space(space=space)
                if 'continuous' in action:
                    actions['action{}'.format(n)] = action
                    n += 1
                else:
                    for action in action.values():
                        actions['action{}'.format(n)] = action
                        n += 1
            return actions
        else:
            raise TensorForceError('Unknown Gym space.')
