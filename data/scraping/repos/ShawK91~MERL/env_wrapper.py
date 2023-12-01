import numpy as np, sys

class RoverDomainPython:
	"""Wrapper around the Environment to expose a cleaner interface for RL

		Parameters:
			env_name (str): Env name


	"""
	def __init__(self, args, num_envs):
		"""
		A base template for all environment wrappers.
		"""
		#Initialize world with requiste params
		self.args = args

		from envs.rover_domain.rover_domain_python import RoverDomainVel

		self.universe = [] #Universe - collection of all envs running in parallel
		for _ in range(num_envs):
			env = RoverDomainVel(args.config)
			self.universe.append(env)

		#Action Space
		self.action_low = -1.0
		self.action_high = 1.0


	def reset(self):
		"""Method overloads reset
			Parameters:
				None

			Returns:
				next_obs (list): Next state
		"""
		joint_obs = []
		for env in self.universe:
			obs = env.reset()
			joint_obs.append(obs)

		joint_obs = np.stack(joint_obs, axis=1)
		#returns [agent_id, universe_id, obs]

		return joint_obs


	def step(self, action): #Expects a numpy action
		"""Take an action to forward the simulation

			Parameters:
				action (ndarray): action to take in the env

			Returns:
				next_obs (list): Next state
				reward (float): Reward for this step
				done (bool): Simulation done?
				info (None): Template from OpenAi gym (doesnt have anything)
		"""

		joint_obs = []; joint_reward = []; joint_done = []; joint_global = []
		for universe_id, env in enumerate(self.universe):
			next_state, reward, done, info = env.step(action[:,universe_id,:])
			joint_obs.append(next_state); joint_reward.append(reward); joint_done.append(done); joint_global.append(info)

		joint_obs = np.stack(joint_obs, axis=1)
		joint_reward = np.stack(joint_reward, axis=1)

		return joint_obs, joint_reward, joint_done, joint_global



	def render(self):

		rand_univ = np.random.randint(0, len(self.universe))
		try: self.universe[rand_univ].render()
		except: 'Error rendering'


class MotivateDomain:
	"""Wrapper around the Environment to expose a cleaner interface for RL

		Parameters:
			env_name (str): Env name


	"""
	def __init__(self, args, num_envs):
		"""
		A base template for all environment wrappers.
		"""
		#Initialize world with requiste params
		self.args = args

		from envs.rover_domain.motivate_domain import MotivateDomain

		self.universe = [] #Universe - collection of all envs running in parallel
		for _ in range(num_envs):
			env = MotivateDomain(args.config)
			self.universe.append(env)

		#Action Space
		self.action_low = -1.0
		self.action_high = 1.0


	def reset(self):
		"""Method overloads reset
			Parameters:
				None

			Returns:
				next_obs (list): Next state
		"""
		joint_obs = []
		for env in self.universe:
			obs = env.reset()
			joint_obs.append(obs)

		joint_obs = np.stack(joint_obs, axis=1)
		#returns [agent_id, universe_id, obs]

		return joint_obs


	def step(self, action): #Expects a numpy action
		"""Take an action to forward the simulation

			Parameters:
				action (ndarray): action to take in the env

			Returns:
				next_obs (list): Next state
				reward (float): Reward for this step
				done (bool): Simulation done?
				info (None): Template from OpenAi gym (doesnt have anything)
		"""

		joint_obs = []; joint_reward = []; joint_done = []; joint_global = []
		for universe_id, env in enumerate(self.universe):
			next_state, reward, done, info = env.step(action[:,universe_id,:])
			joint_obs.append(next_state); joint_reward.append(reward); joint_done.append(done); joint_global.append(info)

		joint_obs = np.stack(joint_obs, axis=1)
		joint_reward = np.stack(joint_reward, axis=1)

		return joint_obs, joint_reward, joint_done, joint_global



	def render(self):

		rand_univ = np.random.randint(0, len(self.universe))
		self.universe[rand_univ].render()
		print(self.universe[rand_univ].poi_visitor_list)


class MultiWalker:
	"""Wrapper around the Environment to expose a cleaner interface for RL

		Parameters:
			env_name (str): Env name


	"""
	def __init__(self, args, num_envs=1):
		"""
		A base template for all environment wrappers.
		"""
		#Initialize world with requiste params
		self.args = args
		self.num_envs = num_envs

		from envs.madrl.walker.multi_walker import MultiWalkerEnv

		self.universe = [] #Universe - collection of all envs running in parallel
		for _ in range(num_envs):
			env = MultiWalkerEnv(n_walkers=args.config.num_agents, position_noise=1e-3, angle_noise=1e-3, reward_mech='local',
                 forward_reward=1.0, fall_reward=-100.0, drop_reward=-100.0, terminate_on_fall=True,
                 one_hot=False)
			self.universe.append(env)

		#Action Space
		self.action_low = -1.0
		self.action_high = 1.0
		self.action_dim = 4
		self.state_dim = 33

		self.global_reward = [0.0 for _ in range(num_envs)]
		self.env_dones = [False for _ in range(num_envs)]


	def reset(self):
		"""Method overloads reset
			Parameters:
				None

			Returns:
				next_obs (list): Next state
		"""
		#Reset Global Reward and dones
		self.global_reward = [0.0 for _ in range(self.num_envs)]
		self.env_dones = [False for _ in range(self.num_envs)]

		#Get joint observation
		joint_obs = []
		for env in self.universe:
			obs = env.reset()
			joint_obs.append(obs)

		joint_obs = np.stack(joint_obs, axis=1)


		return joint_obs


	def step(self, action): #Expects a numpy action
		"""Take an action to forward the simulation

			Parameters:
				action (ndarray): action to take in the env

			Returns:
				next_obs (list): Next state
				reward (float): Reward for this step
				done (bool): Simulation done?
				info (None): Template from OpenAi gym (doesnt have anything)
		"""

		joint_obs = []; joint_reward = []; joint_done = []; joint_global = []
		for universe_id, env in enumerate(self.universe):

			#If this particular env instance in universe is already done:
			if self.env_dones[universe_id]:
				joint_obs.append(env.dummy_state()); joint_reward.append(env.dummy_reward()); joint_done.append(True); joint_global.append(None)

			else:
				next_state, reward, done, _ = env.step(action[:,universe_id,:])
				joint_obs.append(next_state); joint_reward.append(reward); joint_done.append(done)

				self.global_reward[universe_id] += sum(reward)/self.args.config.num_agents
				if done:
					joint_global.append(self.global_reward[universe_id])
					self.env_dones[universe_id] = True
				else: joint_global.append(None)


		joint_obs = np.stack(joint_obs, axis=1)
		joint_reward = np.stack(joint_reward, axis=1)

		return joint_obs, joint_reward, joint_done, joint_global



	def render(self):
		pass
		# rand_univ = np.random.randint(0, len(self.universe))
		# self.universe[rand_univ].render()


class Pursuit:
	"""Wrapper around the Environment to expose a cleaner interface for RL

		Parameters:
			env_name (str): Env name


	"""
	def __init__(self, args, num_envs=1):
		"""
		A base template for all environment wrappers.
		"""
		#Initialize world with requiste params
		self.args = args
		self.num_envs = num_envs

		from envs.pursuit import MAWaterWorld_mod

		self.universe = [] #Universe - collection of all envs running in parallel
		for _ in range(num_envs):
			env = MAWaterWorld_mod(n_pursuers=args.config.num_agents, n_evaders=50,
                         n_poison=50, obstacle_radius=0.04,
                         food_reward=10,
                         poison_reward=-1.0,
                         encounter_reward=0.01,
                         n_coop=args.config.coupling,
                         sensor_range=0.2, obstacle_loc=None, )
			self.universe.append(env)

		#Action Space
		self.action_low = -1.0
		self.action_high = 1.0
		self.action_dim = 4
		self.state_dim = 33

		self.global_reward = [0.0 for _ in range(num_envs)]
		self.env_dones = [False for _ in range(num_envs)]


	def reset(self):
		"""Method overloads reset
			Parameters:
				None

			Returns:
				next_obs (list): Next state
		"""
		#Reset Global Reward and dones
		self.global_reward = [0.0 for _ in range(self.num_envs)]
		self.env_dones = [False for _ in range(self.num_envs)]

		#Get joint observation
		joint_obs = []
		for env in self.universe:
			obs = env.reset()
			joint_obs.append(obs)

		joint_obs = np.stack(joint_obs, axis=1)


		return joint_obs


	def step(self, action): #Expects a numpy action
		"""Take an action to forward the simulation

			Parameters:
				action (ndarray): action to take in the env

			Returns:
				next_obs (list): Next state
				reward (float): Reward for this step
				done (bool): Simulation done?
				info (None): Template from OpenAi gym (doesnt have anything)
		"""

		joint_obs = []; joint_reward = []; joint_done = []; joint_global = []
		for universe_id, env in enumerate(self.universe):

			#If this particular env instance in universe is already done:
			if self.env_dones[universe_id]:
				joint_obs.append(env.dummy_state()); joint_reward.append(env.dummy_reward()); joint_done.append(True); joint_global.append(None)

			else:
				next_state, reward, done, global_reward = env.step(action[:,universe_id,:])
				joint_obs.append(next_state); joint_reward.append(reward); joint_done.append(done)

				self.global_reward[universe_id] += global_reward
				if done:
					joint_global.append(self.global_reward[universe_id])
					self.env_dones[universe_id] = True
				else: joint_global.append(None)


		joint_obs = np.stack(joint_obs, axis=1)
		joint_reward = np.stack(joint_reward, axis=1)

		return joint_obs, joint_reward, joint_done, joint_global



	def render(self):
		pass
		# rand_univ = np.random.randint(0, len(self.universe))
		# self.universe[rand_univ].render()


class PowerPlant:
	"""Wrapper around the Environment to expose a cleaner interface for RL

		Parameters:
			env_name (str): Env name


	"""
	def __init__(self, args, num_envs=1):
		"""
		A base template for all environment wrappers.
		"""
		#Initialize world with requiste params
		self.args = args
		self.num_envs = num_envs

		from envs.hyper.PowerPlant_env import PowerPlant, Fast_Simulator

		self.universe = [] #Universe - collection of all envs running in parallel
		for _ in range(num_envs):
			env = PowerPlant(args.config.target_sensor, args.config.run_time, args.config.sensor_noise, args.config.reconf_shape, args.config.num_profiles)
			self.universe.append(env)

		#Action Space
		self.action_low = -1.0
		self.action_high = 1.0
		self.action_dim = 2
		self.state_dim = 20

		self.global_reward = [0.0 for _ in range(num_envs)]
		self.env_dones = [False for _ in range(num_envs)]


	def reset(self):
		"""Method overloads reset
			Parameters:
				None

			Returns:
				next_obs (list): Next state
		"""
		#Reset Global Reward and dones
		self.global_reward = [0.0 for _ in range(self.num_envs)]
		self.env_dones = [False for _ in range(self.num_envs)]

		#Get joint observation
		joint_obs = []
		for env in self.universe:
			obs = env.reset()
			joint_obs.append(np.transpose(obs))

		joint_obs = np.stack(joint_obs, axis=1)


		return joint_obs


	def step(self, action): #Expects a numpy action
		"""Take an action to forward the simulation

			Parameters:
				action (ndarray): action to take in the env

			Returns:
				next_obs (list): Next state
				reward (float): Reward for this step
				done (bool): Simulation done?
				info (None): Template from OpenAi gym (doesnt have anything)
		"""

		joint_obs = []; joint_reward = []; joint_done = []; joint_global = []
		for universe_id, env in enumerate(self.universe):

			#If this particular env instance in universe is already done:
			if self.env_dones[universe_id]:
				joint_obs.append(env.dummy_state()); joint_reward.append(env.dummy_reward()); joint_done.append(True); joint_global.append(None)

			else:
				next_state, reward, done, _ = env.step(action[:,universe_id,:][0])
				joint_obs.append(np.transpose(next_state)); joint_reward.append([reward]); joint_done.append(done)

				self.global_reward[universe_id] += reward
				if done:
					joint_global.append(self.global_reward[universe_id])
					self.env_dones[universe_id] = True
				else: joint_global.append(None)


		joint_obs = np.stack(joint_obs, axis=1)
		joint_reward = np.stack(joint_reward, axis=1)

		return joint_obs, joint_reward, joint_done, joint_global



	def render(self):
		pass
		# rand_univ = np.random.randint(0, len(self.universe))
		# self.universe[rand_univ].render()


class Cassie:
	"""Wrapper around the Environment to expose a cleaner interface for RL

		Parameters:
			env_name (str): Env name


	"""
	def __init__(self, args, num_envs=1, viz=False):
		"""
		A base template for all environment wrappers.
		"""
		#Initialize world with requiste params
		self.args = args
		self.num_envs = num_envs

		from envs.cassie.cassie_env.cassieRLEnv import cassieRLEnv, cassieRLEnvSparseReward, cassieRlEnvAdaptive

		self.universe = [] #Universe - collection of all envs running in parallel
		for _ in range(num_envs):
			if args.config.config == 'sparse': env = cassieRLEnvSparseReward(viz)
			elif args.config.config == 'dense': env = cassieRLEnv(viz)
			elif args.config.config == 'adaptive': env = cassieRlEnvAdaptive(viz)
			else: sys.exit('Incorrect config given for Cassie')
			self.universe.append(env)

		#Action Space
		self.action_low = -1.0
		self.action_high = 1.0
		self.action_dim = 10
		self.state_dim = 80

		self.global_reward = [0.0 for _ in range(num_envs)]
		self.env_dones = [False for _ in range(num_envs)]


	def reset(self):
		"""Method overloads reset
			Parameters:
				None

			Returns:
				next_obs (list): Next state
		"""
		#Reset Global Reward and dones
		self.global_reward = [0.0 for _ in range(self.num_envs)]
		self.env_dones = [False for _ in range(self.num_envs)]

		#Get joint observation
		joint_obs = []
		for env in self.universe:
			obs = env.reset()
			joint_obs.append(np.array([obs]))

		joint_obs = np.stack(joint_obs, axis=1)


		return joint_obs


	def step(self, action): #Expects a numpy action
		"""Take an action to forward the simulation

			Parameters:
				action (ndarray): action to take in the env

			Returns:
				next_obs (list): Next state
				reward (float): Reward for this step
				done (bool): Simulation done?
				info (None): Template from OpenAi gym (doesnt have anything)
		"""

		joint_obs = []; joint_reward = []; joint_done = []; joint_global = []
		for universe_id, env in enumerate(self.universe):

			#If this particular env instance in universe is already done:
			if self.env_dones[universe_id]:
				joint_obs.append([env.dummy_state()]); joint_reward.append([env.dummy_reward()]); joint_done.append(True); joint_global.append(None)

			else:
				next_state, reward, done, _ = env.step(action[:,universe_id,:].flatten())
				joint_obs.append([next_state]); joint_reward.append([reward]); joint_done.append(done)

				self.global_reward[universe_id] += reward
				if done:
					joint_global.append(self.global_reward[universe_id])
					self.env_dones[universe_id] = True
				else: joint_global.append(None)


		joint_obs = np.stack(joint_obs, axis=1)
		joint_reward = np.stack(joint_reward, axis=1)

		return joint_obs, joint_reward, joint_done, joint_global



	def render(self):
		self.universe[0].vis.draw(self.universe[0].sim)
		# rand_univ = np.random.randint(0, len(self.universe))
		# self.universe[rand_univ].render()





class RoverDomainCython:
	"""Wrapper around the Environment to expose a cleaner interface for RL

		Parameters:
			env_name (str): Env name


	"""
	def __init__(self, args, num_envs):
		"""
		A base template for all environment wrappers.
		"""
		#Initialize world with requiste params
		self.args = args

		from envs.rover_domain.rover_domain_cython import rover_domain_w_setup as r

		self.universe = [] #Universe - collection of all envs running in parallel
		for _ in range(num_envs):
			env = r.RoverDomain()
			env.n_rovers = args.num_agents
			env.n_pois = args.num_poi
			env.interaction_dist = args.act_dist
			env.n_obs_sections = int(360/args.angle_res)
			env.n_req = args.coupling
			env.n_steps = args.ep_len
			env.setup_size = args.dim_x
			self.universe.append(env)


		#Action Space
		self.action_low = -1.0
		self.action_high = 1.0


	def reset(self):
		"""Method overloads reset
			Parameters:
				None

			Returns:
				next_obs (list): Next state
		"""
		joint_obs = []
		for env in self.universe:
			env.reset()
			next_state = env.rover_observations.base
			next_state = next_state.reshape(next_state.shape[0], -1)
			joint_obs.append(next_state)

		next_state = np.stack(joint_obs, axis=1)
		#returns [agent_id, universe_id, obs]

		return next_state


	def step(self, action): #Expects a numpy action
		"""Take an action to forward the simulation

			Parameters:
				action (ndarray): action to take in the env

			Returns:
				next_obs (list): Next state
				reward (float): Reward for this step
				done (bool): Simulation done?
				info (None): Template from OpenAi gym (doesnt have anything)
		"""


		#action = self.action_low + action * (self.action_high - self.action_low)
		#action = [ac*0 for ac in action]


		joint_obs = []; joint_reward = []; joint_done = []
		for universe_id, env in enumerate(self.universe):
			next_state, reward, done, info = env.step(action[:,universe_id,:])
			next_state = next_state.base; reward = reward.base
			next_state = next_state.reshape(next_state.shape[0], -1)
			joint_obs.append(next_state); joint_reward.append(reward); joint_done.append(done)

		joint_obs = np.stack(joint_obs, axis=1)
		joint_reward = np.stack(joint_reward, axis=1)



		#print(self.env.rover_positions.base, self.env.poi_positions.base, action, reward)
		# import numpy as np
		# if np.sum(reward) != 0:
		# 	k = 0

		#print(self.env.rover_positions.base, self.env.poi_positions.base, reward)
		##None

		return joint_obs, joint_reward, joint_done, None

	def render(self):

		# Visualize
		grid = [['-' for _ in range(self.args.dim_x)] for _ in range(self.args.dim_y)]

		rand_univ = np.random.randint(0, len(self.universe))

		# Draw in rover path
		for time_step, joint_pos in enumerate(self.universe[rand_univ].rover_position_histories.base):
			for rover_id, rover_pos in enumerate(joint_pos):
				x = int(rover_pos[0]);
				y = int(rover_pos[1])
				# print x,y
				try: grid[x][y] = str(rover_id)
				except: None

		# Draw in food
		for poi_pos, poi_status in zip(self.universe[rand_univ].poi_positions.base, self.universe[rand_univ].poi_status.base):
			x = int(poi_pos[0]);
			y = int(poi_pos[1])
			marker = '#' if poi_status else '$'
			grid[x][y] = marker

		for row in grid:
			print(row)
		print()

		print('------------------------------------------------------------------------')



class DM_Soccer:
	"""Wrapper around the Environment to expose a cleaner interface for RL
		Parameters:
			env_name (str): Env name
	"""
	def __init__(self, env_name, ALGO):
		"""
		A base template for all environment wrappers.
		"""
		from dm_control.locomotion import soccer as dm_soccer

		import gym
		self.env = dm_soccer.load(team_size=2, time_limit=10.)
		self.action_specs = env.action_spec()
		self.ALGO = ALGO




	def reset(self):
		"""Method overloads reset
			Parameters:
				None
			Returns:
				next_obs (list): Next state
		"""
		return self.env.reset()


	def step(self, action: object): #Expects a numpy action
		"""Take an action to forward the simulation
			Parameters:
				action (ndarray): action to take in the env
			Returns:
				next_obs (list): Next state
				reward (float): Reward for this step
				done (bool): Simulation done?
				info (None): Template from OpenAi gym (doesnt have anything)
		"""

		if self.ALGO == "SAC": action = (action + 1.0) / 2.0  # [-1, 1] => [0, 1]

		action = self.action_low + action * (self.action_high - self.action_low)
		return self.env.step(action)

	def render(self):
		self.env.render()


