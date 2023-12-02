import gym
import gym_gazebo
import jaco_gym
import time
import numpy
import random
import time
import csv

from tensorforce.agents import Agent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym
# from tensorforce.agents.ppo_agent import PPOAgent
from tensorforce.agents.trpo_agent import TRPOAgent


network_spec = [
	dict(type='dense', size=32, activation='relu'),
	dict(type='dense', size=32, activation='relu')
]

train_data = []

def main():
	#tensorforce
	env = OpenAIGym('JacoArm-v0')


	agent = TRPOAgent(
		states_spec=env.states,
		actions_spec=env.actions,
		network_spec=network_spec,
		batch_size=512
	)

	# agent = PPOAgent(
	# 	states_spec=env.states,
	# 	actions_spec=env.actions,
	# 	network_spec=network_spec,
	# 	batch_size=512,
	# 	step_optimizer=dict(
	# 		type='adam',
	# 		learning_rate=1e-4
	# 	)
	# )

	runner = Runner(agent=agent, environment=env)

	raw_input("hit enter when gazebo is loaded...")
	print()
	env.gym.unpause()
	env.gym.hold_init_robot_pos([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
	runner.run(episodes=1500, max_episode_timesteps=1000, episode_finished=episode_finished)

	#old-fashioned way
	# env = gym.make('JacoArm-v0')
	# print "launching the world..."
	# #gz loaing issues, let user start the learning
	# raw_input("hit enter when gazebo is loaded...")
	# env.set_physics_update(0.0001, 10000)
	# raw_input("hit enter when gazebo is loaded...")

	# # env.set_goal([0.167840578046, 0.297489331432, 0.857454500127])

	# total_episodes = 100
	# action = [1,1,1,1,1,1,1,1,1,1]
	# x = 0
	# # for x in range(total_episodes):
	# while True:
	# 	# if x % 10 is 0:
	# 	action = numpy.random.rand(1, 10)[0]
	# 		# print 'new action is', action
		
	# 	state, reward, done, _ = env.step(action)
	# 	print reward
	# 	time.sleep(0.2)
	# 	x += 1

	write_to_csv(train_data, 'test.csv')
	env.close()

def episode_finished(r):
	# print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep, reward=r.episode_rewards[-1]))
	print("{ep}, {ts}, {reward}".format(ep=r.episode, ts=r.episode_timestep, reward=r.episode_rewards[-1]))
	train_data.append([r.episode_timestep, r.episode_rewards[-1]])
	return True

def write_to_csv(data, fn):
	with open(fn, 'wb') as f:
		w = csv.writer(f, dialect='excel')
		for row in data:
			w.writerow(row)
	f.close()

if __name__ == '__main__':
	main()