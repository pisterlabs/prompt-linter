import gym
from gym import wrappers
from openai_ros.msg import RLExperimentInfo
import rospy

class GymRunner:
    def __init__(self, env_id, monitor_dir, max_timesteps=100000):
        self.monitor_dir = monitor_dir
        self.max_timesteps = max_timesteps

        self.env = gym.make(env_id)
        self.env = wrappers.Monitor(self.env, monitor_dir, force=True)
        
        self.pub_reward_obj = PublishRewardClass()

    def calc_reward(self, state, action, gym_reward, next_state, done):
        return gym_reward

    def train(self, agent, num_episodes, do_train=True, do_render=True, publish_reward=True):
        self.run(agent, num_episodes, do_train, do_render, publish_reward)

    def run(self, agent, num_episodes, do_train=False, do_render=True, publish_reward=True):
        

        for episode in range(num_episodes):
            
            state = self.env.reset().reshape(1, self.env.observation_space.shape[0])
            
            total_reward = 0
            
            for t in range(self.max_timesteps):
                
                
                if do_render:
                    self.env.render()
                
                action = agent.select_action(state, do_train)

                # execute the selected action
                next_state, reward, done, _ = self.env.step(action)
                next_state = next_state.reshape(1, self.env.observation_space.shape[0])
                reward = self.calc_reward(state, action, reward, next_state, done)
                # Cumulate reward
                self.pub_reward_obj.update_cumulated_reward(reward)

                # record the results of the step
                if do_train:
                    agent.record(state, action, reward, next_state, done)

                total_reward += reward
                state = next_state
                if done:
                    if publish_reward:
                        self.pub_reward_obj._update_episode()
                    break
                else:
                    pass
                

            # train the agent based on a sample of past experiences
            if do_train:
                agent.replay()

            print("episode: {}/{} | score: {} | e: {:.3f}".format(
                episode + 1, num_episodes, total_reward, agent.epsilon))

    def close_and_upload(self, api_key):
        self.env.close()
        gym.upload(self.monitor_dir, api_key=api_key)
        
        
class PublishRewardClass(object):
    def __init__(self):
        # Set up ROS related variables
        self.episode_num = 0
        self.cumulated_episode_reward = 0
        # Start Reward publishing
        self.reward_pub = rospy.Publisher('/openai/reward', RLExperimentInfo, queue_size=1)
        
    def _publish_reward_topic(self, reward, episode_number=1):
        """
        This function publishes the given reward in the reward topic for
        easy access from ROS infrastructure.
        :param reward:
        :param episode_number:
        :return:
        """
        reward_msg = RLExperimentInfo()
        reward_msg.episode_number = episode_number
        reward_msg.episode_reward = reward
        self.reward_pub.publish(reward_msg)
        
    def _update_episode(self):
        """
        Publishes the cumulated reward of the episode and 
        increases the episode number by one.
        :return:
        """
        self._publish_reward_topic(
                                    self.cumulated_episode_reward,
                                    self.episode_num
                                    )
        self.episode_num += 1
        self.cumulated_episode_reward = 0
        
    def update_cumulated_reward(self, reward):
        """
        Increase reward
        """
        self.cumulated_episode_reward += reward