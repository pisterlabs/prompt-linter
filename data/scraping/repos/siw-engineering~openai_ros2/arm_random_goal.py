import random
from collections import deque
from typing import Dict, Tuple
from enum import Enum, auto
import numpy
import forward_kinematics_py as fk
from openai_ros2.utils import ut_launch, ut_gazebo
from openai_ros2.robots import LobotArmSim
from ament_index_python.packages import get_package_share_directory
from gym.spaces import Box
import os
import math
import rclpy
import pickle


class ArmState(Enum):
    Reached = auto()
    InProgress = auto()
    ApproachJointLimits = auto()
    Collision = auto()
    Timeout = auto()
    Undefined = auto()


class LobotArmRandomGoal:
    def __init__(self, node: rclpy.node.Node, robot, max_time_step: int = 500, accepted_dist_to_bounds=0.001,
                 accepted_error=0.001, reach_target_bonus_reward=0.0, reach_bounds_penalty=0.0, contact_penalty=0.0, timeout_penalty=0.0,
                 episodes_per_goal=1, goal_buffer_size=20, goal_from_buffer_prob=0.0, num_adjacent_goals=0, is_validation=False,
                 random_goal_seed=None, random_goal_file=None, normalise_reward=False, continuous_run=False, reward_noise_mu=None,
                 reward_noise_sigma=None, reward_noise_decay=None, exp_rew_scaling=None):
        self.node = node
        self.robot = robot
        self._max_time_step = max_time_step
        self.accepted_dist_to_bounds = accepted_dist_to_bounds
        self.accepted_error = accepted_error
        self.reach_target_bonus_reward = reach_target_bonus_reward
        self.reach_bounds_penalty = reach_bounds_penalty
        self.contact_penalty = contact_penalty
        self.timeout_penalty = timeout_penalty
        self.episodes_per_goal = episodes_per_goal
        self.goal_buffer_size = goal_buffer_size
        self.goal_from_buffer_prob = goal_from_buffer_prob
        self.num_adjacent_goals = num_adjacent_goals
        self.is_validation = is_validation
        self.random_goal_seed = random_goal_seed
        self.random_goal_file = random_goal_file
        self.normalise_reward = normalise_reward
        self.continuous_run = continuous_run
        self.reward_noise_mu = reward_noise_mu
        self.reward_noise_sigma = reward_noise_sigma
        self.original_reward_noise_sigma = reward_noise_sigma
        self.reward_noise_decay = reward_noise_decay
        self.exp_rew_scaling = exp_rew_scaling
        print(f'-------------------------------Setting task parameters-------------------------------')
        print('max_time_step: %8d               # Maximum time step before stopping the episode' % self._max_time_step)
        print('accepted_dist_to_bounds: %8.7f    # Allowable distance to joint limits (radians)' % self.accepted_dist_to_bounds)
        print('accepted_error: %8.7f             # Allowable distance from target coordinates (metres)' % self.accepted_error)
        print('reach_target_bonus_reward: %8.7f # Bonus reward upon reaching target' % self.reach_target_bonus_reward)
        print('reach_bounds_penalty: %8.7f      # Reward penalty when reaching joint limit' % self.reach_bounds_penalty)
        print('contact_penalty: %8.7f           # Reward penalty for collision' % self.contact_penalty)
        print('timeout_penalty: %8.7f           # Reward penalty for collision' % self.timeout_penalty)
        print('episodes_per_goal: %8d           # Number of episodes before generating another random goal' % self.episodes_per_goal)
        print('goal_buffer_size: %8d            # Number goals to store in buffer to be reused later' % self.goal_buffer_size)
        print('goal_from_buffer_prob: %8.7f      # Probability of selecting a random goal from the goal buffer, value between 0 and 1' % self.goal_from_buffer_prob)
        print('num_adjacent_goals: %8d          # Number of nearby goals to be generated for each randomly generated goal ' % self.num_adjacent_goals)
        print(f'random_goal_seed: {str(self.random_goal_seed):8}            # Seed used to generate the random goals')
        print(f'random_goal_file: {self.random_goal_file}       # Path to the numpy save file containing the random goals')
        print('is_validation: %8r               # Whether this is a validation run, if true will print which points failed and how many reached' % self.is_validation)
        print('normalise_reward: %8r            # Perform reward normalisation, this happens before reward bonus and penalties' % self.normalise_reward)
        print('continuous_run: %8r              # Continuously run the simulation, even after it reaches the destination' % self.continuous_run)
        print(f'reward_noise_mu: {self.reward_noise_mu}            # Reward noise mean (reward noise follows gaussian distribution)')
        print(f'reward_noise_sigma: {self.reward_noise_sigma}         # Reward noise standard deviation, recommended 0.5')
        print(f'reward_noise_decay: {self.reward_noise_decay}            # Constant for exponential reward noise decay (recommended 0.31073, decays to 0.002 in 20 steps)')
        print(f'exp_rew_scaling: {self.exp_rew_scaling}            # Constant for exponential reward scaling (None by default, recommended 5.0, cumulative exp_reward = 29.48)')
        print(f'-------------------------------------------------------------------------------------')

        assert self.accepted_dist_to_bounds >= 0.0, 'Allowable distance to joint limits should be positive'
        assert self.accepted_error >= 0.0, 'Accepted error to end coordinates should be positive'
        assert self.reach_target_bonus_reward >= 0.0, 'Reach target bonus reward should be positive'
        assert self.contact_penalty >= 0.0, 'Contact penalty should be positive'
        assert self.timeout_penalty >= 0.0, 'Timeout penalty should be positive'
        assert self.reach_bounds_penalty >= 0.0, 'Reach bounds penalty should be positive'
        assert isinstance(self.episodes_per_goal, int), f'Episodes per goal should be an integer, current type: {type(self.episodes_per_goal)}'
        assert self.episodes_per_goal >= 1, 'Episodes per goal be greather than or equal to 1, i.e. episodes_per_goal >= 1'
        assert isinstance(self.goal_buffer_size, int), f'Goal buffer size should be an integer, current type: {type(self.goal_buffer_size)}'
        assert self.goal_buffer_size > 0, 'Goal buffer size should be greather than or equal to 1, i.e. episodes_per_goal >= 1'
        assert 0 <= self.goal_from_buffer_prob <= 1, 'Probability of selecting goal from buffer should be between 0 and 1'
        assert isinstance(self.num_adjacent_goals, int), f'Number of adjacent goals should be an integer, current type: {type(self.num_adjacent_goals)}'
        assert self.num_adjacent_goals >= 0, f'Number of adjacent goals should be positive, current value: {self.num_adjacent_goals}'
        if self.random_goal_seed is not None:
            assert isinstance(self.random_goal_seed, int), f'Random goal seed should be an integer, current type: {type(self.random_goal_seed)}'
        if self.random_goal_file is not None:
            assert os.path.exists(self.random_goal_file)
        if reward_noise_decay is not None:
            assert self.reward_noise_mu is not None and self.reward_noise_sigma is not None
            assert isinstance(self.reward_noise_mu, float) and isinstance(self.reward_noise_sigma, float)
        if exp_rew_scaling is not None:
            assert isinstance(self.exp_rew_scaling, float), f'Exponential reward scaling factor should be a float, current type: {type(self.exp_rew_scaling)}'

        self._max_time_step = max_time_step
        lobot_desc_share_path = get_package_share_directory('lobot_description')
        arm_urdf_path = os.path.join(lobot_desc_share_path, 'robots/arm_standalone.urdf')
        self._fk = fk.ForwardKinematics(arm_urdf_path)

        self.coords_buffer = deque(maxlen=self.goal_buffer_size)
        self.target_coords_ik, self.target_coords = self.__get_target_coords()
        target_x = self.target_coords[0]
        target_y = self.target_coords[1]
        target_z = self.target_coords[2]
        if isinstance(robot, LobotArmSim):  # Check if is gazebo
            # Spawn the target marker if it is gazebo
            print(f'Spawning to: {(target_x, target_y, target_z)}')
            spawn_success = ut_gazebo.create_marker(node, target_x, target_y, target_z, diameter=0.004)
        self.previous_coords = None
        self.initial_coords = None
        self.__reset_count: int = 0
        self.fail_points = []
        self.__reach_count: int = 0

    def __del__(self):
        print('Arm random goal destructor called')
        if self.is_validation:
            # If fail_points is not defined yet, do nothing
            if not hasattr(self, 'fail_points'):
                return

            def print_list(list_):
                for item in list_:
                    print(item)

            print('Saving failed goals to data.pkl')

            with open('failed_points.pkl', 'wb') as f:
                pickle.dump(self.fail_points, f, pickle.HIGHEST_PROTOCOL)
            print('Failed goals, target coords: ')
            print_list(enumerate([x for x, y in self.fail_points]))

            # We spawn gazebo markers here to exploit the fact that spinningup test_policy somehow doesn't close gazebo after it's done
            k = 100
            ut_gazebo.create_marker(self.node, 0.0, 0.0, 0.0, diameter=0.004)
            for x, y in self.fail_points:
                ut_gazebo.create_marker(self.node, x[0], x[1], x[2], diameter=0.001, id=k)
                k += 1

    def is_done(self, joint_states: numpy.ndarray, contact_count: int, observation_space: Box, time_step: int = -1) -> Tuple[bool, Dict]:
        failed, arm_state = self.__is_failed(joint_states, contact_count, observation_space, time_step)
        info_dict = {'arm_state': arm_state}
        if failed:
            self.fail_points.append((self.target_coords, self.target_coords_ik))
            if self.is_validation:
                print(f'Failed to reach {self.target_coords}')
            return True, info_dict

        current_coords = self.__get_coords(joint_states)
        # If time step still within limits, as long as any coordinate is out of acceptance range, we are not done
        for i in range(3):
            if abs(self.target_coords[i] - current_coords[i]) > self.accepted_error:
                info_dict['arm_state'] = ArmState.InProgress
                return False, info_dict
        # If all coordinates within acceptance range AND time step within limits, we are done
        info_dict['arm_state'] = ArmState.Reached
        info_dict['step_count'] = time_step
        print(f'Reached destination, target coords: {self.target_coords}, current coords: {current_coords}, time step: {time_step}')
        self.__reach_count += 1
        if self.is_validation:
            print(f'Reach count: {self.__reach_count}')

        if self.continuous_run:
            return False, info_dict
        else:
            return True, info_dict

    def compute_reward(self, joint_states: numpy.ndarray, arm_state: ArmState) -> Tuple[float, Dict]:
        assert len(joint_states) == 3, f'Expected 3 values for joint states, but got {len(joint_states)} values instead'
        coords_get_result = self.__get_coords(joint_states)
        assert len(coords_get_result) == 3, f'Expected 3 values after getting coordinates, but got {len(coords_get_result)} values instead'
        current_coords = coords_get_result

        assert arm_state != ArmState.Undefined, f'Arm state cannot be undefined, please check logic'

        # Give 0 reward on initial state
        # if numpy.array_equal(self.previous_coords, numpy.array([0.0, 0.0, 0.0])):
        #     # print('Initial state detected, giving 0 reward')
        #     reward = 0.0
        # else:
        reward = self.__calc_dist_change(self.previous_coords, current_coords)

        # normalise rewards
        mag_target = numpy.linalg.norm(self.initial_coords - self.target_coords)
        normalised_reward = reward / mag_target

        # Scale up normalised reward slightly such that the total reward is between 0 and 10 instead of between 0 and 1
        normalised_reward *= 10

        # Scale up reward so that it is not so small if not normalised
        normal_scaled_reward = reward * 100

        # Calculate current distance to goal (for information purposes only)
        dist = numpy.linalg.norm(current_coords - self.target_coords)

        reward_info = {'normalised_reward': normalised_reward,
                       'normal_reward': normal_scaled_reward,
                       'distance_to_goal': dist,
                       'target_coords': self.target_coords,
                       'current_coords': current_coords}

        if self.normalise_reward:
            reward = normalised_reward
        else:
            reward = normal_scaled_reward

        # Calculate exponential reward component
        if self.exp_rew_scaling is not None:
            exp_reward = self.__calc_exponential_reward(self.previous_coords, current_coords)
            reward_info['exp_reward'] = exp_reward
            reward += exp_reward
        else:
            reward_info['exp_reward'] = 0.0

        # Add reward noise
        if self.reward_noise_mu is not None and self.reward_noise_sigma is not None:
            rew_noise = numpy.random.normal(self.reward_noise_mu, self.reward_noise_sigma)
            if self.reward_noise_decay is not None:
                self.reward_noise_sigma = self.reward_noise_sigma * math.exp(-self.reward_noise_decay)
            reward_info['rew_noise'] = rew_noise
            reward += rew_noise
        else:
            reward_info['rew_noise'] = 0.0

        self.previous_coords = current_coords

        # Reward shaping logic

        # Check if it has reached target destination
        if arm_state == ArmState.Reached:
            # if reached target destination and is continuous run, we generate another set of coordinates
            # This has to be after the __calc_dist_change function because that uses self.target_coords to calculate
            if self.continuous_run:
                self.target_coords_ik, self.target_coords = self.__get_target_coords()
                print(f'Moving to [{self.target_coords[0]:.6f}, {self.target_coords[1]:.6f}, {self.target_coords[2]:.6f}], '
                      f'Given by joint values [{self.target_coords_ik[0]:.6f}, {self.target_coords_ik[1]:.6f}, {self.target_coords_ik[2]:.6f}]')
                ut_gazebo.create_marker(self.node, self.target_coords[0], self.target_coords[1], self.target_coords[2], diameter=0.004)
            reward += self.reach_target_bonus_reward

        # Check if it has approached any joint limits
        if arm_state == ArmState.ApproachJointLimits:
            reward -= self.reach_bounds_penalty

        # Check for collision
        if arm_state == ArmState.Collision:
            reward -= self.contact_penalty

        if arm_state == ArmState.Timeout:
            reward -= self.timeout_penalty

        return reward, reward_info

    def reset(self):
        self.reward_noise_sigma = self.original_reward_noise_sigma
        # Set initial coordinates
        initial_joint_values = numpy.array([0.0, 0.0, 0.0])
        res = self._fk.calculate('world', 'grip_end_point', initial_joint_values)
        self.previous_coords = numpy.array([res.translation.x, res.translation.y, res.translation.z])
        self.initial_coords = numpy.array([res.translation.x, res.translation.y, res.translation.z])
        self.__reset_count += 1
        if self.__reset_count % self.episodes_per_goal == 0:
            self.target_coords_ik, self.target_coords = self.__get_target_coords()
            print(f'Moving to [{self.target_coords[0]:.6f}, {self.target_coords[1]:.6f}, {self.target_coords[2]:.6f}], '
                  f'Given by joint values [{self.target_coords_ik[0]:.6f}, {self.target_coords_ik[1]:.6f}, {self.target_coords_ik[2]:.6f}]')
            if isinstance(self.robot, LobotArmSim):  # Check if is simulator or real
                # Move the target marker if it is gazebo
                spawn_success = ut_gazebo.create_marker(self.node, self.target_coords[0],
                                                        self.target_coords[1], self.target_coords[2], diameter=0.004)

    def __is_failed(self, joint_states: numpy.ndarray, contact_count: int, observation_space: Box, time_step: int = -1) -> Tuple[bool, ArmState]:
        info_dict = {'arm_state': ArmState.Undefined}
        # If there is any contact (collision), we consider the episode done
        if contact_count > 0:
            return True, ArmState.Collision

        # Check if time step exceeds limits, i.e. timed out
        # Time step starts from 0, that means if we only want to run 2 steps time_step will be 0,1 and we need to stop at 1
        if time_step + 1 >= self._max_time_step:
            return True, ArmState.Timeout

        # Check that joint values are not approaching limits
        upper_bound = observation_space.high[:3]  # First 3 values are the joint states
        lower_bound = observation_space.low[:3]
        min_dist_to_upper_bound = min(abs(joint_states - upper_bound))
        min_dist_to_lower_bound = min(abs(joint_states - lower_bound))
        # self.accepted_dist_to_bounds is basically how close to the joint limits can the joints go,
        # i.e. limit of 1.57 with accepted dist of 0.1, then the joint can only go until 1.47
        if min_dist_to_lower_bound < self.accepted_dist_to_bounds:
            joint_index = abs(joint_states - lower_bound).argmin()
            # print(f'Joint {joint_index} approach joint limits, '
            #       f'current joint value: {joint_states[joint_index]}, '
            #       f'minimum joint value: {lower_bound[joint_index]}')
            return True, ArmState.ApproachJointLimits
        if min_dist_to_upper_bound < self.accepted_dist_to_bounds:
            joint_index = abs(joint_states - upper_bound).argmin()
            # print(f'Joint {joint_index} approach joint limits, '
            #       f'current joint value: {joint_states[joint_index]}, '
            #       f'maximum joint value: {upper_bound[joint_index]}')
            info_dict['arm_state'] = ArmState.ApproachJointLimits
            return True, ArmState.ApproachJointLimits
        # Didn't fail
        return False, ArmState.Undefined

    def __calc_dist_change(self, coords_init: numpy.ndarray,
                           coords_next: numpy.ndarray) -> float:
        # Efficient euclidean distance calculation by numpy, most likely uses vector instructions
        diff_abs_init = numpy.linalg.norm(coords_init - self.target_coords)
        diff_abs_next = numpy.linalg.norm(coords_next - self.target_coords)

        return diff_abs_init - diff_abs_next

    def __calc_exponential_reward(self, coords_init: numpy.ndarray, coords_next: numpy.ndarray) -> float:
        def calc_cum_reward(dist: float, scaling=5.0):
            # Since dist scales from 1 and ends with 0, which is the opposite of the intended curve, we change x to y where y = 1-x
            # Now y scales from 0 to 1, and then we use y as the "normalised distance"
            y = 1 - dist  # Change the variable such that max reward is when dist is = 0, and reward = 0 when dist is 1
            if y < 0:
                # Linear in negative region, if y = -1 reward is -5, y = 0 reward is 0
                if y < -8:
                    cum_neg_rew = (y + 8) * 0.5 + (-40)
                else:
                    cum_neg_rew = y * 5

                # cum_neg_rew = -1 / scaling * (math.exp(scaling * -y) - 1)
                return cum_neg_rew
            else:
                cum_positive_rew = 1 / scaling * (math.exp(scaling * y) - 1)
                return cum_positive_rew

        # compute exponential scaling normalised reward
        # formula = integral(e^0.4x) from x_init to x_final, x is normalised distance from goal
        # total cumulative reward = 1/scaling * (e^0.4 x_final - 1)
        mag_target = numpy.linalg.norm(self.initial_coords - self.target_coords)
        diff_abs_init_scaled = numpy.linalg.norm(coords_init - self.target_coords) / mag_target
        diff_abs_next_scaled = numpy.linalg.norm(coords_next - self.target_coords) / mag_target

        prev_cum_rew = calc_cum_reward(diff_abs_init_scaled, self.exp_rew_scaling)
        current_cum_rew = calc_cum_reward(diff_abs_next_scaled, self.exp_rew_scaling)
        cum_rew_change = current_cum_rew - prev_cum_rew
        return cum_rew_change

    def __get_coords(self, joint_states: numpy.ndarray) -> numpy.ndarray:
        if len(joint_states) != 3:
            print(f'Expected 3 values for joint states, but got {len(joint_states)} values instead')
            return numpy.array([0.0, 0.0, 0.0])

        res = self._fk.calculate('world', 'grip_end_point', joint_states)
        return numpy.array([res.translation.x, res.translation.y, res.translation.z])

    def __generate_target_coords(self) -> Tuple[numpy.ndarray, numpy.ndarray]:
        if self.random_goal_seed is not None:
            self.__seed_numpy()

        while True:
            random_joint_values = numpy.random.uniform([-2.3562, -1.5708, -1.5708], [2.3562, 0.5, 1.5708])
            res = self._fk.calculate('world', 'grip_end_point', random_joint_values)
            if res.translation.z > 0.0:
                break
        target_coords = numpy.array([res.translation.x, res.translation.y, res.translation.z])
        self.np_random_state = numpy.random.get_state()
        return random_joint_values, target_coords

    def __generate_adjacent_coords(self, joint_values: numpy.ndarray):
        if self.random_goal_seed is not None:
            self.__seed_numpy()

        while True:
            rand_range = numpy.array([1.0, 1.0, 1.0]) * 0.1
            random_addition = numpy.random.uniform(-rand_range, rand_range)
            joint_values_adjacent = joint_values + random_addition
            joint_values_adjacent = joint_values_adjacent.clip([-2.356194, -1.570796, -1.570796], [2.356194, 0.5, 1.570796])
            res = self._fk.calculate('world', 'grip_end_point', joint_values_adjacent)
            if res.translation.z > 0.0:
                break
        target_coords = numpy.array([res.translation.x, res.translation.y, res.translation.z])
        self.np_random_state = numpy.random.get_state()
        return joint_values_adjacent, target_coords

    def __get_target_coords(self) -> Tuple[numpy.ndarray, numpy.ndarray]:
        rand_num = numpy.random.rand()
        # if probability to choose from buffer is very high, i.e. p > 0.90, we make sure buffer is filled first before choosing from buffer
        high_prob_but_buffer_unfilled = self.goal_from_buffer_prob > 0.90 and len(self.coords_buffer) < self.coords_buffer.maxlen
        select_from_buffer = rand_num < self.goal_from_buffer_prob
        buffer_is_empty = len(self.coords_buffer) == 0

        # Get coordinates from file if it exists
        if self.random_goal_file is not None:
            if not hasattr(self, 'first_run_goal_file'):
                self.first_run_goal_file = True
                print("Loading coordinates from file, not using random coords")
                self.file_target_coords = numpy.load(self.random_goal_file)
                self.file_coords_index = 0
            current_file_target_coords = self.file_target_coords[self.file_coords_index]
            self.file_coords_index += 1
            return numpy.array([0.0, 0.0, 0.0]), current_file_target_coords

        if buffer_is_empty or high_prob_but_buffer_unfilled or not select_from_buffer:
            # Generate random coords and store
            random_joint_values, target_coords = self.__generate_target_coords()
            # Store into buffer, since it is a deque with maxlen configured, it will auto pop, no need to manual pop
            self.coords_buffer.append((random_joint_values, target_coords))
            for _ in range(self.num_adjacent_goals):
                joint_vals, coords = self.__generate_adjacent_coords(random_joint_values)
                self.coords_buffer.append((joint_vals, coords))
            return random_joint_values, target_coords

        if len(self.coords_buffer) == self.coords_buffer.maxlen and not hasattr(self, 'first_time_printing_coords'):
            # Use this attribute to determine if we have printed coords or not, if no such attribute means first time printing
            self.first_time_printing_coords = True

            def print_list(list_):
                for item in list_:
                    print(item)

            print('Using target coords: ')
            print_list(enumerate([y for x, y in self.coords_buffer]))

        random_joint_values, target_coords = random.choice(self.coords_buffer)
        return random_joint_values, target_coords

    def __seed_numpy(self):
        # Properly seed the numpy RNG if a random seed is given
        # The set state and get_state is such that this generator function always returns the same set of values given the same seed
        # This is regardless of how many random calls are used in between
        if hasattr(self, 'np_random_state'):
            numpy.random.set_state(self.np_random_state)
        else:
            numpy.random.seed(self.random_goal_seed)
