## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl.pool.envs.ur5jointcontrol
## -- Module  : urjointcontrol.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-09-13  0.0.0     WB       Creation
## -- 2021-09-13  1.0.0     WB       Released first version
## -- 2021-09-13  1.0.1     WB       Instantiated without WrEnvGym
## -- 2021-09-23  1.0.2     WB       Increased C_LATENCY
## -- 2021-10-05  1.0.3     SY       Update following new attributes done and broken in State
## -- 2021-12-03  1.0.4     DA       Refactoring
## -- 2021-12-20  1.0.5     DA       Replaced 'done' by 'success'
## -- 2021-12-20  1.0.6     WB       Update 'success' and 'broken' rule
## -- 2021-12-21  1.0.7     DA       Class UR5JointControl: renamed method reset() to _reset()
## -- 2022-01-20  1.0.8     MRD      Use the gym wrapper to wrap the ur5 environment
## -- 2022-03-04  1.1.0     WB       Adds the ability to control gripper
## -- 2022-06-06  2.0.0     MRD      Add ability to self build the ros workspace
## --                       MRD      Add the connection to the real robot, wrapped in Gym environment
## -- 2022-06-07  2.0.1     MRD      Define _export_action and _import_action for each communication type
## -- 2022-06-11  2.1.0     MRD      Finalizing simulation and real environment
## -------------------------------------------------------------------------------------------------

"""
Ver. 2.1.0 (2022-06-11)

This module provides an environment with multivariate state and action spaces 
based on the Gym-based environment 'UR5RandomTargetTask-v0'. 
"""

import sys
import platform
import subprocess
import time
import os
from dotenv import load_dotenv
import em
import netifaces as ni
import mlpro
from mlpro.bf.various import Log
from mlpro.wrappers.openai_gym import *
from mlpro.rl.models import *
import numpy as np



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class UR5JointControl(Environment):
    """
    This environment multivariate space and action spaces by duplicating the
    Gym-based environment 'UR5RandomTargetTask-v0'. 
    """

    C_NAME = 'UR5JointControl'
    C_LATENCY = timedelta(0, 0, 0)
    C_INFINITY = np.finfo(np.float32).max

    ## -------------------------------------------------------------------------------------------------
    def __init__(self, p_seed=0, p_build=True, p_real=False, p_start_simulator=True, p_start_ur_driver=False,
        p_ros_server_ip="localhost", p_server_port=11311, p_net_interface="eth0", p_robot_ip="", p_reverse_ip="", 
        p_reverse_port=50001, p_visualize=False, p_logging=True):
        """
        Parameters:
            p_logging       Boolean switch for logging
        """

        # Use ROS as the simulation and real
        if not p_real:
            super().__init__(p_mode=Mode.C_MODE_SIM, p_logging=p_logging)
        else:
            super().__init__(p_mode=Mode.C_MODE_REAL, p_logging=p_logging)
            self._real_ros_state = None

        if p_build:
            self._build()

        try:
            import rospy
            from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
            from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
        except (ModuleNotFoundError, ImportError) as e:
            self.log(Log.C_LOG_TYPE_E, e)
            self.log(Log.C_LOG_TYPE_E, "ROS or UR5 Workspace is not installed")
            self.log(Log.C_LOG_TYPE_W, "Please use the guideline here:")
            self.log(Log.C_LOG_TYPE_W, "https://mlpro.readthedocs.io/en/latest/content/rl/env/pool/ur5jointcontrol.html")
        else:
            ros_ws_path = mlpro.rl.pool.envs.ur5jointcontrol.__file__.replace("/__init__.py", "")
            dot_env_file_template = os.path.join(ros_ws_path, "ur5.env.template")
            dot_env_file = os.path.join(ros_ws_path, "ur5.env")
            env_data = {
                "ros_master_ip" : "http://"+p_ros_server_ip+":"+str(p_server_port),
                "ros_ip" : ni.ifaddresses(p_net_interface)[ni.AF_INET][0]['addr']
            }

            data = None
            with open(dot_env_file_template, 'r') as f:
                data = f.read()
            data = em.expand(data, env_data)

            with open(dot_env_file, 'w+') as f:
                f.write(data)

            load_dotenv(dot_env_file)
            
            if p_ros_server_ip == "localhost":
                roscore = subprocess.Popen('roscore')
            rospy.init_node('ur5_lab_training_start', anonymous=True, log_level=rospy.WARN)

            LoadYamlFileParamsTest(rospackage_name="ur5_lab",
                                rel_path_from_package_to_file="config",
                                yaml_file_name="ur5_lab_task_param.yaml")

            rospy.set_param('ros_ws_path', ros_ws_path)
            rospy.set_param('sim', not p_real)

            # Init OpenAI_ROS ENV
            if not p_real:
                environment = rospy.get_param('/ur5_lab/simulation_environment')
                rospy.set_param('visualize', p_visualize)
                rospy.set_param('start_gazebo', p_start_simulator)
            else:
                environment = rospy.get_param('/ur5_lab/real_environment')
                rospy.set_param('robot_ip', p_robot_ip)
                rospy.set_param('reverse_ip', p_reverse_ip)
                rospy.set_param('reverse_port', p_reverse_port)
                rospy.set_param('start_driver', p_start_ur_driver)

            max_step_episode = rospy.get_param('/ur5_lab/max_iterations')

            self._gym_env = StartOpenAI_ROS_Environment(environment, max_step_episode)
            self._gym_env.seed(p_seed)

            self.C_NAME = 'Env "' + self._gym_env.spec.id + '"'

            self._state_space = WrEnvGYM2MLPro.recognize_space(self._gym_env.observation_space)
            self._action_space = WrEnvGYM2MLPro.recognize_space(self._gym_env.action_space)
        

    ## -------------------------------------------------------------------------------------------------
    @staticmethod
    def setup_spaces():
        return None, None


    ## -------------------------------------------------------------------------------------------------
    def _export_action(self, p_action: Action) -> bool:
        try:
            self._real_ros_state = self.simulate_reaction(None, p_action)
        except Exception as e:
            self.log(Log.C_LOG_TYPE_E, e)
            return False
        else:
            return True

    ## -------------------------------------------------------------------------------------------------
    def _import_state(self) -> bool:
        try:
            self._set_state(self._real_ros_state)
        except Exception as e:
            self.log(Log.C_LOG_TYPE_E, e)
            return False
        else:
            return True

    ## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed=None):
        # 1 Reset Gym environment and determine initial state
        try:
            observation = self._gym_env.reset(seed=p_seed)
        except:
           self._gym_env.seed(p_seed)
           observation = self._gym_env.reset() 
        obs = DataObject(observation)

        # 2 Create state object from Gym observation
        state = State(self._state_space)
        state.set_values(obs.get_data())
        self._set_state(state)

    ## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_state: State, p_action: Action) -> State:

        # 1 Convert action to Gym syntax
        action_sorted = p_action.get_sorted_values()
        dtype = self._gym_env.action_space.dtype

        if (dtype == np.int32) or (dtype == np.int64):
            action_sorted = action_sorted.round(0)

        if action_sorted.size == 1:
            action_gym = action_sorted.astype(self._gym_env.action_space.dtype)[0]
        else:
            action_gym = action_sorted.astype(self._gym_env.action_space.dtype)

        # 2 Process step of Gym environment
        try:
            observation, reward_gym, done, info = self._gym_env.step(action_gym)
        except:
            observation, reward_gym, done, info = self._gym_env.step(np.atleast_1d(action_gym))

        obs = DataObject(observation)

        # 3 Create state object from Gym observation
        state = State(self._state_space, p_terminal=done)
        state.set_values(obs.get_data())

        # 4 Create reward object
        self._last_reward = Reward(Reward.C_TYPE_OVERALL)
        self._last_reward.set_overall_reward(reward_gym)

        # 5 Return next state
        return state

    ## -------------------------------------------------------------------------------------------------
    def _compute_reward(self, p_state_old: State = None, p_state_new: State = None) -> Reward:
        return self._last_reward

    ## -------------------------------------------------------------------------------------------------
    def _compute_broken(self, p_state: State) -> bool:
        return self.get_broken()

    ## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure=None):
        pass

    ## -------------------------------------------------------------------------------------------------
    def update_plot(self):
        pass

    ## -------------------------------------------------------------------------------------------------
    def get_cycle_limit(self):
        return self._gym_env._max_episode_steps

    ## -------------------------------------------------------------------------------------------------
    def _compute_success(self, p_state: State) -> bool:
        obs = p_state.get_values()
        close = np.allclose(a=obs[:3],
                            b=obs[3:],
                            atol=0.1)

        if close:
            self._state.set_terminal(True)

        return close

    ## -------------------------------------------------------------------------------------------------
    def _build(self):
        # Check OS
        self.log(Log.C_LOG_TYPE_I, "Checking Operating System ......")
        if platform.system() != "Linux":
            self.log(Log.C_LOG_TYPE_E, "Operating System is not supported!")
            self.log(Log.C_LOG_TYPE_E, "Please use Linux")
            self.log(Log.C_LOG_TYPE_E, "Exiting....")
            sys.exit()
        else:
            self.log(Log.C_LOG_TYPE_S, "Operating System is supported")

        # Check if ROS is installed
        process = subprocess.run("which roscore", shell=True, stdout=subprocess.PIPE)
        output = process.stdout
        self.log(Log.C_LOG_TYPE_I, "Checking ROS Installation ......")
        if output==bytes():
            self.log(Log.C_LOG_TYPE_E, "ROS is not installed!")
            self.log(Log.C_LOG_TYPE_E, "Please install ROS")
            self.log(Log.C_LOG_TYPE_E, "Exiting....")
            sys.exit()
        else:
            self.log(Log.C_LOG_TYPE_S, "ROS is installed")

        import rospkg

        # Check if UR5 Workspace is installed
        installed = False
        rospack = rospkg.RosPack()
        try:
            rospack.get_path("ur5_lab")
        except rospkg.common.ResourceNotFound:
            self.log(Log.C_LOG_TYPE_E, "UR5 Workspace is not installed!")
            self.log(Log.C_LOG_TYPE_W, "If you have ran this script, please CTRL+C and restart terminal")
        else:
            installed = True

        if not installed:
            self.log(Log.C_LOG_TYPE_W, "Building ROS Workspace in 10 Seconds")
            for sec in range(10):
                time.sleep(1)
                self.log(Log.C_LOG_TYPE_W, str(9-sec)+"...")

            ros_workspace = os.path.dirname(mlpro.__file__)+"/rl/pool/envs/ur5jointcontrol"

            # Check dependencies
            command = "cd " + ros_workspace + " && sudo apt-get update && rosdep update && rosdep install --from-paths src --ignore-src -r -y"
            try:
                process = subprocess.check_output(command, shell=True)
            except subprocess.CalledProcessError as e:
                self.log(Log.C_LOG_TYPE_E, "Installing Dependencies Failed")
                sys.exit()

            # Build Workspace
            command = "cd " + ros_workspace + " && catkin_make"
            try:
                process = subprocess.check_output(command, shell=True)
            except subprocess.CalledProcessError as e:
                self.log(Log.C_LOG_TYPE_E, "Build Failed")
                sys.exit()

            self.log(Log.C_LOG_TYPE_S, "Successfully Built")
            command = "echo 'source "+ros_workspace+"/devel/setup.bash"+"' >> ~/.bashrc"
            process = subprocess.run(command, shell=True)
            self.log(Log.C_LOG_TYPE_W, "Please restart your terminal and run the Howto script again")
            sys.exit()
        else:
            self.log(Log.C_LOG_TYPE_S, "UR5 Workspace is installed")