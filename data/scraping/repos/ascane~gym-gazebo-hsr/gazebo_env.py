import gym
import os
from os import path
import random
import rospy
import subprocess
import time


class GazeboEnv(gym.Env):
    """Superclass for all Gazebo environments.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, launchfile):
        random_number = random.randint(10000, 15000)
        self.port = '11311'  # str(random_number) # os.environ["ROS_PORT_SIM"]
        self.port_gazebo = str(random_number + 1)  # os.environ["ROS_PORT_SIM"]

        with open("log.txt", "a") as myfile:
            myfile.write("export ROS_MASTER_URI=http://localhost:" + self.port + "\n")
            myfile.write("export GAZEBO_MASTER_URI=http://localhost:" + self.port_gazebo + "\n")

        # Start roscore
        self.proc_core = subprocess.Popen(['roscore', '-p', self.port])
        time.sleep(1)
        print("Roscore launched!")

        # Launch the simulation with the given launchfile name
        if launchfile.startswith("/"):
            fullpath = launchfile
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", "launch", launchfile)
        if not path.exists(fullpath):
            raise IOError("File " + fullpath + " does not exist")

        self.proc_launch = subprocess.Popen(['roslaunch', '-p', self.port, fullpath])
        print ("Gazebo launched!")
        rospy.wait_for_service('/gazebo/reset_world')

        self.proc_gzclient = None

        # In order for a ROS node to use simulation time according to the /clock topic, the /use_sim_time parameter must
        # be set to true before the node is initialized. This is done in the launchfile. See wiki.ros.org/Clock.
        rospy.init_node('gym', anonymous=True)

    def reset(self):
        # Implemented in subclass
        raise NotImplementedError

    def step(self, action):
        # Implement this method in every subclass
        # Perform a step in gazebo. E.g. move the robot
        raise NotImplementedError

    def _terminate_gzclient(self):
        # self.proc_gzclient.poll() is None means child process has not terminated
        if self.proc_gzclient is not None and self.proc_gzclient.poll() is None:
            self.proc_gzclient.terminate()
            self.proc_gzclient.wait()

    def render(self, mode="human", close=False):
        if close:
            self._terminate_gzclient()
            return

        if self.proc_gzclient is None or self.proc_gzclient.poll() is not None:
            self.proc_gzclient = subprocess.Popen("gzclient")

    def close(self):
        # Terminate gzclient, roslaunch and roscore
        self._terminate_gzclient()
        self.proc_launch.terminate()
        self.proc_core.terminate()
        self.proc_launch.wait()
        self.proc_core.wait()

    def _configure(self):
        # TODO
        # From OpenAI API: Provides runtime configuration to the environment
        # Maybe set the Real Time Factor?
        pass

    def seed(self):
        # TODO
        # From OpenAI API: Sets the seed for this env's random number generator(s)
        pass
