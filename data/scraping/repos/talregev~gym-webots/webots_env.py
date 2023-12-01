import gym
import rospy
#import roslaunch
import sys
import os
import signal
import subprocess
import time
from std_srvs.srv import Empty
import random
from rosgraph_msgs.msg import Clock

class WebotsEnv(gym.Env):
    """Superclass for all Gazebo environments.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, worldfile):
        self.last_clock_msg = Clock()

        random_number = random.randint(10000, 15000)
        # self.port = "11311"#str(random_number) #os.environ["ROS_PORT_SIM"]
        # self.port_webots = "11345"#str(random_number+1) #os.environ["ROS_PORT_SIM"]
        self.port = str(random_number) #os.environ["ROS_PORT_SIM"]
        self.port_webots = str(random_number+1) #os.environ["ROS_PORT_SIM"]

        #self.port = str(11311)

        os.environ["ROS_MASTER_URI"] = "http://localhost:"+self.port
        #os.environ["Webots stream"] = "http://localhost:"+self.port_webots
        #
        # self.ros_master_uri = os.environ["ROS_MASTER_URI"];

        print("ROS_MASTER_URI=http://localhost:"+self.port + "\n")
        print("Webots Stream=http://localhost:"+self.port_webots + "\n")

        # self.port = os.environ.get("ROS_PORT_SIM", "11311")
        #roscore_file = subprocess.check_output(["which", "roscore"])
        #webots_file = (subprocess.check_output(["which", "webots"]))

        # NOTE: It doesn't make sense to launch a roscore because it will be done when spawing Gazebo, which also need
        #   to be the first node in order to initialize the clock.
        # # start roscore with same python version as current script
        # self._roscore = subprocess.Popen([sys.executable, os.path.join(ros_path, b"roscore"), "-p", self.port])
        # time.sleep(1)
        # print ("Roscore launched!")



        if worldfile.startswith("/"):
            fullpath = worldfile
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", "worlds", worldfile)

        if not os.path.exists(fullpath):
            raise IOError("File " + fullpath + " does not exist")

        self._roscore = subprocess.Popen(["roscore", "-p", self.port])
        #self._roscore = subprocess.Popen(["rosparam", "set", "/use_sim_time", "true"])
        port_param = '--stream="port=' + self.port_webots + '"'
        stdout_param ='--stdout'
        self._webots = subprocess.Popen(["webots", "--batch", "--no-sandbox", "--stderr", port_param,  fullpath])


        print ("Webots launched!")

        self.webots_pid = 0

        # Launch the simulation with the given launchfile name
        rospy.init_node('gym', anonymous=True)

        ################################################################################################################
        # r = rospy.Rate(1)
        # self.clock_sub = rospy.Subscriber('/clock', Clock, self.callback, queue_size=1000000)
        # while not rospy.is_shutdown():
        #     print("initialization: ", rospy.rostime.is_rostime_initialized())
        #     print("Wallclock: ", rospy.rostime.is_wallclock())
        #     print("Time: ", time.time())
        #     print("Rospyclock: ", rospy.rostime.get_rostime().secs)
        #     # print("/clock: ", str(self.last_clock_msg))
        #     last_ros_time_ = self.last_clock_msg
        #     print("Clock:", last_ros_time_)
        #     # print("Waiting for synch with ROS clock")
        #     # if wallclock == False:
        #     #     break
        #     r.sleep()
        ################################################################################################################

    # def callback(self, message):
    #     """
    #     Callback method for the subscriber of the clock topic
    #     :param message:
    #     :return:
    #     """
    #     # self.last_clock_msg = int(str(message.clock.secs) + str(message.clock.nsecs)) / 1e6
    #     # print("Message", message)
    #     self.last_clock_msg = message
    #     # print("Message", message)

    def step(self, action):

        # Implement this method in every subclass
        # Perform a step in gazebo. E.g. move the robot
        raise NotImplementedError

    def reset(self):

        # Implemented in subclass
        raise NotImplementedError

    def _render(self, mode="human", close=False):

        if close:
            tmp = os.popen("ps -Af").read()
            proccount = tmp.count('webots')
            if proccount > 0:
                if self.webots_pid != 0:
                    os.kill(self.webots_pid, signal.SIGTERM)
                    os.wait()
            return

        tmp = os.popen("ps -Af").read()
        proccount = tmp.count('webots')
        if proccount < 1:
            subprocess.Popen("webots")
            self.webots_pid = int(subprocess.check_output(["pidof","-s","webots"]))
        else:
            self.webots_pid = 0

    def _close(self):

        # Kill webots and roscore
        tmp = os.popen("ps -Af").read()
        webots_count = tmp.count('webots')
        roscore_count = tmp.count('roscore')
        rosmaster_count = tmp.count('rosmaster')

        if webots_count > 0:
            os.system("killall -9 webots")
        if rosmaster_count > 0:
            os.system("killall -9 rosmaster")
        if roscore_count > 0:
            os.system("killall -9 roscore")

        if (webots_count or roscore_count or rosmaster_count >0):
            os.wait()

    def _configure(self):

        # TODO
        # From OpenAI API: Provides runtime configuration to the enviroment
        # Maybe set the Real Time Factor?
        pass
    def _seed(self):

        # TODO
        # From OpenAI API: Sets the seed for this env's random number generator(s)
        pass
