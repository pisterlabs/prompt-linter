#import rospy
#import roslaunch
import os
import signal
import subprocess
import time
from os import path

class GazeboEnv():
	"""Superclass for all Gazebo environments.
	"""
	metadata = {'render.modes': ['human']}

	def __init__(self, launchfile, gui=True):

		#port = os.environ["ROS_PORT_SIM"]
		port = "11311"
		#start roscore
		subprocess.Popen(["roscore", "-p", port])
		time.sleep(1)
		print ("Roscore launched!")

		# Launch the simulation with the given launchfile name
		#rospy.init_node('gym', anonymous=True)

		if launchfile.startswith("/"):
			fullpath = launchfile
		elif launchfile.startswith("~"):
			fullpath = os.path.expanduser('~') + launchfile [1:]
		else:
			fullpath = os.path.join(os.path.dirname(__file__), "assets","launch", launchfile)
		if not path.exists(fullpath):
			raise IOError("File "+fullpath+" does not exist")

		if not gui:
			subprocess.Popen(["roslaunch","-p", port, fullpath, "world_name:=$(find spyndra_gazebo)/worlds/spyndra_rl.world", "gui:=false"])
		else:
			subprocess.Popen(["roslaunch","-p", port, fullpath, "world_name:=$(find spyndra_gazebo)/worlds/spyndra_rl.world"])
		print ("Gazebo launched!")

		self.gzclient_pid = 0

	def _step(self, action):

		# Implement this method in every subclass
		# Perform a step in gazebo. E.g. move the robot
		raise NotImplementedError

	def _reset(self):

		# Implemented in subclass
		raise NotImplementedError

	def _render(self, mode="human", close=False):

		if close:
			tmp = os.popen("ps -Af").read()
			proccount = tmp.count('gzclient')
			if proccount > 0:
				if self.gzclient_pid != 0:
					os.kill(self.gzclient_pid, signal.SIGTERM)
					os.wait()
			return

		tmp = os.popen("ps -Af").read()
		proccount = tmp.count('gzclient')
		if proccount < 1:
			subprocess.Popen("gzclient")
			self.gzclient_pid = int(subprocess.check_output(["pidof","-s","gzclient"]))
		else:
			self.gzclient_pid = 0

	def _close(self):

		# Kill gzclient, gzserver and roscore
		tmp = os.popen("ps -Af").read()
		gzclient_count = tmp.count('gzclient')
		gzserver_count = tmp.count('gzserver')
		roscore_count = tmp.count('roscore')
		rosmaster_count = tmp.count('rosmaster')

		if gzclient_count > 0:
			os.system("killall -9 gzclient")
		if gzserver_count > 0:
			os.system("killall -9 gzserver")
		if rosmaster_count > 0:
			os.system("killall -9 rosmaster")
		if roscore_count > 0:
			os.system("killall -9 roscore")

		if (gzclient_count or gzserver_count or roscore_count or rosmaster_count >0):
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