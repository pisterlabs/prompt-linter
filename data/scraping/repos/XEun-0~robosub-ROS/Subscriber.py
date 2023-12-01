import rospy
import threading
from computer_vision.msg import all_targets
from guidance_navigation_control.msg import sensorInfo_actuatorStatus
from guidance_navigation_control.msg import task_desiredAction
from guidance_navigation_control.msg import controlCommand

topic_to_datatype = {'target':all_targets, 'sensorInfo_actuatorStatus':sensorInfo_actuatorStatus,
		     'task_desiredAction':task_desiredAction, 'controlCommand':controlCommand}

class Subscribe_to():
	def __init__(self, topic):
		self.mutex = threading.Lock()
		self.foo = rospy.Subscriber(topic, topic_to_datatype[topic], self.callback)
		self.data = topic_to_datatype[topic]()
		self.new_data = False

	def callback(self, cb_data):	#Gets data from publisher
		self.mutex.acquire()
		self.data = cb_data
		self.new_data = True
		self.mutex.release()

	def get_data(self):		#Gives you the most recent data got
		self.mutex.acquire()
		self.final_data = self.data
		self.new_data = False
		self.mutex.release()
		return self.final_data

	def was_data_sent(self):	#Tells you if new data was acquired yet
		self.mutex.acquire()
		self.check = self.new_data
		self.mutex.release()
		return self.check
