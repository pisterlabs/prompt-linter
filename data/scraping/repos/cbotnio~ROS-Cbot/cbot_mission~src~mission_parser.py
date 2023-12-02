#! /usr/bin/env python

###################################################
# Mission Parser made for NIO
# Author: Mohit Gupta, BITS Goa
###################################################

from __future__ import print_function
import rospy
import sys
from std_msgs.msg import Bool
from cbot_ros_msgs.srv import *
from cbot_ros_msgs.msg import *
import dynamic_reconfigure.client

import sys, math, time, json
from Compilers import MissionCompiler
from Handlers import GuidanceHandler,ControllerHandler, ActuatorHandler


guidanceStatus = False

wptInputs = {"X1": 0.0, "Y1": 0.0, "Z1": 0.0,
			 "heading": 0.0, "pitch": 0.0, "roll":0.0, "speed": 0.0,
			 "captureRadius": 1.0, "slipRadius": 2.0,
			 "mode": 0, "timeout": float("inf")}

lfwInputs = {"X1": 0.0, "Y1": 0.0, "Z1": 0.0,
			 "X2": 0.0, "Y2": 0.0, "Z2": 0.0,
			 "heading": 0.0, "pitch": 0.0, "roll":0.0, "speed": 0.0,
			 "captureRadius": 2.0, "mode": 0, "timeout": float("inf")}

arcInputs = {"Xs": 0.0, "Ys": 0.0, "Zs": 0.0,
			 "Xe": 0.0, "Ye": 0.0, "Ze": 0.0,
			 "Xc": 0.0, "Yc": 0.0, "Zc": 0.0,
			 "heading": 0.0, "pitch": 0.0, "speed": 0.0,
			 "arc_follow_direction": 0,
			 "captureRadius": 1.0, "mode": 0, "timeout": float("inf")}

guidanceTypes = {"wpt": ["position", "depth", "speed", "captureRadius", "slipRadius", "timeout"],
				 "wptFA": ["position", "depth", "speed", "heading","pitch","roll", "captureRadius", "slipRadius", "timeout"],
				 "lfw": ["position1", "position2", "depth", "speed", "captureRadius", "timeout"],
				 "lfwFA": ["position1", "position2", "depth", "speed", "heading","pitch","roll", "captureRadius", "timeout"],
				 "arc": ["centerCoord", "depth", "radius","speed", "captureRadius", "direction", "start", "timeout"],
				 "arcFA": ["centerCoord", "depth", "radius","speed", "heading","pitch","roll", "captureRadius", "direction", "start", "timeout"],
				 "dock": ["position", 	"depth", "heading", "runwayLength"]}
				 
controlTypes = {"constDepth": ["depth","timeout"],
				"constSpeed": ["speed","timeout"],
				"constHeading": ["heading","timeout"],
				"constPitch": ["pitch","timeout"]}
				
actuatorTypes = {"constThrust": ["cmf","dmf","cmv","dmv", "timeout"]}

behaviourTypes = {"loiter": ["timeout"]}

modeTable = {"wpt": 0, "lfw": 1, "arc": 2, "stkp": 3, "wptFA": 4, "lfwFA": 5, "arcFA": 6}

stopMissionFlag = 0

missionsCompletedFlag = 0

Mission = {}

uuv_mode = "rov"
uuv_status = "stop"

line_srv = rospy.ServiceProxy('/line_inputs', LineInputs)
arc_srv = rospy.ServiceProxy('/arc_inputs', ArcInputs)
control_inputs = rospy.Publisher('/controller_inputs', ControllerInputs, queue_size=1)
thruster_inputs = rospy.Publisher('/thruster_cmdm', ThrusterCMDM, queue_size=1)

guidance_handle = GuidanceHandler.GuidanceHandler()
control_handle = ControllerHandler.ControllerHandler()
actuator_handle = ActuatorHandler.ActuatorHandler()

def guidanceStatusCallback(data):
	global guidanceStatus
	guidanceStatus = data.data

def checkTimeout(startTime,timeout):
	currTime = time.time()
	timeElapsed = [(float(timeout[j]) - (currTime-x))>0 for j,x in enumerate(startTime)]
	try:
		Index = len(timeElapsed) - timeElapsed.index(0)
	except:
		Index = 0
	return Index

def updateTimeout(startTime, timeout):
	difference = time.time() - startTime[-1]
	for i in range(len(startTime)):
		startTime[i] += difference

def  checkStatus():
	global stopMissionFlag
	# Add all check conditions like resume, pause and stop
	if(uuv_mode=="auv" and uuv_status=="drive"):
		stopMissionFlag = 0
		return 1
	elif(uuv_status=="park"):
		try:
			updateTimeout(startTime,timeout)
		except:
			pass
		stopMissionFlag = 0
		return 2
	elif(uuv_status=="stop"):
		stopMissionFlag = 1
		return 3


def sendMission(inputs,missionType,dyn_params):
	global stopMissionFlag
	rospy.wait_for_service("/set_dyn_params")
	set_dyn_params_client = rospy.ServiceProxy('/set_dyn_params', String)
	if(bool(dyn_params)):
		data_string = str(json.dumps(dyn_params))
		ctrl_msg = StringRequest()
		ctrl_msg.data = data_string
		retries = 10
		resp = set_dyn_params_client(ctrl_msg) 
		if(resp.response==1):
			pass
		elif(resp.response==0):
			while(resp.response==0 and retries>0):
				print("Failed to update control params. Trying again in 1 second...")
				retries-=1
				time.sleep(1)
				resp = set_dyn_params_client(ctrl_msg)
			if(retries==0):
				print("Failed to update control params.")
				print("Aborting Mission...")
				stopMissionFlag = 1
				return 0

	if(missionType in guidanceTypes.keys()):
		if(missionType=="wpt" or missionType=="wptFA"):
			rospy.wait_for_service("/waypoint_inputs")
			waypoint_client = rospy.ServiceProxy('/waypoint_inputs', WaypointInputs)
			return waypoint_client(inputs)
		elif(missionType=="lfw" or missionType=="lfwFA"):
			rospy.wait_for_service("/linefollow_inputs")
			lfw_client = rospy.ServiceProxy('/linefollow_inputs', LineInputs)
			return lfw_client(inputs)
		elif(missionType=="arc" or missionType=="arcFA"):
			rospy.wait_for_service("/arcfollow_inputs")
			arc_client = rospy.ServiceProxy('/arcfollow_inputs', ArcInputs)
			return arc_client(inputs)
	elif(missionType in controlTypes.keys()):
		control_inputs.publish(inputs)
		return 1
	elif(missionType in actuatorTypes.keys()):
		thruster_inputs.publish(inputs)
		return 1


def parseSingleMission(names,timeout,startTime):
	global guidanceStatus, stopMissionFlag, guidance_handle, control_handle, actuator_handle
	i=0
	bhvType = ""
	flag=0
	while i<len(names):
		if(stopMissionFlag):
			break
		name = names[i]
		print("Mission Name: ", name)

		#################### Parse Behaviour Missions #############################
		if(name in Mission["BHVTable"].keys()):
			if(flag==0):
				try:
					timeout.append(Mission["BHVTable"][name]["timeout"])
				except:
					timeout.append(float('inf'))
				startTime.append(time.time())

			count = checkTimeout(startTime,timeout)
			while(count==0 and stopMissionFlag==0):
				parseSingleMission(Mission["BHVTable"][name]["names"],timeout,startTime)
				count = checkTimeout(startTime,timeout)
				flag=1
			timeout.pop()
			startTime.pop()
			
		#################### Parse Single Missions #############################
		elif(name in Mission["GuidanceTable"].keys()):
			MissionType = Mission["GuidanceTable"][name]["type"]
			MissionData = Mission["GuidanceTable"][name]["data"]

			################## Guidance Mission #############################
			if(MissionType in guidanceTypes.keys()):
				inputs,dyn_params = guidance_handle.parseGuidanceMission(MissionData,MissionType)

			#################### Control Mission #############################
			elif(MissionType in controlTypes.keys()):
				inputs,dyn_params = control_handle.parseControlMission(MissionData,MissionType)

			#################### Actuator Mission #############################
			elif(MissionType in actuatorTypes.keys()):
				inputs,dyn_params = actuator_handle.parseActuatorMission(MissionData,MissionType) 

			response =sendMission(inputs,MissionType,dyn_params)

			print("Next Mission Sent: ", response)
			guidanceStatus = False

			try:
				timeout.append(Mission["GuidanceTable"][name]["data"]["timeout"])
			except:
				timeout.append(float('inf'))

			startTime.append(time.time())
			n = checkStatus()

			################## Loop till mission completion or timeout #####################
			while(not guidanceStatus and stopMissionFlag==0):
				n = checkStatus()
				if(stopMissionFlag==1):
					break
				elif(n==2):
					while(n==2):
						updateTimeout(startTime,timeout)
						n=checkStatus()
						if(stopMissionFlag==1):
							break
					response = sendMission(inputs,MissionType,dyn_params)
				count = checkTimeout(startTime,timeout)
				if(count==1):
					break
				elif(count>1):
					startTime.pop()
					timeout.pop()
					return

			if(guidanceStatus):
				print("reached")
			else:
				print("Timeout")

			startTime.pop()
			timeout.pop()
		i+=1

def serviceCallback(req):
	global Mission
	Mission = json.loads(req.data)
	print("Mission recieved")

	res = StringResponse()
	res.response = 1
	return res

def thrusterCallback(config):
	global uuv_mode,uuv_status
	print("Mode Config Callback")
	uuv_mode = str(config.mode).lower()
	uuv_status = str(config.status).lower()

def controlCallback(config):
	pass

def guidanceCallback(config):
	pass

if __name__=='__main__':
		rospy.init_node('mission_parser')
		guidanceStatusSub = rospy.Subscriber('/guidanceStatus',Bool, guidanceStatusCallback)
		missionServer = rospy.Service('/missionParser', String, serviceCallback)
		
		thr_client = dynamic_reconfigure.client.Client("thruster_node", timeout=5, config_callback=thrusterCallback)
		control_client = dynamic_reconfigure.client.Client("control_node", timeout=5, config_callback=controlCallback)
		guidance_client = dynamic_reconfigure.client.Client("guidance_node", timeout=5, config_callback=guidanceCallback)

		r = rospy.Rate(10)

		while not rospy.is_shutdown():
			if(missionsCompletedFlag):
				thr_client.update_configuration({"status":"stop"})
				missionsCompletedFlag = 0

			while(not (uuv_mode=="auv" and uuv_status=="drive")):
				time.sleep(1)

			checkStatus()

			if(stopMissionFlag==0 and bool(Mission)):
				print("In Mission")

				MissionList = ["M"+str(y) for y in sorted([int(x[1:]) for x in Mission["Missions"].keys()])]
				print("Mission List: ", MissionList)

				for Mi in MissionList:
					if(stopMissionFlag):
						break
					missionsCompletedFlag = 1
					timeout = []
					startTime = []
					print("Mission Number: ", Mi)
					parseSingleMission(Mission["Missions"][Mi]["names"],timeout,startTime)

			r.sleep()
			
