#!/usr/bin/env python3
import rospy
import actionlib
import time
from std_msgs.msg import String
from std_msgs.msg import Bool
# from actions.msg import navServAction, navServGoal, navServResult 
#
from intercom.msg import action_selector_cmd, bring_something_cmd
from object_detector.msg import DetectObjects3DAction, DetectObjects3DGoal  
import time
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from enum import Enum
from actionlib_msgs.msg import GoalID
import os
import openai

import os
import json
import sys


START_STATE = "start"
SPEECH_STATE = "speech"
NAVIGATION1_STATE = "navigation1"
OBJECTDETECTION_STATE = "object_detection"
NAVIGATION2_STATE = "navigation2"
END_STATE = "done"

class MoveGoals(Enum):
    KITCHEN = 1
    COUCH = 2
    BATHROOM = 3
    CLOSET = 4

class ManipulationGoals(Enum):
    VEGETABLES = 1
    
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="",
  temperature=0.7,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

class Tmr2022Main(object):

    def __init__(self):
        self.currentState =  START_STATE
        # Conversation/Speech
        self.speech_enable = rospy.Publisher("inputAudioActive", Bool, queue_size=10)
        self.parser_listener = rospy.Subscriber('action/bring_something', bring_something_cmd, self.listen_parser)
        
        # Navigation
        self.cancel_move = rospy.Publisher("/move_base/cancel", GoalID, queue_size=10)
        
        self.initial_pose = None
        self.initial_pose_sub = rospy.Subscriber('/initialpose', PoseWithCovarianceStamped, self.initial_pose_cb)
        rospy.loginfo("Waiting for MoveBase AS...")
        self.move_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.move_client.wait_for_server()
        rospy.loginfo("MoveBase AS Loaded ...")
        # self.nav_client = actionlib.SimpleActionClient('navServer', navServAction)
        # self.nav_client.wait_for_server()

        # Vision
        self.vision2D_enable = rospy.Publisher("detectionsActive", Bool, queue_size=10)

        self.run()

    def initial_pose_cb(self, msg):
        self.initial_pose = PoseStamped(header = msg.header, pose = msg.pose.pose)

    def run(self):
        self.currentState = SPEECH_STATE
        while (self.currentState == SPEECH_STATE) and not rospy.is_shutdown():
            pass
        self.speech_enable.publish(Bool(False))

    def nav_goal(self, target = MoveGoals.KITCHEN):
        start_time = time.time()
        # rostopic pub /move_base/cancel actionlib_msgs/GoalID -- {}
        class NavGoalScope:
            target_location = target.name
            result = False
            pose = PoseStamped()
            
            result_received = False
        
        def nav_goal_feedback(feedback_msg):
            NavGoalScope.pose = feedback_msg.pose
        
        def get_result_callback(state, result):
            NavGoalScope.result = result.result

            NavGoalScope.result_received = True
            rospy.loginfo("Nav Goal Finished")

        rospy.loginfo("Sending Nav Goal")
        self.nav_client.send_goal(
                    navServGoal(target_location = NavGoalScope.target_location),
                    feedback_cb=nav_goal_feedback,
                    done_cb=get_result_callback)
        
        while not NavGoalScope.result_received and time.time() - start_time < 50:
            pass

        if not NavGoalScope.result_received:
            self.cancel_move.publish(GoalID())
            return False
        
        return NavGoalScope.result

    def back_to_origin(self):
        goal = MoveBaseGoal()
        self.move_client.send_goal(self.initial_pose)
        self.move_client.wait_for_result()

    def listen_parser(self, msg):
        rospy.loginfo("Parser Received " + msg.place + "-" + msg.object)
        if len(msg.place) == 0 or len(msg.object) == 0:
            return
        ## Received Cmd
        self.targetPlace = MoveGoals[msg.place.upper()]
        self.targetObject = ManipulationGoals[msg.object.upper()]
        time.sleep(15)
        self.speech_enable.publish(Bool(False))
        # GOTO-NAV
        self.nav_goal(self.targetPlace)
        self.currentState = NAVIGATION1_STATE

        # 15 Seconds Vision Enable
        start = time.time()
        self.vision2D_enable.publish(Bool(True))
        while time.time() - start < 60:
            pass
        self.vision2D_enable.publish(Bool(False))

        ## GO BACK TO ORIGIN
        self.back_to_origin()

def main():
    rospy.init_node('Tmr2022Main', anonymous=True)
    rospy.loginfo("Tmr2022Main initialized.")
    Tmr2022Main()
    rospy.spin()

if __name__ == '__main__':
    main()