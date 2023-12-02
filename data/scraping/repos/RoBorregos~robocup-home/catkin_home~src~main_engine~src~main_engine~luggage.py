#!/usr/bin/env python3
import rospy
import actionlib
import time
from std_msgs.msg import String, Bool
from action_selectors.msg import RawInput

import os
import openai
import json

promts = {
    "start": "Let's start carry my luggage task",
    "wait": "I am waiting for you to talk to me",
    "self_intro": "Hello, I'm homie, your personal assistant.",
    "point_to_bag" : "Which bag should I carry?",
    "right": "The one on your right, is that correct?",
    "left": "The one on your left, is that correct?",
    "correction" : "Thank you for the correction",
    "grasp": "I will now pick your luggage",
    "follow" : "I am ready to follow you",
    "car" : "Is this the car?",
    "deliver" : "Please take your bag",
    "returning" : "I am heading back home",
    "home" : "I am already back home"
}

calls = {
    "confirm" : "True or False the next message is a confirmation: ",
    "salute" : "True or False the next message is talking to me: "
}


INITIALIZATION_STATE = "init"
START_STATE = "start"
WAIT_STATE = "waiting for operator"
IDENTIFY_STATE = "identifying the bag"
GOING_STATE = "going to the car"
DELIVER_STATE = "deliver bag"
RETURN_STATE = "returning to the arena"
END_STATE = "done"

class receptionist(object):
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.currentState =  INITIALIZATION_STATE
        self.iteration = 0
        # GPT API
        self.GPT_model="text-davinci-003"
        self.GPT_temperature=0.7
        self.GPT_top_p=1
        self.GPT_frequency_penalty=0
        self.GPT_presence_penalty=0 
        rospy.logdebug("GPT API initialized")
        # Conversation initialization
        self.inputText = ""
        self.speech_enable = rospy.Publisher("inputAudioActive", Bool, queue_size=10)
        self.saying = False
        self.say_listener = rospy.Subscriber("saying", Bool, self.say_callback)
        self.speech_enable.publish(Bool(False))
        self.say_publisher = rospy.Publisher('robot_text', String, queue_size=20)
        self.input_suscriber = rospy.Subscriber("RawInput", RawInput, self.input_callback)

        #Nav subscribers 
        #self.cancel_move = rospy.Publisher("/move_base/cancel", GoalID, queue_size=10)
        #self.initial_pose = None
        #self.initial_pose_sub = rospy.Subscriber('/initialpose', PoseWithCovarianceStamped, self.initial_pose_cb)
        self.is_moving = False

        rospy.loginfo("Carry-Luggage initialized")

        # Conversation dependencies
        self.currentState = START_STATE
        # wait for the robot to be ready
        time.sleep(1)
        # start the main loop
        self.run()
        rospy.spin()

    def say_callback(self, data):
        self.saying = data.data
        rospy.logwarn("saying: " + str(self.saying))
    
    def run(self):
        rospy.loginfo("-----------------------------------------RUN-----------------------------------------")

        while not rospy.is_shutdown():
            rospy.loginfo("-----------------------------------------RUN-----------------------------------------")

            if self.currentState == START_STATE:
                self.speech_enable.publish(Bool(False))

                rospy.loginfo("Carry mu luggage is ready to start")
                self.say(promts["start"])
                self.currentState = WAIT_STATE
                
            elif self.currentState == WAIT_STATE:
                if self.is_someone():
                    self.say(promts["self_intro"])
                    self.currentState = IDENTIFY_STATE
                else:
                    #if there is no one, wait for 3 seconds and check again
                    self.say(promts["wait"])           
                
            elif self.currentState == IDENTIFY_STATE:
                rospy.loginfo("Starting pick of luggage")
                grasped = self.get_bag()
                if grasped != "":
                    self.currentState = GOING_STATE
                else:
                    self.currentState = IDENTIFY_STATE   

            elif self.currentState == GOING_STATE:
                rospy.loginfo("Following the person to the car")
                reached = self.go_state()
                if reached != "":
                    self.currentState = RETURN_STATE #DELIVER_STATE
                else:
                    self.currentState == GOING_STATE
            
            elif self.currentState == RETURN_STATE:
                rospy.loginfo("Going back home")
                returned = self.return_to()
                if returned != "":
                    self.currentState = END_STATE
                else:
                    self.currentState == RETURN_STATE
                    
    def say(self, text):
        time.sleep(1)   
        if self.saying == False:
            self.saying == True
            self.say_publisher.publish(text)
        time.sleep(len(text)/8 )        
    
    def get_bag(self):
        self.speech_enable.publish(Bool(False))
        grasped = ""
        self.inputText = ""
        i = 0
        rospy.logwarn("Waiting for operator")
        self.say(promts["point_to_bag"])
        #como poner aqui algo, como suscribirse a un topico
        #que le indique a donde apunto
        #flag = var->subscriber()
        flag = 0 #for tests in 0, should be on -1
        if flag == "0":
            self.say(promts["right"])
        elif flag == "1":
            self.say(promts["left"])
        else: #flag == ""
            return grasped
        time.sleep(5)
        rospy.logdebug("speech enabled")
        self.speech_enable.publish(Bool(True))
        while not rospy.is_shutdown() and grasped == "":
            i+=1
            time.sleep(1)
            rospy.logdebug(grasped)
            if self.inputText != "":
                    grasped = self.parser(calls["confirm"])
            if i%28 == 0:
                self.speech_enable.publish(Bool(False))
                if flag == "0":
                    self.say(promts["right"])
                else:
                    self.say(promts["left"])
                self.speech_enable.publish(Bool(True))
                i = 0
        rospy.logdebug("speech disabled")
        self.speech_enable.publish(Bool(False))
        if grasped == "False":
            #settear la variable con el lado de la bag correcto
            flag = "left" if flag == "right" else "right"
            self.say(promts["correction"])
        #Indicar que ya tomara la bag
        self.say(promts["grasp"])
        
        #Aqui podria ir una flag que indique si lo tomo o no (if true then set grasped as true)
        #aqui tiene que saber que ya lo tomo
        self.say(promts["follow"])
        grasped = "True"
        return grasped
        
        
    def go_state(self):
        reached = ""
        self.inputText = ""
        i = 0
        # una manera de saber que la persona se paro
        person = 1 #for tests 1, should be 0
        if person:
            ##3 de calls[confirm] para saber si si llegamos
            self.say(promts["car"])
            time.sleep(5)
            rospy.logdebug("speech enabled")
            self.speech_enable.publish(Bool(True))
            
            while not rospy.is_shutdown() and reached == "":
                i += 1
                time.sleep(1)
                rospy.logdebug(reached)
                if self.inputText != "":
                    reached = self.parser(calls["confirm"])
                if i % 28 == 0:
                    self.speech_enable.publish(Bool(False))
                    self.say(promts["car"])
                    self.speech_enable.publish(Bool(True))
                    i = 0
            
            rospy.logdebug("speech disabled")
            self.speech_enable.publish(Bool(False))
            if reached == "False":
                # en caso que aun no lleguen que 
                self.go_state()
            #Si si llegaron entonces paso #4
            self.say(promts["deliver"])
            #escuchar algo? o no, o nada
            reached = "True"
        return reached
            
    def return_to(self):
        returned = ""
        self.say(promts["returning"])
        #checar si ya llego y si si
        self.say(promts["home"])
        returned = "True"
        return returned
        

    def is_someone(self):
        salute_ = ""
        self.inputText = ""
        i = 0
        time.sleep(5)
        rospy.logdebug("speech enabled")
        self.speech_enable.publish(Bool(True))
        
        while not rospy.is_shutdown() and salute_ == "":
            i += 1
            time.sleep(1)
            rospy.logdebug(salute_)
            if self.inputText != "":
                salute_ = self.parser(calls["salute"])

            if i % 28 == 0:
                self.speech_enable.publish(Bool(False))
                self.is_someone()
                self.speech_enable.publish(Bool(True))
                i = 0
        
        rospy.logdebug("speech disabled")
        self.speech_enable.publish(Bool(False))
        return salute_

    def parser(self, command):
        rospy.loginfo("Receptionist is parsing")
        rospy.logwarn(self.inputText)
        time.sleep(1)
        intent = ""
        try:
            rospy.logwarn(command + self.inputText)
            intent = self.callGPT((command + " " + self.inputText))
            # rospy,logwarn(intent)
            
            rospy.loginfo("Intent: " + intent)
            
        except Exception as e:
            rospy.logwarn(e)
            self.say("I'm sorry, Could you rephrase?")
            rospy.logdebug("Failed response")
        rospy.logwarn(intent)
        
        return intent

    def input_callback(self, msg):
        self.inputText = msg.inputText
        rospy.logdebug("I heard: " + self.inputText)
            
    def callGPT(self, pr, t_max=256):
        rospy.logwarn("**************I am parsing in GPT: " + "'" + pr + "'"  + "*****************")

        response = openai.Completion.create(
            model=self.GPT_model,
            prompt=pr,
            temperature=self.GPT_temperature,
            max_tokens=t_max,
            top_p=1,
            frequency_penalty=self.GPT_frequency_penalty,
            presence_penalty=self.GPT_presence_penalty, 
        )
        print
        return response.choices[0].text

def main():
    rospy.init_node('carrybag', anonymous=True)
    rospy.loginfo("Carry my Luggage task created.")
    receptionist()


if __name__ == '__main__':
    main()