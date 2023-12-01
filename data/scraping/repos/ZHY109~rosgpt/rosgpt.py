#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
import threading
import openai
import rospy
from std_msgs.msg import String


class RosGPT:
    def __init__(self):
        openai.api_key = ""
        self.Response_pub = rospy.Publisher('Response', String, queue_size=10)
        rospy.Subscriber('Question', String, self.callback)
        self.lock = threading.Lock()


    def getResponce(self,data):
        self.lock.acquire()
        prompt = data.data
        completions = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=2048,
            n=1,
            stop=None,
            temperature=0.9,
        )
        message = completions.choices[0].text
        self.Response_pub.publish(message)
        self.lock.release()
        return message

    def callback(self, data):
        rospy.loginfo("Waiting for server")
        responce = self.getResponce(data)
        rospy.loginfo("\033[94m"+responce+"\033[0m")


    def run(self):
        rospy.init_node('RosGPT')
        rate = rospy.Rate(1)
        rospy.loginfo('Staring listening to the question')
        while not rospy.is_shutdown():
            rate.sleep()


if __name__ == '__main__':
    try:
        rosgpt = RosGPT()
        rosgpt.run()
    except rospy.ROSInterruptException:
        pass



