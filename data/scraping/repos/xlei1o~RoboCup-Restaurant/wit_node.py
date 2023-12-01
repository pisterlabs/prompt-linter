#!/usr/bin/env python
"""ROS node for the Wit.ai API"""
import sys
import rospy
import os
import json
from wit import Wit
import speech_recognition as sr
import openai

from wit_ros.srv import Interpret, InterpretResponse, ListenAndInterpret, ListenAndInterpretResponse
from wit_ros.msg import Outcome, Entity

import warnings
warnings.filterwarnings('ignore')


class WitRos(object):
    def __init__(self, APIKEY):
        self.APIKEY = APIKEY
        self.wit = Wit(self.APIKEY)
        self.pub = rospy.Publisher('/restaurant/stt', Outcome, queue_size=1)

    def start(self):
        rospy.Service('restaurant/wit/interpret', Interpret, self.interpret)
        rospy.Service('restaurant/wit/listen_interpret',
                      ListenAndInterpret, self.listen_and_interpret)

    def parse_response(self, response, klass):
        rospy.logdebug("Data: '{0}'".format(json.dumps(
            response, indent=4, separators=(',', ': '))))
        ros_entities = []

        if "WARNING" in response:
            rospy.logwarn("Response contains a warning: {warn}".format(
                warn=response["WARNING"]))

        outcome = None
        entities = []

        if "entities" in response:
            entities = response["entities"]
        elif "outcomes" in response:
            outcome = response["outcomes"][0]
            entities = outcome["entities"]

        for entity_name, entity_properties in entities.items():
            entity_properties = entity_properties[0]
            rospy.logdebug("Entity '{name}' has properties{prop}".format(
                name=entity_name, prop=entity_properties))

            entity = Entity(name=str(entity_name))
            if 'type' in entity_properties:
                entity.type = str(entity_properties["type"])
            if 'value' in entity_properties:
                entity.value = str(entity_properties["value"])
            if 'unit' in entity_properties:
                entity.unit = str(entity_properties["unit"])
            if 'suggested' in entity_properties:
                entity.suggested = str(entity_properties["suggested"])
            if 'confidence' in entity_properties:
                entity.confidence = float(entity_properties["confidence"])
            rospy.logdebug("Adding {ent}".format(ent=entity))
            ros_entities += [entity]

        outcome = Outcome(entities=ros_entities,
                          intent=str(outcome["intent"]) if outcome else None,
                          text=str(response["text"]))

        response = klass(msg_body=str(response),
                         outcome=outcome)
        self.pub.publish(outcome)
        return response

    def result(self, response, klass):
        rospy.logdebug("Data: '{0}'".format(json.dumps(
            response, indent=4, separators=(',', ': '))))
        ros_entities = []

        if "WARNING" in response:
            rospy.logwarn("Response contains a warning: {warn}".format(
                warn=response["WARNING"]))

        outcome = None
        entities = []
        object = 'nothing'

        if "entities" in response:
            entities = response["entities"]
        elif "outcomes" in response:
            outcome = response["outcomes"][0]
            entities = outcome["entities"]

        for entity_name, entity_properties in entities.items():
            entity_properties = entity_properties[0]
            rospy.logdebug("Entity '{name}' has properties{prop}".format(
                name=entity_name, prop=entity_properties))

            entity = Entity(name=str(entity_name))
            if 'type' in entity_properties:
                entity.type = str(entity_properties["type"])
            if 'value' in entity_properties:
                object = str(entity_properties["value"])
                entity.value = str(entity_properties["value"])
            if 'unit' in entity_properties:
                entity.unit = str(entity_properties["unit"])
            if 'suggested' in entity_properties:
                entity.suggested = str(entity_properties["suggested"])
            if 'confidence' in entity_properties:
                entity.confidence = float(entity_properties["confidence"])
            rospy.logdebug("Adding {ent}".format(ent=entity))
            ros_entities += [entity]

        outcome = Outcome(entities=ros_entities,
                          intent=str(outcome["intent"]) if outcome else None,
                          text=str(response["text"]))

        self.pub.publish(outcome)

        return object

    def interpret(self, rosrequest):
        sentence = rosrequest.sentence
        rospy.logdebug("Interpreting '{0}'".format(sentence))
        wit_response = self.wit.message(sentence)
        rospy.logdebug("WitResponse: {0}".format(wit_response))

        return self.parse_response(wit_response, InterpretResponse)

    def listen_and_interpret(self, rosrequest):
        rospy.logdebug("About to record audio")

        r = sr.Recognizer()
        # with sr.Microphone() as source:
        #     rospy.loginfo("Say something!")
        #     audio = r.listen(source)

        # process = subprocess.Popen(
        #     "roslaunch audio_capture capture_wave.launch", shell=True, cwd="./")
        rospy.loginfo("say something")
        os.system("rec output.wav trim 0 05")

        with sr.AudioFile("/home/yi/.ros/output.wav") as source:
            audio = r.record(source)

        try:
            sentence = r.recognize_wit(audio, key=self.APIKEY)
            rospy.loginfo(sentence)
        except sr.UnknownValueError:
            rospy.loginfo("Wit.ai could not understand audio")
            return ListenAndInterpretResponse("nothing")
        except sr.RequestError as e:
            rospy.loginfo(
                "Could not request results from Wit.ai service; {0}".format(e))
            return ListenAndInterpretResponse("request fail")

        # print(response)

        wit_response = self.wit.message(sentence)
        rospy.logdebug("WitResponse: {0}".format(wit_response))
        return self.result(wit_response, ListenAndInterpretResponse)


if __name__ == "__main__":
    print(sys.version)
    print(sys.version_info)
    rospy.init_node("wit_ros", log_level=rospy.INFO)

    wr = WitRos("GG424C7762FEFSTDI2BWR3MZQMO6UPUP")

    wr.start()

    rospy.spin()
