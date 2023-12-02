#!/usr/bin/env python
#===================================================================
# This is the AI Respond Node.
#
# Subscribes To:
# - recognized_speech
# - recognized_face
# - speech_info
#
# Publishes To:
# - say
# - target_face
# - control
# - joy
#
# Responses:
# 1) The node will generate text responses to recognized_speech,
#    publishing them to the say topic.
# 2) The node will attempt to match names and faces, saving the
#    pairings to an in-memory knowledge base.
# 3) If a new face is recognized, the node will generate
#    a greeting and attempt to get the person's name.
# 4) On receiving a kill signal, the node will save
#    the currently loaded knowledge base back to a file
#    for later reuse.
#
# AIML is used for the basic chat functionality.
#
# Copyright 2015, IEEE ENCS Humanoid Robot Project
#===================================================================

from __future__ import division
from __future__ import print_function
from time import time
from vision.msg import DetectedFace
from vision.msg import NamedFace
from vision.msg import RecognizedFace
from vision.msg import TargetFace
import collections
from datetime import date
import json
import random
import sys
import numpy as np
import openai
import os.path
import pickle
import rospy
from std_msgs.msg import String
import atexit
import datetime

kb_file = "kb.p"      # default name for the knowledge base
faces_file = "faces.txt"  # default name for the face-name mapping file


def read_bot_info_from_file(filename="who_is_ken.txt"):
    with open(filename, 'r') as f:
        return f.read().strip()


def read_api_key_from_file():
    with open(os.path.expanduser('~/openai_api_key.txt'), 'r') as f:
        return f.read().strip()


class AIRespondNode(object):
    def __init__(self):
        global min_match, delta_xy_px, delta_t_ms, max_recent_face_time_ms
        global enable_learning, enable_proximity_match
        rospy.init_node('llm_respond_node')

        openai.api_key = read_api_key_from_file()

        self.faces_in_view = []
        self.tracked_faces = {}  # map keyed by track_id, value of Face

        bot_info = read_bot_info_from_file()
        bot_info += "\nToday's date is " + str(date.today())
        self.messages = [
                { "role": "instructions", "content": bot_info }  # tell the bot about itself
        ]

        # maximum number of messages to track in the conversation
        self.max_messages = int(self.get_param('~max_messages', '7'))

        self.session_id = "test1234"
        self.is_speaking = False
        enable_learning = True
        enable_proximity_match = False

        speech_topic = self.get_param('~in_speech', '/recognized_speech')
        detected_face_topic = self.get_param('~in_detected_face', '/detected_face')
        face_topic = self.get_param('~in_face', '/recognized_face')
        speech_info_topic = self.get_param('~in_speech_info', '/speech_info')
        say_topic = self.get_param('~out_response', '/say')
        target_topic = self.get_param('~out_target', '/target_face')
        name_topic = self.get_param('~name', '/named_face')
        self.max_target_hold_sec = float(self.get_param('~max_target_hold_sec', '30.0'))
        min_match = int(self.get_param('~min_match', '4'))
        delta_xy_px = int(self.get_param('~delta_xy_px', '20'))
        # TODO perhaps delta_t_ms and max_recent_face_time_ms should be one parameter?
        delta_t_ms = int(self.get_param('~delta_t_ms', '2000'))
        max_recent_face_time_ms = int(self.get_param('~max_recent_face_time_ms', '2000'))

        self.load_kb()

        self.pub = rospy.Publisher(say_topic, String, queue_size=1)
        self.target_face_pub = rospy.Publisher(target_topic, TargetFace, queue_size=1)
        self.named_face_pub = rospy.Publisher(name_topic, NamedFace, queue_size=5)
        rospy.Subscriber(speech_topic, String, self.on_recognized_speech)
        rospy.Subscriber(detected_face_topic, DetectedFace, self.on_detected_face)
        rospy.Subscriber(face_topic, RecognizedFace, self.on_recognized_face)
        rospy.Subscriber(speech_info_topic, String, self.on_speech_info)


    def get_param(self, param_name, param_default):
        value = rospy.get_param(param_name, param_default)
        rospy.loginfo('Parameter %s has value %s', rospy.resolve_name(param_name), value)
        return value


    def load_kb(self):
        # if kb found, load file to memory
        # otherwise just create a new dict
        if os.path.exists(kb_file):
            with open(kb_file, "rb") as f:
                self.kb = pickle.load(f)
            for pred, value in self.kb.items():
                self.setkb(pred, value)
            # clear the input stack in case the program died in the middle of processing
            self.setkb("_inputStack", [])
            self.setkb("target_face", None)
            self.setkb("name", "")
            self.setkb("look_at_name", "")
        else:
            self.kb = self.bot.getSessionData(self.session_id)
            self.setkb("faces", Faces())

        # load a map of encounter ids to face name frequencies
        if os.path.exists(faces_file):
            with open(faces_file, "rb") as f:
                m = json.load(f)
            # the JSON save process converts the encounter ids to strings; make them ints again
            self.faces_map = {}
            for k in m.keys():
                self.faces_map[int(k)] = m[k]
        else:
            self.faces_map = {}


    def getkb(self, key):
        '''
        convenience function to access bot predicates
        '''
        return None  # TODO handle bot predicates with LLM
        #result = self.bot.getPredicate(key, self.session_id)
        #if key == "target_face" and result == "":
        #    result = None
        #return result


    def setkb(self, key, value):
        '''
        convenience function to modify bot predicates
        '''
        # self.bot.setPredicate(key, value, self.session_id)
        pass  # TODO handle bot predicates with LLM


    def on_recognized_speech(self, msg):
        self.respond_to(msg.data)
        # ---- HACK ----
        # update the target face timestamp if we get a recognized speech message
        # while not perfect, this is an attempt to avoid interrupting conversations
        # when a new face is recognised
        self.setkb("target_face_timestamp_s", time())


    def on_speech_info(self, msg):
        self.is_speaking = (msg.data == "start_speaking")
        if self.is_speaking:
            rospy.loginfo(rospy.get_caller_id() + ": speaking")
        else:
            rospy.loginfo(rospy.get_caller_id() + ": done speaking")


    def respond_to(self, heard_text):
        if not self.is_speaking:
            rospy.loginfo(rospy.get_caller_id() + ": I heard: %s", heard_text)
            self.update_count_of_faces_in_view()

            utterance = self.generate_response(heard_text)

            if utterance != "":
                self.pub.publish(utterance)
            rospy.loginfo(rospy.get_caller_id() + ": I said: %s", utterance)
        else:
            rospy.loginfo(rospy.get_caller_id() + ": still speaking, can't respond to: %s", heard_text)


    def generate_response(self, prompt):
        self.messages.append({ "role": "user", "content": prompt })

        use_davinci = True
        if use_davinci:
            if len(self.messages) <= self.max_messages:
                lines = [m["role"] + ": " + m["content"] for m in self.messages]
            else:
                lines = [m["role"] + ": " + m["content"] for m in self.messages[0:1]] + \
                        [m["role"] + ": " + m["content"] for m in self.messages[-self.max_messages:]]
            script = '\n'.join(lines + ["assistant: "])

            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=script,
                max_tokens=50,
                n=1,
                stop='user: ',
                temperature=0.7,
            )

            reply = response.choices[0].text.strip()
            prefixes = ["ken: ", "assistant: "]
            while any(reply.lower().startswith(prefix) for prefix in prefixes):
                for prefix in prefixes:
                    if reply.lower().startswith(prefix):
                        reply = reply[len(prefix):].lstrip()
        else:
            # ref: https://stackoverflow.com/questions/74711107/openai-api-continuing-conversation
        
            # Generate a response using ChatGPT
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=self.messages,
                max_tokens=50,
                temperature=0.5,
                n=1
            )

            reply = response.choices[0]["message"]["content"]

        self.messages.append({ "role": "assistant", "content": reply })
        return reply


    def on_detected_face(self, detected_face):
        # each time a face is detected,
        # update the list of recent detected faces and
        # update the facesinview property with the max count of faces from either camera within
        # the last time period

        to_remove = []
        for f in self.faces_in_view:
            # if new face overlaps with position of old face, remove the old one
            if detected_face.track_id == f.f.track_id:
                to_remove.append(f)

        for f in to_remove:
            self.faces_in_view.remove(f)

        # TODO merge the logic in this function with usage of self.tracked_faces
        self.faces_in_view.append(FaceInView(detected_face, time()))


    def update_count_of_faces_in_view(self):
        now = time()
        to_remove = []
        counts = {}

        # remove faces which haven't been seen recently
        for f in self.faces_in_view:
            #rospy.loginfo(str(now - f.stamp))
            if now - f.stamp >= max_recent_face_time_ms / 1000.:
                to_remove.append(f)
        
        for f in to_remove:
            self.faces_in_view.remove(f)

        in_view = len(self.faces_in_view)
        self.setkb("facesinview", str(in_view))
        #rospy.loginfo(str(self.faces_in_view))
        rospy.loginfo('%s faces in view', self.getkb("facesinview"))


    def on_recognized_face(self, recognized_face):
        global enable_learning, enable_proximity_match
        faces = self.getkb("faces")
        face = faces.find(recognized_face, self)
        if not face:
            face = faces.add(recognized_face)
            rospy.loginfo('Adding new recognized face %s', str(sorted(recognized_face.encounter_ids)))
            rospy.loginfo('Face count is %d', len(faces))
        else:
            face.update(recognized_face.encounter_ids)

        if recognized_face.track_id not in self.tracked_faces:
            self.tracked_faces[recognized_face.track_id] = Face(recognized_face.track_id)
        tracked_face = self.tracked_faces[recognized_face.track_id]
        # add the recognized face encounter ids to the track encounter ids
        tracked_face.update(recognized_face.encounter_ids)
        # find most likely name given track encounter ids + recognized face encounter ids, set as track name
        names = Names.for_face(tracked_face, self.faces_map)
        # publish NamedFace message with track id and name (include all matching names and confidence)
        self.publish_named_face(recognized_face, names)
        if len(names.names) > 0:
            # set the face name to the most common name for the recognized face encounter ids
            tracked_face.name = names.names[0]
            # update faces_map to increment count for name for recognized_face encounter ids <-- this is the permanent memory
            if tracked_face.name is not None:
                for encounter_id in recognized_face.encounter_ids:
                    m = self.faces_map
                    if encounter_id not in m:
                        m[encounter_id] = { tracked_face.name: 1 }
                    elif tracked_face.name in m[encounter_id]:
                        m[encounter_id][tracked_face.name] += 1
                    else:
                        m[encounter_id][tracked_face.name] = 1
        # TODO Clean up the following when old mechanism is removed
        # replace face with the tracked face
        face = tracked_face

        # store the face for matching by proximity
        if enable_proximity_match:
            faces.append_recent_face(RecentFace(face, recognized_face))

        target_face, greeting = self.update_target_face(face, recognized_face)

        # if the target face has provided a name, store it
        converser_name = self.getkb("name")
        if converser_name:
            if converser_name != target_face.name:
                # correct the name of the target face
                for encounter_id in target_face.encounter_ids:
                    if encounter_id in self.faces_map and target_face.name in self.faces_map[encounter_id]:
                        self.faces_map[encounter_id][converser_name] = self.faces_map[encounter_id][target_face.name]
                        del self.faces_map[encounter_id][target_face.name]
                    else:
                        self.faces_map[encounter_id] = { converser_name: 1 }
                target_face.name = converser_name

        look_at_name = self.getkb("look_at_name")
        if look_at_name:
            # Do we know that name?
            candidates = [tf for tf in self.tracked_faces.items() if str(tf[1].name).lower() == str(look_at_name).lower()]
            if len(candidates) == 0:
                self.respond_to("ken has not seen " + look_at_name)
            else:
                # Is that name in view?
                tracks_in_view = [f.f.track_id for f in self.faces_in_view]
                candidates = [tf for tf in candidates if tf[0] in tracks_in_view]
                if len(candidates) == 0:
                    self.respond_to("ken does not see " + look_at_name)
                else:
                    self.set_target_face(candidates[-1][1])  # target last known occurence
            self.setkb("look_at_name", "")
               
        # if we have a name for both target and converser
        #converser_name = self.getkb("name")
        #if target_face.name:
        #    # if the target is different from the current converser's name
        #    if not converser_name:
        #    	# update converser name to match target face name
        #	# set in ken-recognize.aiml rather than here
        #	rospy.loginfo('Recognized face of %s', str(target_face.name))
        #	# trigger response which includes converser's name
        #	#self.respond_to("recognize " + target_face.name)
        #    elif target_face.name != converser_name:
        #    	# correct the target name to the converser name
        #	target_face.name = converser_name
        #	for encounter_id in target_face.encounter_ids:
        #	    self.faces_map[encounter_id] = { target_face.name: 1 }
    #	    else:
    #		rospy.loginfo('got target=%s and convers=%s', str(target_face.name), converser_name)
        #elif converser_name and converser_name.strip():
        #    # we have a converser name, but no target name.  Assume that
        #    # the converser name applies to the target, since we cleared
        #    # the converser name when the target face was last changed.
        #    target_face.name = converser_name
        #    for encounter_id in target_face.encounter_ids:
        #    	self.faces_map[encounter_id] = { target_face.name: 1 }
        #    rospy.loginfo('Associated name %s to face %s', converser_name, str(sorted(target_face.encounter_ids)))
        #elif greeting:
        #    # we know neither target face nor converser's name
        #    # generate a generic greeting to solicit the person's name
        #    rospy.loginfo('Recognized face of stranger %s', str(sorted(target_face.encounter_ids)))
        #    #self.respond_to("hello")
    #	else:
    #	    rospy.loginfo('no greeting for %s', str(target_face.name))

        self.publish_target_face(target_face, face, recognized_face)


    def update_target_face(self, face, recognized_face):
        target_face = self.getkb("target_face")
        #rospy.loginfo('update_target_face called: face=%s, target_face=%s', str(sorted(face.encounter_ids)), str(target_face))
        greeting = False
        if not target_face:		     # no target face set yet
            rospy.loginfo('target face is null')
            # set target face if not set
            target_face = self.set_target_face(face)
            greeting = True
        elif target_face != face and target_face.name != face.name:  # recognized a different face than the target
            #rospy.loginfo('Target %d, recognized %d: Face of %s, %s', target_face.id, face.id, face.name, str(sorted(face.encounter_ids)))
            # get the time since the last match to the target face
            last_seen_time = self.getkb("target_face_timestamp_s")
            time_since_last = time() - last_seen_time
            # change the target with a probability that increases with the time
            # in other words, if we haven't seen the target face in a while, it
            # becomes increasingly likely that we will switch the target to the
            # current recognized face.
            r = random.random()
            p = (time_since_last / self.max_target_hold_sec)**2
            if r < p:
                rospy.loginfo('Switching target face from %d: %s to %d: %s after %g seconds with %g probability', target_face.id, str(target_face.name), face.id, str(face.name), time_since_last, p)
                if not face.name or target_face.name != face.name: # only greet if the name changes
                    greeting = True
                target_face = self.set_target_face(face)
        else:				     # recognized current target face
            #rospy.loginfo('target face update timestamp')
            # update the time that the current target was last recognized
            target_face = self.set_target_face(face)

        return target_face, greeting


    def set_target_face(self, face):
        target_face = face
        previous_target = self.getkb("target_face")
        self.setkb("target_face", target_face)
        self.setkb("target_face_timestamp_s", time())
        if target_face != previous_target:
            # set the converser name to the new target
            if target_face.name:
                self.setkb("name", target_face.name)
            else:
                self.setkb("name", "")
            rospy.loginfo('Set target face to %d: %s, %s', target_face.id, str(target_face.name), str(sorted(target_face.encounter_ids)))
        return target_face


    def publish_target_face(self, target_face, face, recognized_face):
        # publish target face message
        target_face_msg = TargetFace()
        target_face_msg.header = recognized_face.header
        target_face_msg.track_id = target_face.id
        target_face_msg.track_color = recognized_face.track_color  # TODO get the correct color for the target face
        target_face_msg.x = recognized_face.x
        target_face_msg.y = recognized_face.y
        target_face_msg.w = recognized_face.w
        target_face_msg.h = recognized_face.h
        target_face_msg.encounter_ids = list(target_face.encounter_ids)
        target_face_msg.name = str(target_face.name)
        target_face_msg.id = target_face.id
        target_face_msg.recog_name = str(face.name)
        target_face_msg.recog_id = face.id
        self.target_face_pub.publish(target_face_msg)


    def publish_named_face(self, recognized_face, names):
        # publish named face message
        named_face_msg = NamedFace()
        named_face_msg.header = recognized_face.header
        named_face_msg.track_id = recognized_face.track_id
        named_face_msg.names = names.names
        named_face_msg.confs = names.confs
        self.named_face_pub.publish(named_face_msg)


    def save_kb(self):
        global enable_learning
        rospy.loginfo(rospy.get_caller_id() + ": I received a kill signal")
        if enable_learning:
            # TODO implement storing information captured from the conversation
            rospy.loginfo(rospy.get_caller_id() + ": Not implemented yet")
            # rospy.loginfo(rospy.get_caller_id() + ": Writing KB to %s", kb_file)
            # self.kb = self.bot.getSessionData(self.session_id)
            # with open(kb_file, "wb") as f:
            #     pickle.dump(self.kb, f)
            # faces = self.getkb("faces")
            # for i, face in enumerate(faces.faces):
            #     rospy.loginfo("%d: Face %d of %s, %s", i, face.id, face.name, str(sorted(face.encounter_ids)))
            # # export the face encounter to name frequence mapping data
            # with open(faces_file, "wb") as f:
            #     json.dump(self.faces_map, f)
        rospy.loginfo(rospy.get_caller_id() + ": Exiting AI Respond Node")

    def run(self):
        rospy.spin()


class Rect(object):
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


class FaceInView(object):
    '''
    Helper class for tracking faces in view.
    '''
    def __init__(self, detected_face, timestamp):
        self.f = detected_face
        self.stamp = timestamp


class RecentFace(object):
    '''
    Helper class to hold a recently recognized face along with its coordinates and timestamp.
    '''
    def __init__(self, face, recognized_face):
        self.face = face
        self.stamp = recognized_face.header.stamp
        self.frame_id = recognized_face.header.frame_id
        self.rect = Rect(recognized_face.x, recognized_face.y, recognized_face.w, recognized_face.h)


class Faces(object):
    '''
    Keep track of known faces and provide an interface for looking up recognized faces.
    A known face is recognizable and has a known name.
    A recognized, unknown face will solicit a query from the robot to determine the
    name to put with the face.
    '''
    def __init__(self):
        self.faces = []
        self.recent_faces = []
        self.next_id = 0


    def __len__(self):
        return len(self.faces)


    def find(self, recognized_face, node):
        global enable_proximity_match
        face = self.find_similar_face(recognized_face, node)
        # TODO try to enable proximity match to avoid duplicate faces
        #if not face:
        #    face = self.find_recent_face(recognized_face)
        #if face and enable_proximity_match:
        #    self.recent_faces.append(RecentFace(face, recognized_face))
        return face


    def append_recent_face(self, recent_face):
        global enable_proximity_match
        if enable_proximity_match:
            self.recent_faces.append(recent_face)


    # TODO not currently used - either use or remove
    def find_recent_face(self, recognized_face):
        global delta_xy_px, delta_t_ms, max_recent_face_time_ms, enable_proximity_match
        if not enable_proximity_match:
            return None

        # construct a list of recent faces which are equivalent to the recognized face
        # due to their proximity in space and time
        faces = set()
        if not hasattr(self, 'recent_faces'):
            self.recent_faces = []
        for f in self.recent_faces:
            delta_x = abs(f.rect.x - recognized_face.x)
            delta_y = abs(f.rect.y - recognized_face.y)
            delta_w = abs(f.rect.w - recognized_face.w)
            delta_h = abs(f.rect.h - recognized_face.h)
            delta_t = abs((f.stamp - recognized_face.header.stamp).to_sec() * 1000)
            if delta_x <= delta_xy_px and delta_y <= delta_xy_px and delta_w <= delta_xy_px and delta_h <= delta_xy_px and delta_t <= delta_t_ms:
                print(recognized_face.header.frame_id, delta_x, delta_y, delta_w, delta_h, delta_t, f.face.id, f.face.name)
                faces.add(f.face)
        
        # remove the original faces from the memory and
        # add a single new face which combines the original faces
        face = self.combine_faces(faces)

        # purge faces which are too old to keep
        size = len(self.recent_faces)
        self.recent_faces = [f for f in self.recent_faces if abs((f.stamp - recognized_face.header.stamp).to_sec() * 1000) <= max_recent_face_time_ms]
        if size > 0:
            print("recent faces reduced from", size, "to", len(self.recent_faces))

        return face


    # TODO this code needs some refactoring attention
    def find_similar_face(self, recognized_face, node):
        global min_match
        ids = set(recognized_face.encounter_ids)
        #rospy.loginfo("finding faces similar to " + str(sorted(ids)))
        
        # before searching for a matching face, first use information from this
        # recognized face to consolidate duplicate faces.  If two known faces
        # overlap with a single recognized face by more than min_match (the
        # identity recognition threshold) then they are duplicates and can be
        # merged.
        self.consolidate_duplicate_faces(ids, node)

        # match the recognized face to known faces
        top_overlap = min_match # init to min_match to exclude low matches
        top_faces = []
        for face in self.faces:
            overlap = len(set(face.encounter_ids) & ids)
            if overlap >= top_overlap:
                #rospy.loginfo("top overlap with " + str(face.id) + " is " + str(overlap))
                # track equivalent top matches
                if overlap > top_overlap:
                    top_faces[:] = []	    # clear list if better match found
                top_faces.append(face) 
                top_overlap = overlap
        #rospy.loginfo("found " + str(len(top_faces)) + " top faces: " + str([f.id for f in top_faces]))
        top_face = None

        # look for a face with a name first
        for face in top_faces:
            if face.name:
                top_face = face
                break
        #if top_face:
        #    rospy.loginfo("found top face with name: " + str(top_face.id) + ", " + str(top_face.name))
        #else:
        #    rospy.loginfo("no top face with name found")

        # if no face with a name, choose the most general match
        longest = top_overlap
        if not top_face:
            for face in top_faces:
                if len(face.encounter_ids) >= longest: # use >= to ensure at least one match
                    top_face = face
                    longest = len(face.encounter_ids)
        #if top_face:
        #    rospy.loginfo("returning top face: " + str(top_face.id) + ", " + str(top_face.name) + ", longest=" + str(longest))
        #else:
        #    rospy.loginfo("no similar face found for min_match=" + str(min_match))
        return top_face  # may be None if no match was > min_match threshold


    def consolidate_duplicate_faces(self, ids, node):
        # first, combine all faces with at least min_match overlap to the recognized face
        # as they all represent the same person; keep the most common name, if any
        same_faces = []
        name_count = collections.defaultdict(int)
        # collect all faces for the same person
        #rospy.loginfo("collecting faces of same person from " + str(len(self.faces)) + " faces")
        for face in self.faces:
            overlap = len(set(face.encounter_ids) & ids)
            #if overlap > 0:
            #	rospy.loginfo("overlap with " + str(face.id) + " is " + str(overlap))
            if overlap >= min_match:
                same_faces.append(face)
                if face.name is not None:
                    name_count[face.name] += 1
        # for each face name, group the most common faces to it
        # First, group all faces with the same name
        faces_by_name = self.group_faces_by_name(same_faces)
        # Second, add closest matching no-name faces to each named face
        # determine the most common name of the collected faces
        if len(name_count) > 0:
            names = [n for n in name_count.keys()]
            counts = [name_count[n] for n in names]
            name = names[np.argmax(counts)]
        else:
            name = None
        #rospy.loginfo("determined most common name is " + str(name))
        # combine the faces into one object under a single name
        self.combine_faces2(same_faces, name, node)


    def group_faces_by_name(self, same_faces):
        faces_by_name = collections.defaultdict(list)
        for face in same_faces:
            if face.name is not None:
                faces_by_name[face.name].append(face)
        return faces_by_name


    def combine_faces2(self, same_faces, name, node):
        if len(same_faces) > 1:
            rospy.loginfo("found " + str(len(same_faces)) + " duplicates")
            face = same_faces[0]  # keep the first face
            face.name = name      # assign the name
            rospy.loginfo("keeping face " + str(face.id) + " with name " + str(face.name))
            # merge the other faces into it
            for other_face in same_faces[1:]:
                for i in other_face.encounter_id_hist:
                    if i in face.encounter_id_hist:
                        face.encounter_id_hist[i] = max(face.encounter_id_hist[i], other_face.encounter_id_hist[i])
                    else:
                        face.encounter_id_hist[i] = other_face.encounter_id_hist[i]
                # if removed face is the target, then update target to the kept face
                target_face = node.getkb("target_face")
                if target_face == other_face:
                    node.setkb("target_face", face)
                    rospy.loginfo("replaced target face " + str(target_face.id) + " with face " + str(face.id))
                rospy.loginfo("removing face " + str(other_face.id) + " with name " + str(other_face.name))
                # remove the other faces from the faces list
                self.faces.remove(other_face)
            face.encounter_ids = set(face.encounter_id_hist.keys())
            rospy.loginfo("merged face " + str(face.id) + " encounter ids: " + str(sorted(face.encounter_ids)))
            return face
        return None


    def add(self, recognized_face):
        global enable_learning
        face = Face(self.next_face_id())
        if enable_learning:
            face.update(recognized_face.encounter_ids)
            self.faces.append(face)
        return face


    def next_face_id(self):
        next_id = self.next_id
        self.next_id += 1
        return next_id


    # TODO not currently used - either use or remove
    def combine_faces(self, faces):
        if len(faces) == 0:
            return None
        faces = list(faces)  # convert set to list for easy indexing
        #rospy.loginfo('Combining faces %s', str([f.id for f in faces]))
        if len(faces) == 1:
            return faces[0]
        face = faces[0]  # TODO determine the 'best' face to keep from the list
        for f in faces[1:]:
            # remove original face
            if f in self.faces:
                self.faces.remove(f)
            # add the encounter ids - do this one face at a time to 
            # rebuild a histogram of common ids
            face.update(f.encounter_ids)
            if face.name is None and f.name is not None:
                face.name = f.name
        return face


class Face(object):
    def __init__(self, id):
        self.id = id
        self.encounter_ids = set()
        self.encounter_id_hist = dict()
        self.name = None


    def update(self, encounter_ids):
        '''
        Update the histogram of encounter ids for this face and reset
        the encounter ids set to the histogram's key set.

        Note: Although the histogram is maintained here, it is not currently used.
        '''
        global enable_learning
        if enable_learning:
            for id in encounter_ids:
                if id not in self.encounter_id_hist:
                    self.encounter_id_hist[id] = 0
                self.encounter_id_hist[id] += 1

            self.encounter_ids = set(self.encounter_id_hist.keys())


class Names(object):

    @staticmethod
    def for_face(tracked_face, face_names_map):
        # determine matching names for the face from the face names map
        encounter_ids = tracked_face.encounter_ids
        name_freqs = {}
        for encounter_id in encounter_ids:
            if encounter_id in face_names_map:
                counts_by_name = face_names_map[encounter_id]
                #rospy.loginfo("Counts by name for %d: %s", encounter_id, str(counts_by_name))
                total = sum(counts_by_name.values())
                names = list(counts_by_name.keys())
                freqs = [counts_by_name[n] / total for n in names]
                for name, freq in zip(names, freqs):
                    if name in name_freqs:
                        name_freqs[name] += freq
                    else:
                        name_freqs[name] = freq
            #else:
            #	rospy.loginfo("Encounter id %d is not in face names map", encounter_id)
            #	rospy.loginfo("keys = %s", str(face_names_map.keys()))
        names = list(name_freqs.keys())
        if len(names) > 0:
            freqs = [name_freqs[n] / len(encounter_ids) for n in names]
            # return a Names object with the names and confidences ordered from high to low
            n, c = zip(*sorted(zip(names, freqs), key=lambda x: x[1], reverse=True))
            return Names(list(n), list(c))
        else:
            return Names([], [])


    def __init__(self, names, confidences):
        self.names = names
        self.confs = confidences


if __name__ == "__main__":
    try:
        ai = AIRespondNode()

        atexit.register(ai.save_kb)
        ai.run()
    except rospy.ROSInterruptException:
        pass

