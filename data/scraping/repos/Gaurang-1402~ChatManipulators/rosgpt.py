#!/usr/bin/env python3
# This file is part of rosgpt package.
#
# Copyright (c) 2023 Anis Koubaa.
# All rights reserved.
#
# This work is licensed under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 4.0
# International Public License. See https://creativecommons.org/licenses/by-nc-sa/4.0/ for details.

import os
import json
import openai
import rclpy
import threading
from rclpy.node import Node
from std_msgs.msg import String
from flask import Flask, request, send_from_directory, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS
import pyttsx3  # pip install pyttsx3 #you need to install libespeak1 on Ubuntu # sudo apt-get install libespeak1
from rclpy.executors import SingleThreadedExecutor
import subprocess

from ament_index_python import get_package_share_directory

# Instantiate a Flask application object with the given name
app = Flask(__name__)

# Enable Cross-Origin Resource Sharing (CORS) for the Flask app
CORS(app)

# Create an API object that wraps the Flask app to handle RESTful requests
api = Api(app)

#You must add OPENAI_API_KEY as an environment variable
#In Ubuntu: echo 'export OPENAI_API_KEY=your_api_key' >> ~/.bashrc
# Get the API key from the environment variable. 
openai_api_key = os.getenv('OPENAI_API_KEY')
#print(openai_api_key)

# Now you can use the openai_api_key variable to authenticate with the OpenAI API


# Initialize a threading lock for synchronizing access to shared resources
# when multiple threads are involved
spin_lock = threading.Lock()

# Initialize the Text-to-Speech (TTS) engine using the pyttsx3 library
# you need to install the following dependencies 
#       sudo apt-get install libespeak1
#       pip3 install pyttsx3
tts_engine = pyttsx3.init()

# Create a separate threading lock for synchronizing access to the TTS engine
tts_lock = threading.Lock()




class ROSGPTNode(Node):
    def __init__(self):
        """
        Initialize the ROSGPTNode class which is derived from the rclpy Node class.
        """
        # Call the superclass constructor and pass the name of the node
        super().__init__('chatgpt_ros2_node')
        # Create a publisher for the 'voice_cmd' topic with a message queue size of 10
        self.publisher = self.create_publisher(String, 'voice_cmd', 10)

    def publish_message(self, message):
        """
        Publish the given message to the 'voice_cmd' topic.
        Args:
            message (str): The message to be published.
        """
        msg = String() # Create a new String message 
        msg.data = message # Convert the message to a JSON string and set the data field of the message
        self.publisher.publish(msg) # Publish the message using the publisher 
        #print('message Published: ', message) # Log the published message
        #print('msg.data Published: ', msg.data) # Log the published message
        
        



def process_and_publish_chatgpt_response(chatgpt_ros2_node, text_command, chatgpt_response, use_executors=True):
    """
    Process the chatbot's response and publish it to the 'voice_cmd' topic.

    Args:
        chatgpt_ros2_node (ROSGPTNode): The ROS2 node instance.
        text_command (str): The text command received from the user.
        chatgpt_response (str): The response from the chatbot.
        use_executors (bool, optional): Flag to indicate whether to use SingleThreadedExecutor. Defaults to True.
    """
    chatgpt_ros2_node.publish_message(chatgpt_response) # Publish the chatbot's response using the ROS2 node
    # If use_executors flag is True, use SingleThreadedExecutor
    if use_executors:
        executor = SingleThreadedExecutor()# Create a new executor for each request 
        executor.add_node(chatgpt_ros2_node) # Add the node to the executor
        executor.spin_once()#  Spin the executor once
        executor.remove_node(chatgpt_ros2_node) # Remove the node from the executor
    # If use_executors flag is False, use spin_lock to synchronize access
    else:
        with spin_lock:
            rclpy.spin_once(chatgpt_ros2_node)



class ROSGPTProxy(Resource):
    """
    A class derived from flask_restful.Resource, responsible for handling incoming HTTP POST requests.
    """

    def __init__(self, chatgpt_ros2_node):
        """
        Initialize the ROSGPTProxy class with the given ROS2 node.

        Args:
            chatgpt_ros2_node (ROSGPTNode): The ROS2 node instance.
        """
        self.chatgpt_ros2_node = chatgpt_ros2_node

    def askGPT(self, text_command):
        """
        Send a text command to the GPT-3 model and receive a response.
        Args:
            text_command (str): The text command to be sent to the GPT-3 model.
        Returns:
            str: The response from the GPT-3 model as a JSON string.
        """
        # Create the GPT-3 prompt with example inputs and desired outputs
        prompt = '''
        Consider the following ontology:
        {"action": "center", "params": {}}
        {"action": "random", "params": {}}
        {"action": "move_joint", "params": {"joint": joint, "angle": angle, "speed": speed}}

        The 'joint' parameter can take values "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint" to indicate the joint to move. The 'angle' parameter represents the target angle for the joint in radians, and 'speed' is the speed at which to move the joint in radians per second. The 'angle' value must be between -6.13 and 6.13 for all joints except 'elbow_joint', which must be between -2.99 and 2.99. Here are some examples.

        If speed is not given in the prompt, it is assumed to be 0.5 radians per second.
        All numerical answers should be in float form.

        All joint values are between -6.13 to 6.13
        except elbow_joint which has joint values between -2.99 to 2.99 

        Prompt: "Move the shoulder_pan_joint to angle 0.7 radians at a speed of 0.2 radians per second."
        Returns: {"action": "move_joint", "params": {"joint": "shoulder_pan_joint", "angle": 0.7, "speed": 0.2}}

        Prompt: "Move the elbow_joint to angle 1.3 radians."
        Returns: {"action": "move_joint", "params": {"joint": "elbow_joint", "angle": 1.3, "speed": 0.5}}

        Prompt: "Move the wrist_1_joint to angle 2.2 radians at a speed of 0.3 radians per second."
        Returns: {"action": "move_joint", "params": {"joint": "wrist_1_joint", "angle": 2.2, "speed": 0.3}}

        Prompt: "Move the robot to the center position."
        Returns: {"action": "center", "params": {}}

        Prompt: "Move the robot to a random joint configuration."
        Returns: {"action": "random", "params": {}}

        You will be given human language prompts, and you need to return a JSON conformant to the ontology. Any action not in the ontology must be ignored.

        Examples in other languages:

        German:
        Prompt: "Bewegen Sie das shoulder_pan_joint auf einen Winkel von 0,7 Radiant mit einer Geschwindigkeit von 0,2 Radiant pro Sekunde."
        Returns: {"action": "move_joint", "params": {"joint": "shoulder_pan_joint", "angle": 0.7, "speed": 0.2}}

        Japanese:
        Prompt: "Elbow_joint o 1.3 radian ni ido sase, speed wa 0.5 radian per second desu."
        Returns: {"action": "move_joint", "params": {"joint": "elbow_joint", "angle": 1.3, "speed": 0.5}}

        Hindi:
        Prompt: "Wrist_1_joint ko 2.2 radians par le jao, speed 0.3 radians per second ke saath."
        Returns: {"action": "move_joint", "params": {"joint": "wrist_1_joint", "angle": 2.2, "speed": 0.3}}
        '''


        prompt = prompt+'\nprompt: '+text_command
        #print(prompt) #for testing
        

        # Create the message structure for the GPT-3 model
        messages = [
            {"role": "system", "content": "You are a robot brain that translates human natural language to robot control commands in json."},
            {"role": "user", "content": prompt}
        ]

        # Try to send the request to the GPT-3 model and handle any exceptions
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
            )
        except openai.error.InvalidRequestError as e:
            print(f"Error: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None
        
        # Extract the GPT-3 model response from the returned JSON
        chatgpt_response = response.choices[0].message['content'].strip()
        #print(chatgpt_response)
        # Find the start and end indices of the JSON string in the response
        start_index = chatgpt_response.find('{')
        end_index = chatgpt_response.rfind('}') + 1
        # Extract the JSON string from the response
        json_response_dict = chatgpt_response[start_index:end_index]
        #print('\n\n\njson_response_dict ',json_response_dict)
        return json.dumps({'text': chatgpt_response, 'json': json_response_dict})



    def post(self):
        """
        Handles an incoming POST request containing a text command. The method sends the text command
        to the GPT-3 model and processes the response using the process_and_publish_chatgpt_response function in a separate thread.
        
        Returns:
            dict: A dictionary containing the GPT-3 model response as a JSON string.
        """

        text_command = request.form['text_command']
        print ('[ROSGPT] Command received. ', text_command, '. Asking ChatGPT ...')
        # Run the speak function on a separate thread
        #print('text_command:', text_command,'\n')
        chatgpt_response = self.askGPT(text_command)
        print ('[ROSGPT] Response received from ChatGPT. \n', str(json.loads(chatgpt_response))[:60], '...')
        #print('eval(chatgpt_response)', eval(chatgpt_response))
        # Run the speak function on a separate thread

        if chatgpt_response is None:
            return {'error': 'An error occurred while processing the request'}

        threading.Thread(target=process_and_publish_chatgpt_response, args=(self.chatgpt_ros2_node, text_command, chatgpt_response, True)).start()
        #print(json.loads(chatgpt_response))
        return json.loads(chatgpt_response)


@app.route('/')
def index():
    print(os.path.join(get_package_share_directory('rosgpt'), 'webapp'))
    return send_from_directory(os.path.join(get_package_share_directory('rosgpt'), 'webapp'), 'index.html')


def main():
    rclpy.init(args=None)
    chatgpt_ros2_node = ROSGPTNode()
    api.add_resource(ROSGPTProxy, '/rosgpt', resource_class_args=(chatgpt_ros2_node,))
    app.run(debug=True, host='0.0.0.0', port=5000)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
