import json
import os
import threading
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Union

from flask import Flask, jsonify, request
from flask_restful import Api, Resource
import json
import openai
import rclpy
import threading
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.executors import SingleThreadedExecutor
import rclpy
from ament_index_python import get_package_share_directory
from flask import Flask, request, send_from_directory
from flask_cors import CORS
from flask_restful import Api, Resource
from geometry_msgs.msg import Pose
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from pydantic import BaseModel, Field, validator
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.executors import SingleThreadedExecutor
import subprocess
import copy

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ros_agent.agent import load_agent

# Instantiate a Flask application object with the given name
app = Flask(__name__)

# Enable Cross-Origin Resource Sharing (CORS) for the Flask app
CORS(app, origins=['*'])

# Create an API object that wraps the Flask app to handle RESTful requests
api = Api(app)


# Now you can use the openai_api_key variable to authenticate with the OpenAI API


# Initialize a threading lock for synchronizing access to shared resources
# when multiple threads are involved
spin_lock = threading.Lock()


# Create a separate threading lock for synchronizing access to the TTS engine
tts_lock = threading.Lock()


class ROSGPTNode(Node):
    def __init__(self):
        """
        Initialize the ROSGPTNode class which is derived from the rclpy Node class.
        """
        super().__init__('chatgpt_ros2_node')
        self.publisher = self.create_publisher(String, 'voice_cmd', 10)
        
        # Subscribe to the 'position_data' topic
        self.subscription = self.create_subscription(
            Pose, 
            '/drone/gt_pose', 
            self.position_callback, 
            10)
        
        self.subscription
        self.position = None  # Initialize the position attribute to None

    def publish_message(self, message):
        """
        Publish the given message to the 'voice_cmd' topic.

        Parameters
        ----------
        message : str
            The message to be published.
        """
        msg = String()
        msg.data = message
        self.publisher.publish(msg)

    def position_callback(self, msg):
        """
        Callback function that gets executed whenever a new position message is received.

        Parameters
        ----------
        msg : PointStamped
            The received message containing position data.
        """
        self.position = (msg.position.x, msg.position.y, msg.position.z)

    def get_loc(self):
        """
        Return the latest position data received by the subscriber.

        Returns
        -------
        tuple or None
            The latest position data in the form (x, y, z), or None if no position data has been received.
        """
        return self.position
    
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
        self.agent = load_agent()


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

        Ontology:

            [{'command': {'action': 'land'}}]
            [{'command': {'action': 'takeoff'}}]
            [{'command': {'action': 'move', 'params': {'linear_speed': linear_speed, 'distance': distance, 'direction': direction}}}]
            [{'command': {'action': 'stop'}}]

            Example:


            Consider the following ontology:

            [{'command': {'action': 'land'}}]
            [{'command': {'action': 'takeoff'}}]
            [{'command': {'action': 'move', 'params': {'linear_speed': linear_speed, 'distance': distance, 'direction': direction}}}]
            [{'command': {'action': 'stop'}}]

            You may get a command in another language, translate it to English, and then create the JSON.

            The 'direction' parameter can take values "forward", "backward", "left", "right", "up", "down" to indicate the direction of movement. Here are some examples.

            If speed is not given in the prompt, it is assumed to be 0.5 meters per second.
            All numerical answers should be in float form.

            Command: "takeoff and Move forward for 1 meter at a speed of 0.5 meters per second."
            Final Answer: [{'command': {'action': 'takeoff'}}, {'command': {'action': 'move', 'params': {'linear_speed': 0.5, 'distance': 1.0, 'direction': 'forward'}}}]

            Examples:

            Command: "Takeoff, move forward for 3 meters, then land."
            Final Answer: [{'command': {'action': 'takeoff'}}, {'command': {'action': 'move', 'params': {'linear_speed': 0.5, 'distance': 3.0, 'direction': 'forward'}}}, {'command': {'action': 'land'}}]

            Command: "Move left for 2 meters, move upwards for 1 meter, then stop."
            Final Answer: [{'command': {'action': 'move', 'params': {'linear_speed': 0.5, 'distance': 2.0, 'direction': 'left'}}}, {'command': {'action': 'move', 'params': {'linear_speed': 0.5, 'distance': 1.0, 'direction': 'up'}}}, {'command': {'action': 'stop', 'params': {}}}]
            

            Command: "Go to the corn_garden."
            Final Answer: [{'command': {'action': 'move', 'params': {'linear_speed': 0.5, 'distance': 10.0, 'direction': 'left'}}},
                        {'command': {'action': 'move', 'params': {'linear_speed': 0.5, 'distance': 10.0, 'direction': 'backward'}}},
                        {'command': {'action': 'move', 'params': {'linear_speed': 0.5, 'distance': 2.0, 'direction': 'down'}}}]

            Command: "Go to the tree_garden."
            Thought: The user wants the drone to move to the location known as "tree_garden."
            Action: Use the RetrievePOICoordinates tool
            Action Input: "tree_garden"
            Observation: (5, -5, 3)  # Coordinates for the tree_garden
            Thought: I've got the coordinates for the tree_garden. Now, I need to check the current location of the drone.
            Action: Use the GetDroneLocation tool
            Observation: (3, -4, 2)  # Current drone location, as provided in the hypothetical scenario
            Thought: I have the drone's current location. Now, I will compute the movements required for the drone to reach the tree_garden.
            Action: Use the ComputeDroneMovements tool
            Action Input: "[3, -4, 2, 5, -5, 3, 0.5]"  # Using the format [drone_x, drone_y, drone_z, poi_x, poi_y, poi_z, speed]
            Observation: ("The drone should move right 2 meters at 0.5 meters per second", 
                        "The drone should move backward 1 meter at 0.5 meters per second",
                        "The drone should move up 1 meter at 0.5 meters per second")
            Thought: I have calculated the movements required for the drone. I will now convert these instructions into a suitable command format.
            Action: Convert to JSON format using the CustomCommandToJSON tool
            Action Input: "The drone should move right 2 meters at 0.5 meters per second"
            Observation: {'command': {'action': 'move', 'params': {'linear_speed': 0.5, 'distance': 2.0, 'direction': 'right'}}}
            Final Answer: [{'command': {'action': 'move', 'params': {'linear_speed': 0.5, 'distance': 2.0, 'direction': 'right'}}},
                        {'command': {'action': 'move', 'params': {'linear_speed': 0.5, 'distance': 1.0, 'direction': 'backward'}}},
                        {'command': {'action': 'move', 'params': {'linear_speed': 0.5, 'distance': 1.0, 'direction': 'up'}}}]


        You will be given human language prompts, and you need to return a JSON conformant to the ontology. Any action not in the ontology must be ignored.

        Examples in other languages:

        German:
        Prompt: "Bewegen Sie sich vorwärts für 1 Meter mit einer Geschwindigkeit von 0,5 Metern pro Sekunde."
        Returns: {"action": "move", "params": {"linear_speed": 0.5, "distance": 1, "direction": "forward"}}

        Japanese:
        Prompt: "Mae ni 1 meter, 0.5 meters per second no sokudo de susumu."
        Returns: {"action": "move", "params": {"linear_speed": 0.5, "distance": 1, "direction": "forward"}}

        Hindi:
        Prompt: "Aage badho 1 meter, 0.5 meters per second ke liye."
        Returns: {"action": "move", "params": {"linear_speed": 0.5, "distance": 1, "direction": "forward"}}
        '''


        prompt = prompt+'\nprompt: '+text_command
        #print(prompt) #for testing
        

        # Create the message structure for the GPT-3 model
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
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

        print('subhmx', text_command)

        if text_command.startswith('[') and text_command.endswith(']'):
            text_command = json.loads(text_command)
        else:
            text_command = [text_command]
        
        print(text_command)
        chatgpt_response = self.agent.run(text_command)

        print('subhmx122', chatgpt_response)

        if not chatgpt_response:
            return {'error': 'An error occurred while processing the request'}
        elif chatgpt_response == "Agent stopped due to iteration limit or time limit." or chatgpt_response == "[]":
            return {'error': 'An error occurred while processing the request'}
        elif chatgpt_response[0] != '[':
            return {
                'require_more_info': True,
                'help_text': chatgpt_response
            }
        else:
            print('subhmx123: response was successful')
            threading.Thread(
                target=process_and_publish_chatgpt_response, 
                args=(self.chatgpt_ros2_node, text_command, chatgpt_response, True)).start()
            return {
                'response': "Captain, your command is being executed now!",
                'chatgpt_response': chatgpt_response
            }

        # text_command = request.form['text_command']
        # print ('[ROSGPT] Command received. ', text_command, '. Asking ChatGPT ...')
        # # Run the speak function on a separate thread
        # #print('text_command:', text_command,'\n')
        # chatgpt_response = self.askGPT(text_command)
        # print ('[ROSGPT] Response received from ChatGPT. \n', str(json.loads(chatgpt_response))[:60], '...')
        # #print('eval(chatgpt_response)', eval(chatgpt_response))
        # # Run the speak function on a separate thread

        # if chatgpt_response is None:
        #     return {'error': 'An error occurred while processing the request'}

        # threading.Thread(target=process_and_publish_chatgpt_response, args=(self.chatgpt_ros2_node, text_command, chatgpt_response, True)).start()
        # #print(json.loads(chatgpt_response))
        # return json.loads(chatgpt_response)


@app.route('/')
def index():
    webapp_dir = Path(get_package_share_directory('rosgpt')) / 'webapp'
    return send_from_directory(str(webapp_dir), 'index.html')


@app.route('/get_drone_location', methods=['GET'])
def get_drone_location():
    print('subhmx: get_drone_location')
    location = chatgpt_ros2_node.get_loc()

    # location = [0,0,0]
    
    print('subhmx: get_drone_location222')

    print(location)
    if location:
        return jsonify({
            "x": round(location[0], 2),
            "y": round(location[1], 2),
            "z": round(location[2], 2)
        })
    else:
        return jsonify({"error": "No location data available"}), 404


def main():
    rclpy.init(args=None)
    global chatgpt_ros2_node  # This makes sure we use the correct node instance in the Flask route
    chatgpt_ros2_node = ROSGPTNode()
    api.add_resource(ROSGPTProxy, '/rosgpt', resource_class_args=(chatgpt_ros2_node,))
    app.run(debug=False, host='0.0.0.0', port=5000)
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()
