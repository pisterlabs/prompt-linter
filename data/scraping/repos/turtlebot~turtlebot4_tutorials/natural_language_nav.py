#!/usr/bin/env python3

# Copyright 2023 Clearpath Robotics, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# @author Ryan Gariepy (rgariepy@clearpath.ai)

import os

import rclpy
import ament_index_python
from rclpy.node import Node
from std_msgs.msg import String, Bool
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Directions, TurtleBot4Navigator

import openai


class GPTNode(Node):
    def __init__(self, navigator):
        super().__init__('gpt_node')
        self.declare_parameter('openai_api_key', '')
        self.declare_parameter('model_name', 'gpt-3.5-turbo')
        self.declare_parameter('parking_brake', True)

        # OpenAI key, model, prompt setup
        openai.api_key = self.get_parameter('openai_api_key').value
        self.model_name = self.get_parameter('model_name').value
        self.prompts = []
        self.full_prompt = ""

        # ROS stuff. Navigator node handle, topics
        self.navigator = navigator
        self.sub_input = self.create_subscription(String, 'user_input', self.user_input, 10)
        self.pub_ready = self.create_publisher(Bool, 'ready_for_input', 10)
        self.publish_status(False)

    def query(self, base_prompt, query, stop_tokens=None, query_kwargs=None, log=True):
        new_prompt = f'{base_prompt}\n{query}'
        """ Query OpenAI API with a prompt and query """

        use_query_kwargs = {
            'model': self.model_name,
            'max_tokens': 512,
            'temperature': 0,
        }
        if query_kwargs is not None:
            use_query_kwargs.update(query_kwargs)

        messages = [
            {"role": "user", "content": new_prompt}
        ]
        response = openai.ChatCompletion.create(
            messages=messages, stop=stop_tokens, **use_query_kwargs
        )['choices'][0]['message']['content'].strip()

        if log:
            self.info(query)
            self.info(response)

        return response

    def publish_status(self, status):
        """Publish whether or not the system is ready for more input."""
        msg = Bool()
        msg.data = status
        self.ready_for_input = status
        self.pub_ready.publish(msg)

    def user_input(self, msg):
        """Process user input and optionally execute resulting code."""
        # User input
        if not self.ready_for_input:
            # This doesn't seem to be an issue when there's only one publisher
            # but it's good practice
            self.info(f"Received input <{msg.data}> when not ready, skipping")
            return
        self.publish_status(False)
        self.info(f"Received input <{msg.data}>")

        # Add "# " to the start to match code samples, issue query
        query = '# ' + msg.data
        result = self.query(f'{self.full_prompt}', query, ['#', 'objects = ['])

        # This is because the example API calls 'navigator', not 'self.navigator'
        navigator = self.navigator

        # Execute?
        if not self.get_parameter('parking_brake').value:
            try:
                exec(result, globals(), locals())
            except Exception:
                self.error("Failure to execute resulting code:")
                self.error("---------------\n"+result)
                self.error("---------------")
        self.publish_status(True)

    def info(self, msg):
        self.get_logger().info(msg)
        return

    def warn(self, msg):
        self.get_logger().warn(msg)
        return

    def error(self, msg):
        self.get_logger().error(msg)
        return

    def debug(self, msg):
        self.get_logger().debug(msg)
        return


def read_prompt_file(prompt_file):
    """Read in a specified file which is located in the package 'prompts' directory."""
    data_path = ament_index_python.get_package_share_directory('turtlebot4_openai_tutorials')
    prompt_path = os.path.join(data_path, 'prompts', prompt_file)

    with open(prompt_path, 'r') as file:
        return file.read()


def main():
    rclpy.init()

    navigator = TurtleBot4Navigator()
    gpt = GPTNode(navigator)

    gpt.prompts.append(read_prompt_file('turtlebot4_api.txt'))
    for p in gpt.prompts:
        gpt.full_prompt = gpt.full_prompt + '\n' + p

    # No need to talk to robot if we're not executing
    if not gpt.get_parameter('parking_brake').value:
        gpt.warn("Parking brake not set, robot will execute commands!")
        # Start on dock
        if not navigator.getDockedStatus():
            navigator.info('Docking before initialising pose')
            navigator.dock()

        # Set initial pose
        initial_pose = navigator.getPoseStamped([0.0, 0.0], TurtleBot4Directions.NORTH)
        navigator.setInitialPose(initial_pose)

        # Wait for Nav2
        navigator.waitUntilNav2Active()

        # Undock
        navigator.undock()
    else:
        gpt.warn("Parking brake set, robot will not execute commands!")

    # Add custom context
    context = "destinations = {'wood crate': [0.0, 3.0, 0], 'steel barrels': [2.0, 2.0, 90], \
        'bathroom door': [-6.0, -6.0, 180] }"
    exec(context, globals())
    gpt.info("Entering input parsing loop with context:")
    gpt.info(context)
    gpt.full_prompt = gpt.full_prompt + '\n' + context

    # Main loop
    gpt.publish_status(True)
    try:
        rclpy.spin(gpt)
    except KeyboardInterrupt:
        pass

    gpt.destroy_node()
    navigator.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
