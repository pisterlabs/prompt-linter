import numpy
from typing import Dict

from openai_ros2.utils import ut_gazebo
from openai_ros2.robots import LobotArmSim
from .arm_random_goal import LobotArmRandomGoal

import rclpy


class LobotArmFixedGoal(LobotArmRandomGoal):
    def __init__(self, node: rclpy.node.Node, robot, **kwargs):
        super().__init__(node, robot, **kwargs)

        # The target coords is currently arbitrarily set to some point achievable
        # This is the target for grip_end_point when target joint values are: [1.00, -1.00, 1.00]
        target_x = 0.10175
        target_y = -0.05533
        target_z = 0.1223
        self.target_coords = numpy.array([target_x, target_y, target_z])
        if isinstance(robot, LobotArmSim):  # Check if is gazebo or not
            # Spawn the target marker if it is gazebo
            print(f"Fixed goal: spawning to: {self.target_coords}")
            ut_gazebo.create_marker(node, target_x, target_y, target_z, diameter=0.004)
        self.previous_coords = numpy.array([0.0, 0.0, 0.0])

    def reset(self):
        self.previous_coords = numpy.array([0.0, 0.0, 0.0])
        if isinstance(self.robot, LobotArmSim):  # Check if is simulator or real
            # Move the target marker if it is gazebo
            print(f'Moving to {(self.target_coords[0], self.target_coords[1], self.target_coords[2])}')
            spawn_success = ut_gazebo.create_marker(self.node, self.target_coords[0],
                                                    self.target_coords[1], self.target_coords[2], diameter=0.004)
