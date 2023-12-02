"""The main node for the guidance system that is repsonsible for handling
the guidance of NAIAD. The node can start and stop pinging sessions,
responds to guidance requests and send reference position.

Author: Johannes Deivard 2021-10
"""
import rclpy
from baseclasses.tridentstates import GotoWaypointStatus, AthenaGuidanceSystemState
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.action import ActionClient, ActionServer

from geometry_msgs.msg import Pose, Point, Quaternion      # https://github.com/ros2/common_interfaces/blob/master/geometry_msgs/msg/Pose.msg
from std_srvs.srv import Trigger                           # https://github.com/ros2/common_interfaces/blob/master/std_srvs/srv/Trigger.srv
from std_msgs.msg import String
from trident_msgs.action import StartMission, GotoWaypoint
from trident_msgs.msg import Waypoint, WaypointAction, Mission
from trident_msgs.srv import GuidanceRequest, GetGoalPose, GetState


class GuidanceSystemNode(Node):
    """Node for the guidance system in Athena.
    """
    def __init__(self, node_name):
        super().__init__(node_name)
        self._guidance_system_state = AthenaGuidanceSystemState.IDLE
        self._goto_waypoint_status = None


        # Servers
        # -------
        self._guidance_request_server = self.create_service(
            GuidanceRequest,
            'guidance_system/guidance/request',
            self._guidance_request_callback
        )
        self._guidance_stop_server = self.create_service(
            Trigger,
            'guidance_system/guidance/stop',
            self._guidance_stop_callback
        )
        # Service to retrieve the state of the node
        self._get_state_server = self.create_service(
            GetState,
            'guidance_system/state/get',
            self._get_state_callback
        )

        # Clients
        # -------
        # The guidance system should know what position the agent is trying to hold
        # in order to send a good reference position in the guidance request.
        self._get_goal_pose_client = self.create_client(
            GetGoalPose,
            'navigation/waypoint/get/goal_pose',
        )

        # Publishers
        # ----------
        self._guidance_system_state_publisher = self.create_publisher(
            String,
            'guidance_system/state',
            1
        )

        # Subscriptions
        # -------------
        self._goto_waypoint_status_subscriber = self.create_subscription(
            String,
            'navigation/waypoint/go/status',
            self._goto_waypoint_status_subscriber_callback,
            1
        )

        self.get_logger().info("Created Guidance System node.")


    def update_state(self, new_state):
        """Updates the state of the node and publishes the new state to the guidance_system/state topic.

        Args:
            new_state (GuidanceSystemState): The new state.
        """
        self._guidance_system_state = new_state
        state_msg = String()
        state_msg.data = str(new_state)
        self._guidance_system_state_publisher.publish(state_msg)


    # Callbacks
    # ---------
    def _get_state_callback(self, _, response):
        """Simple getter for the node's state.
        """
        response.success = True
        response.state = str(self._guidance_system_state)
        response.int_state = self._guidance_system_state

        return response

    async def _guidance_request_callback(self, request, response):
        """Callback for the guidance request service.
        If the agent's GotoWaypoint status is Holding, the guidance request can be accepted,
        if not, the guidance request will be rejected since the agent isn't ready to guide.

        The guidance request will return a reference poisiton, accept/reject status, and a descriptive message.

        If the guidance request is accepted, this callback starts the pinger and a timer that shut downs the pinger
        and resets the state after the duration specified in the request. If the duration is 0, no timer will be created,
        since a 0 means indefinite guiding time.
        """
        # Check if the agent is in a state where it is NOT ready to guide
        if self._goto_waypoint_status is not GotoWaypointStatus.HOLDING_POSITION:
            response.accepted = False
            response.message = f"Not ready to guide. Current node state is {self._guidance_system_state}, and GotoWaypoint status is {self._goto_waypoint_status} (status needs to be HOLDING)."
            return response

        # If this point is reached it means the guidance system is ready to guide!
        # Request the goal position from the navigation node so we get a reference position that we will try to hold
        # instead of a reference position that we just happen to be on when the request is made.
        # (This would happen if we just read the agent state from position node.)
        request_future = await self._get_goal_pose_client.send_request()
        goal_pose = request_future.result()
        self.get_logger().info(f"Received response from GetGoalPose service: {goal_pose}")

        try:
            # TODO: Start pinger
            if request.duration != 0:
                # TODO: Create pinger shutdown timer
                pass
            self.update_state(AthenaGuidanceSystemState.GUIDING)
            response.accepted = True
            response.reference_position = goal_pose.position
            response.message = f"Guidance request accepted. Pinger started with reference position: {goal_pose.position}"
        except Exception as e:
            # We failed to start the pinger.
            response.accepted = False
            response.message = f"Guidance request rejected. Failed to start the pinger. Error message: {e}"
            return response

        return response
        
        
    def _guidance_stop_callback(self, _, response):
        """Callback for the guidance stop service.
        Tries to stop the pinger and resets the guidance status.

        Args:
            request ([type]): [description]
        """
        try:
            # TODO: Stop the pinger
            # Set the new state to idle
            response.success = True
            response.message = "Successfully stopped the guidance."
            self.update_state(AthenaGuidanceSystemState.IDLE)
        except Exception as e:
            response.success = False
            response.message = f"Failed to stop the guidance. Error {e}"

        return response


    def _goto_waypoint_status_subscriber_callback(self, msg):
        """Callback that handles the GotoWaypoint status messages sent by the Navigation node.
        NOTE: It might be better to make this a Getter service instead of constantly reading and
        updating the status.
        """
        self.get_logger().info(f"Read GotoWaypoint status update: {msg.data}. Updating state in guidance system.")
        self._goto_waypoint_status = GotoWaypointStatus[msg.data]


def main(args=None):
    rclpy.init(args=args)
    guidance_system_node = GuidanceSystemNode("guidance_system")
    executor = MultiThreadedExecutor()
    rclpy.spin(guidance_system_node, executor)
    rclpy.shutdown()


if __name__=="__main__":
    main()
