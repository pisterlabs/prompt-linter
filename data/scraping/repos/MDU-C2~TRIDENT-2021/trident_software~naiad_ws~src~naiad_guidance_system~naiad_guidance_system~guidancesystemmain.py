"""The main node for the guidance system that is repsonsible for handling
the guidance logic in NAIAD. Currently, the node can only request a guiding
session from Athena, but in the future, it will be more intelligent and
tell the navigation module when a new reference position is needed.

Author: Johannes Deivard 2021-11
"""
import rclpy
from baseclasses.tridentstates import GotoWaypointStatus, NaiadGuidanceSystemState
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.action import ActionClient, ActionServer

from geometry_msgs.msg import Pose, Point, Quaternion      # https://github.com/ros2/common_interfaces/blob/master/geometry_msgs/msg/Pose.msg
from std_srvs.srv import Trigger                           # https://github.com/ros2/common_interfaces/blob/master/std_srvs/srv/Trigger.srv
from std_msgs.msg import String
from trident_msgs.action import StartMission, GotoWaypoint
from trident_msgs.msg import Waypoint, WaypointAction, Mission
from trident_msgs.srv import GuidanceRequest, GetGoalPose


class GuidanceSystemNode(Node):
    """Node for the guidance system in Athena.
    """
    def __init__(self, node_name):
        super().__init__(node_name)
        self._guidance_system_state = NaiadGuidanceSystemState.IDLE
        # (Not really used at the moment)
        self._goto_waypoint_status = None
        # Reference position received from the guidance request response
        self._reference_position = None
        # Timer with callback that sends a guidance stop request
        self._stop_guidance_timer = None


        # Clients
        # -------
        self._guidance_request_client = self.create_client(
            GuidanceRequest,
            'guidance_system/guidance/request',
        )
        self._guidance_stop_client = self.create_client(
            Trigger,
            'guidance_system/guidance/stop',
        )


        # Servers
        # -------


        # Publishers
        # ----------
        self._guidance_system_state_publisher = self.create_publisher(
            String,
            'guidance_system/state',
            1
        )
        self._guidance_system_reference_position_publisher = self.create_publisher(
            Point,
            'guidance_system/reference_position',
            1
        )

        # Subscriptions
        # -------------
        # (NOTE: This isn't really needed in this node)
        self._goto_waypoint_status_subscriber = self.create_subscription(
            String,
            'navigation/waypoint/go/status',
            self._goto_waypoint_status_subscriber_callback,
            1
        )

        self.get_logger().info("Created Guidance System node.")

    def request_guidance(self, guidance_duration):
        """Sends a guidance request to Athena and starts a timer that sends a cancel request
        once the guidance duration has been reached.
        """
        while not self._guidance_request_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Guidance request service not available, waiting again...')
        request = GuidanceRequest()
        request.duration = int(guidance_duration)
        self.update_state(NaiadGuidanceSystemState.AWAITING_GUIDANCE)
        # Send the request
        response = self._guidance_request_client.call(request)
        # Create a rate object
        rate = self.create_rate(0.1) # 0.1 Hz
        # Keep trying to get a guidance request until it is accepted by the server
        while(not response.accepted):
            self.get_logger().info(f"Guidance request rejected: {response.message}")
            response = self._guidance_request_client.call(request)
            rate.sleep()
        
        self.get_logger().info(f"Guidance request accepted: {response.message}. Reference position: {response.reference_position}")
        self.update_state(NaiadGuidanceSystemState.BEING_GUIDED)
        self.update_reference_position(response.reference_position)
        # Check if the guidance duration is greater than 0
        if guidance_duration > 0:
            # Start the guidance stop timer
            self._stop_guidance_timer = self.create_timer(guidance_duration, self._stop_guidance_timer_callback)

    def _stop_guidance_timer_callback(self):
        """Callback for the stop_guidance_timer that is started when a guidance request
        with guidance_duration > 0 is accepted.
        """
        self._stop_guidance_timer.cancel()
        self._stop_guidance_timer.destroy()
        self.stop_guidance()

    def stop_guidance(self):
        """Sends a stop guidance request to the guidance server.
        """
        response = self._guidance_stop_client.call()
        if not response.success:
            self.get_logger().info(f"Failed to stop the guidance session, trying again in 10 seconds. Message: {response.message}")
            # If the request failed, try again in 10 seconds
            rate = self.create_rate(0.1) # 0.1 Hz
            rate.sleep()
            response = self._guidance_stop_client.call()

        if response.success:
            self.get_logger().info(f"Successfully stopped the guidance session. Message: {response.message}")
        else:
            self.get_logger().info(f"Second attempt to stop the guidance session failed. Message: {response.message}")

        self.update_state(NaiadGuidanceSystemState.IDLE)

        return response.success

    def update_reference_position(self, new_ref_pos):
        """Updates the reference position with the received reference position
        and publishes the reference position on the reference_position topic
        so that nodes that are using it will stay up to date.

        Args:
            new_ref_pos (Point): The new reference position.
        """
        # Check if the new_ref_pos is a Point
        if not isinstance(new_ref_pos, Point):
            # If not, convert it to a point
            p = Point()
            p.x, p.y, p.z = new_ref_pos[0], new_ref_pos[1], new_ref_pos[2]
            new_ref_pos = p
        # Update the reference position
        self._reference_position = new_ref_pos
        # Publish the new position
        self._guidance_system_reference_position_publisher.publish(new_ref_pos)

    def update_state(self, new_state):
        """Updates the state of the node and publishes the new state to the guidance_system/state topic.

        Args:
            new_state (GuidanceSystemState): The new state.
        """
        self._guidance_system_state = new_state
        state_msg = String()
        state_msg.data = str(new_state)
        self._guidance_system_state_publisher.publish(state_msg)

    # CALLBACKS
    # ---------
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
