#!/usr/bin/env python3

import rospy
import rospkg
import subprocess
import numpy as np

import control
import guidance
import control_util

import perception.msg
import drone_interface.msg

class MissionController():

    def __init__(self) -> None:
        node_name = "mission_control"
        rospy.init_node(node_name, anonymous=False)

        self._action_sequence = self._generate_action_sequence()
        mission_plan_params = control_util.load_config(node_name, "mission_plan_config_file")
        self._locations, self._locations_type = self._load_locations(mission_plan_params)

        control_params = control_util.load_control_params_config(node_name)

        self._controller = control.Controller(control_params)

        self._guidance_law_type = rospy.get_param("~guidance_law")
        rospy.loginfo(f"Using guidance law: {self._guidance_law_type}")
        guidance_law_params = control_params["guidance"][self._guidance_law_type]
        velocity_limits = control_params["guidance"]["velocity_limits"]
        guidance_law = guidance.get_guidance_law(self._guidance_law_type)
        self._guidance_law = guidance_law(guidance_law_params, velocity_limits)

        self._prev_telemetry_timestamp: float = None
        self._prev_telemetry: drone_interface.msg.AnafiTelemetry = None
        self._new_telemetry_available: bool = False
        self._prev_atttiude: np.ndarray = None # roll and pitch
        self._prev_velocity: np.ndarray = None # vx and vy

        self._require_user_confirmation = rospy.get_param("~require_user_confirmation")

        self._prev_pos_timestamp: float = None
        self._prev_pos: np.ndarray = None

        rospy.Subscriber("/drone/out/telemetry", drone_interface.msg.AnafiTelemetry, self._drone_telemetry_cb)
        rospy.Subscriber("/estimate/ekf", perception.msg.PointWithCovarianceStamped, self._ekf_cb)

    def _generate_action_sequence(self):
        mission_number = rospy.get_param("~mission_number")
        if mission_number == "test":
            return ["Takeoff", "Trackheli", "Land"]
        elif mission_number == "track":
            return ["Takeoff", "Trackheli"]

        rospack = rospkg.RosPack()
        graphplan_path = rospack.get_path("graphplan")

        subprocess.run(["python", f"{graphplan_path}/scripts/GraphPlan_main.py", "drone_domain.txt", f"drone_problem_{mission_number}.txt", "zero"])

        action_sequence = np.loadtxt(f"{graphplan_path}/output/problem{mission_number}.txt", dtype=str)

        return action_sequence

    def _load_locations(self, mission_plan_config: dict):
        mission_number = rospy.get_param("~mission_number")
        loc_type = mission_plan_config[f"mission_{mission_number}"]["loc_type"]

        locations = np.vstack((
            np.array([mission_plan_config["locations"]["loc_1"][f"{loc_type}_coords"]]),
            np.array([mission_plan_config["locations"]["loc_2"][f"{loc_type}_coords"]]),
            np.array([mission_plan_config["locations"]["loc_3"][f"{loc_type}_coords"]]),
        ))

        return locations, loc_type

    def _drone_telemetry_cb(self, msg: drone_interface.msg.AnafiTelemetry) -> None:
        self._prev_telemetry_timestamp = msg.header.stamp.to_sec()
        self._prev_telemetry = msg
        self._new_telemetry_available = True

    def _ekf_cb(self, msg: perception.msg.PointWithCovarianceStamped) -> None:

        self._prev_pos_timestamp = msg.header.stamp.to_sec()

        self._prev_pos = np.array([
            msg.position.x,
            msg.position.y,
            msg.position.z
        ])

    def _wait_for_hovering(self):
        rospy.loginfo("Waiting for drone to hover")
        # Require 5 messages in a row with hovering
        counter = 0
        while not rospy.is_shutdown():
            if self._new_telemetry_available:
                flying_state = self._prev_telemetry.flying_state
                if flying_state == "hovering":
                    counter += 1
                    if counter >= 5:
                        break
                else:
                    counter = 0
                self._new_telemetry_available = False
            rospy.sleep(0.1)
        rospy.loginfo("Hovering")

    def _get_reliable_altitude_estimate(self):
        # Use EKF if altitude is above 1m
        # if self._prev_pos[2] > 2:
        #     return self._prev_pos[2]
        # else:
        #     return -self._prev_telemetry.relative_altitude # negative to get it in the BODY frame
        return self._prev_pos[2]


    def _get_action_function(self, action: str):
        if action == "Takeoff":
            return self.takeoff
        elif action == "Land":
            return self.land
        elif "Move" in action:
            return self.move
        elif action == "Trackheli":
            return self.track_helipad
        elif "Search" in action:
            return self.search
        elif "Drop" in action:
            return self.drop
        elif action == "Resupply":
            return self.resupply
        else:
            print(f"Unknown action: {action}")
            raise ValueError

    def start(self):

        print("\nSelected action sequence:")
        for i, action in enumerate(self._action_sequence):
            print(f"\t{i+1}. {action}")

        control_util.await_user_confirmation(f"Start action sequence")

        for action in self._action_sequence:
            if not rospy.is_shutdown():
                function = self._get_action_function(action)
                if self._require_user_confirmation:
                    control_util.await_user_confirmation(f"Start action {action}")
                function(action)
                rospy.loginfo(f"Finished action {action}")
                rospy.sleep(1)

    def takeoff(self, action: str):
        # Take off and wait for drone to be stable in the air
        self._controller.takeoff(require_confirmation=False)
        self._wait_for_hovering()

        # Move up to a total of 3m altitude
        rospy.loginfo("Moving up 2m")
        self._controller.move_relative(0, 0, -2, 0)
        self._wait_for_hovering()

    def land(self, action: str):
        # Assuming that the altitude above the helipad is about 0.5m (done by the tracking
        # helipad action) and therefore we can just execute the landing here.
        self._controller.land(require_confirmation=False)

    def move(self, action: str):
        dest = int(action[-1])

        if self._locations_type == "relative":
            if dest == 1:
                origin = int(action[-2])
                dxyz = - self._locations[origin - 1] # -1 as locations are labeled 1,2,3 and not 0,1,2
            else:
                dxyz = self._locations[dest - 1]
            self._controller.move_relative(*dxyz, 0)
        else:
            print("GPS not implemented")
        # use_gps_coordinates should only be set to true in the simulator and if used in real
        # life one must be very careful to actually select the correct GPS location.
        self._wait_for_hovering()

    def track_helipad(self, action: str):
        rate = rospy.Rate(20)
        dt = 0.05
        v_d = np.zeros(4)

        pos_error_threshold = 0.2 # m

        # control_util.await_user_confirmation("Move away from the helipad")
        # self._controller.move_relative(-1, -1, 0, 0)
        # control_util.await_user_confirmation("Start tracking")

        # First align the drone with the helipad horizontally
        rospy.loginfo("Aligning horizontally, then descending")
        descending = False
        landing_position_ref = np.array([0, 0, 0.7]) # in body frame

        ready_to_land_counter = 0

        while not rospy.is_shutdown():

            if np.linalg.norm(self._prev_pos[:2]) < pos_error_threshold:
                if descending == False:
                    print("Starting to descend")
                descending = True
            else:
                if descending == True:
                    print("Hovering")
                descending = False

            if descending:
                alt = self._get_reliable_altitude_estimate()
                alt_error = alt - landing_position_ref[2]
                # Sign of position errro in z must be switched as positive climb rate is defined as upwards
                # in the drone interface, but since these measurements are in BODY, being above the desired
                # altitude will result in a positive error, hence this error must be made negative to work with
                # the control
                alt_error *= -1

                pos_error = np.hstack((self._prev_pos[:2], alt_error))
                # print(f"Error{pos_error}, altitude: {alt}")
                # if np.abs(pos_error[2]) < 0.2 and np.all(pos_error[:2] < 0.2):
                if np.all(np.abs(pos_error) < 0.2):
                    ready_to_land_counter += 1
                    if ready_to_land_counter >= 10:
                        break
                else:
                    ready_to_land_counter = 0
            else:
                pos_error = np.hstack((self._prev_pos[:2], 0))

            v_ref = self._guidance_law.get_velocity_reference(pos_error, self._prev_pos_timestamp, debug=False)
            v_d = self._controller.get_smooth_reference(v_d, v_ref[:2], dt)

            prev_vel = np.array([
                self._prev_telemetry.vx,
                self._prev_telemetry.vy,
                self._prev_telemetry.vz
            ])

            vd_3D = np.hstack((v_d[:2], v_ref[2]))
            self._controller.set_attitude3D(
                vd_3D, prev_vel, self._prev_telemetry_timestamp
            )

            rate.sleep()

        rospy.loginfo("Ready to land")

    def search(self, action: str):
        print(f"Searching in location {action[-1]}")
        print("Not implemented")

    def drop(self, action: str):
        print(f"Dropping life buoy in location {action[-1]}")

    def resupply(self, action: str):
        print("Resupplying")


def main():
    mission_controller = MissionController()
    mission_controller.start()

if __name__ == "__main__":
    main()