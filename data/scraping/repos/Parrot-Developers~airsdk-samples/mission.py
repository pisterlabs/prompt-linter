#  Copyright (c) 2023 Parrot Drones SAS
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#  * Neither the name of the Parrot Company nor the names
#    of its contributors may be used to endorse or promote products
#    derived from this software without specific prior written
#    permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
#  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
#  PARROT COMPANY BE LIABLE FOR ANY DIRECT, INDIRECT,
#  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
#  OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
#  AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
#  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
#  SUCH DAMAGE.

# fsup mandatory library
from fsup.genmission import AbstractMission

# msg_id: Get the full message Id from a protobuf Command/Event class and the
# name of an 'id' field.
from msghub_utils import msg_id

###################################################
# Stages and transitions

from fsup.missions.default.ground.stage import (
    GROUND_STAGE as DEF_GROUND_STAGE,
)
from fsup.missions.default.hovering.stage import (
    HOVERING_STAGE as DEF_HOVERING_STAGE,
)
from fsup.missions.default.landing.stage import (
    LANDING_STAGE as DEF_LANDING_STAGE,
)
from fsup.missions.default.critical.stage import (
    CRITICAL_STAGE as DEF_CRITICAL_STAGE,
)

from fsup.missions.default.mission import TRANSITIONS as DEF_TRANSITIONS

###################################################
# Messages

# Drone messages
import drone_controller.drone_controller_pb2 as dctl_msgs
import flight_supervisor.autopilot_pb2 as autopilot_msgs
import guidance.ascent_pb2 as gdnc_ascent_msgs
import flight_supervisor.internal_pb2 as internal_fsup_msgs

# AirSDK Service messages (cv_road)
import road_runner.cv_road.messages_pb2 as rr_service_msgs

# AirSDK guidance messages
import road_runner.guidance.look_down.messages_pb2 as gdnc_look_down_msgs
import road_runner.guidance.road_following.messages_pb2 as gdnc_road_following_msgs  # noqa: E501

###################################################
# Overwritten Stages

from .flying.stage import FLYING_STAGE  # noqa: E402
from .takeoff.stage import TAKEOFF_STAGE  # noqa: E402

###################################################
# Messages channel

_CV_ROAD_SERVICE_CHANNEL = "unix:/tmp/road-runner-cv-road-service"

###################################################
# Mission


class Mission(AbstractMission):
    def __init__(self, env):
        super().__init__(env)

        # AIRSDK GUIDANCE MODE <---> FSUP
        # Look Down mode
        self.gdnc_look_down_handler_messages = None
        # Road following mode
        self.gdnc_road_following_handler_messages = None
        self.gdnc_road_following_messages_observer = None

        # AIRSDK SERVICE (cv_road) <---> FSUP
        self.airsdk_service_cv_road_messages_channel = None
        self.airsdk_service_cv_road_handler_messages = None
        self.airsdk_service_cv_road_messages_observer = None

    def on_load(self):
        # AIRSDK GUIDANCE MODE <---> FSUP
        # AirSDK guidance channel is already set up by flight supervisor
        # [self.mc.gdnc_channel]

        # AIRSDK SERVICE (cv_road) <---> FSUP
        self.airsdk_service_cv_road_messages_channel = (
            self.mc.start_client_channel(
                _CV_ROAD_SERVICE_CHANNEL
            )
        )

    def on_unload(self):
        # AIRSDK SERVICE (cv_road) <---> FSUP
        self.mc.stop_channel(self.airsdk_service_cv_road_messages_channel)
        self.airsdk_service_cv_road_messages_channel = None

    def on_activate(self):
        # AIRSDK GUIDANCE MODE <---> FSUP
        # Look Down mode
        self.gdnc_look_down_handler_messages = (
            self.mc.attach_client_service_pair(
                self.mc.gdnc_channel,
                gdnc_look_down_msgs,
                forward_events=True,
            )
        )

        # Road following mode
        self.gdnc_road_following_handler_messages = (
            self.mc.attach_client_service_pair(
                self.mc.gdnc_channel,
                gdnc_road_following_msgs,
                forward_events=True,
            )
        )

        # Road following mode event forwarder used to start the computer vision
        # service while in road_following mode
        # AIRSDK GUIDANCE MODE (Road following) --> FSUP --> AIRSDK SERVICE (cv_road)
        self.gdnc_road_following_messages_observer = (
            self.gdnc_road_following_handler_messages.evt.observe(
                {
                    msg_id(
                        gdnc_road_following_msgs.Event,
                        "road_following_enabled",
                    ): lambda *args: self._send_cv_road_enable(True),
                    msg_id(
                        gdnc_road_following_msgs.Event,
                        "road_following_disabled",
                    ): lambda *args: self._send_cv_road_enable(False),
                }
            )
        )

        # AIRSDK SERVICE (cv_road) ---> FSUP
        self.airsdk_service_cv_road_handler_messages = (
            self.mc.attach_client_service_pair(
                self.airsdk_service_cv_road_messages_channel,
                rr_service_msgs,
                forward_events=True,
            )
        )

        self.airsdk_service_cv_road_messages_observer = (
            self.airsdk_service_cv_road_handler_messages.evt.observe(
                {
                    msg_id(
                        rr_service_msgs.Event, "road_lost"
                    ): lambda *args: self._send_cv_road_enable(False),
                }
            )
        )

    def on_deactivate(self):
        # AIRSDK GUIDANCE
        # Look Down
        self.mc.detach_client_service_pair(self.gdnc_look_down_handler_messages)  # noqa: E501
        self.gdnc_look_down_handler_messages = None

        # Road following
        self.gdnc_road_following_messages_observer.unobserve()
        self.gdnc_road_following_messages_observer = None

        self.mc.detach_client_service_pair(self.gdnc_road_following_handler_messages)  # noqa: E501
        self.gdnc_road_following_handler_messages = None

        # AIRSDK SERVICE (cv_road)
        self.airsdk_service_cv_road_messages_observer.unobserve()
        self.airsdk_service_cv_road_messages_observer = None

        self.mc.detach_client_service_pair(self.airsdk_service_cv_road_handler_messages)  # noqa: E501
        self.airsdk_service_cv_road_handler_messages = None

        self.airsdk_service_cv_road_messages_channel = None

    def states(self):
        return [
            DEF_GROUND_STAGE,
            TAKEOFF_STAGE,
            DEF_HOVERING_STAGE,
            FLYING_STAGE,
            DEF_LANDING_STAGE,
            DEF_CRITICAL_STAGE,
        ]

    def transitions(self):
        transitions = TRANSITIONS + DEF_TRANSITIONS
        return transitions

    def _send_cv_road_enable(self, enable):
        self.airsdk_service_cv_road_handler_messages.cmd.sender.enable_cv(enable)  # noqa: E501
        self.log.info(f"receive message enable {enable}r")


Autopilot = lambda evt: msg_id(autopilot_msgs.Command, evt)  # noqa: E731
Dctl = lambda evt: msg_id(dctl_msgs.Event, evt)  # noqa: E731
GdncAscent = lambda evt: msg_id(gdnc_ascent_msgs.Event, evt)  # noqa: E731
Internal = lambda evt: msg_id(internal_fsup_msgs.Event, evt)  # noqa: E731
CvRoadService = lambda evt: msg_id(rr_service_msgs.Event, evt)  # noqa: E731
GdncLookDown = lambda evt: msg_id(gdnc_look_down_msgs.Event, evt)  # noqa: E731
GdncRoadFollowing = lambda evt: msg_id(
    gdnc_road_following_msgs.Event, evt
)  # noqa: E731, E501

TRANSITIONS = [
    # Overwritten transitions
    [
        Dctl("motors_ramping_done"),
        "takeoff.normal.wait_ascent",
        "takeoff.road_runner.ascent",
    ],
    [
        Dctl("motors_ramping_done"),
        "takeoff.normal.wait_motor_ramping",
        "takeoff.road_runner.ascent",
    ],
    # New transitions
    [GdncAscent("done"), "takeoff.road_runner.ascent", "flying.road_runner.look_down"],  # noqa: E501
    [
        GdncAscent("done_without_immobility"),
        "takeoff.road_runner.ascent",
        "flying.road_runner.look_down",
    ],
    [
        GdncLookDown("done"),
        "flying.road_runner.look_down",
        "flying.road_runner.road_following",
    ],
    # mission interrupted if:
    #   - The drone lose the road
    #   - The Road_following mode no longer receives telemetry sent by the
    #         cv_road service
    #   - An horizontal,vertical or yaw command is received.
    [CvRoadService("road_lost"), "flying.road_runner.road_following", "flying.manual"],  # noqa: E501
    [
        GdncRoadFollowing("telemetry_missed_too_long"),
        "flying.road_runner.road_following",
        "flying.manual",
    ],
    [
        Internal("pcmd_horizontal_move"),
        "flying.road_runner.road_following",
        "flying.manual",
    ],
    [Internal("pcmd_yaw"), "flying.road_runner.road_following", "flying.manual"],
    [Internal("pcmd_vertical"), "flying.road_runner.road_following", "flying.manual"],  # noqa: E501
]
