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

import numpy as np

import libguidance_utils_binding as pyguidance
import libquaternion_binding as quaternion
import cfgreader
import telemetry
import guidance.core as gdnc_core

import cam_controller.frame_of_reference_pb2 as cam_for_pb2
import cam_controller.control_mode_pb2 as cam_cm_pb2
import road_runner.guidance.look_down.messages_pb2 as look_down_mode_pb2

LOOK_DOWN_CONFIG_FILENAME = "/etc/guidance/look_down/mode.cfg"


def _config(field):
    cfg = cfgreader.build_config_start()
    for (root, include) in field:
        cfg = cfgreader.build_config_update(cfg, root, include)
    return cfgreader.build_config_end(cfg)


class LookDownMode(gdnc_core.Mode):
    def __init__(self, guidance, name):
        super().__init__(guidance, name)

        # Get guidance context
        self.msghub = self.guidance.get_message_hub()

        mode_config_path = guidance.get_config_file(LOOK_DOWN_CONFIG_FILENAME)

        # Get configuration values
        field = [
            (
                mode_config_path,
                "cameraPitchPosition",
            ),
            (
                mode_config_path,
                "reachedThresholdPercentageOfAngle",
            ),
        ]
        look_down_cfg = _config(field)

        #  Convertion from degree to radian
        self.camera_pitch_position = (
            look_down_cfg.look_down.cameraPitchPosition * np.pi / 180
        )

        #  Target reached detector configuration
        self.att_reached_detector_cfg = (
            pyguidance.AttitudeReachedDetector.Configuration()
        )
        self.att_reached_detector_cfg.target_kind = (
            pyguidance.AttitudeReachedDetector.ANGLE
        )
        self.att_reached_detector_cfg.angle_threshold = 0.0
        self.att_reached_detector_cfg.rate_threshold = 0.0

        self.target_reached = False

        # Target reached detector init
        self.attitude_reached_detector = pyguidance.AttitudeReachedDetector(
            self.att_reached_detector_cfg
        )

        self.attitude_reached_detector.set_threshold(
            look_down_cfg.look_down.reachedThresholdPercentageOfAngle
            * np.abs(self.camera_pitch_position),
        )

        # Telemetry consumer configuration
        subset = [
            "ref_ned_start_angles_yaw",
            "ref_ned_start_angles_pitch",
            "ref_ned_start_angles_roll",
        ]

        self.tlm_fcam = telemetry.TlmSection(
            "/dev/shm", "fcam_controller", subset=subset
        )

        #  Msghub configuration
        self.channel = self.guidance.get_channel(
            gdnc_core.ChannelKind.GUIDANCE
        )
        self.evt_sender = gdnc_core.MessageSender(
            look_down_mode_pb2.Event.DESCRIPTOR.full_name
        )

    def shutdown(self):
        # Msghub
        self.evt_sender = None
        self.channel = None
        self.msghub = None

        # Telemetry
        self.tlm_fcam = None

        # Target reached detector
        self.att_reached_detector_cfg = None

    def configure(self, msg, disable_oa, override_fcam, override_stereo):
        # Telemetry
        self.tlm_fcam.fetch_sample()

        # Target reached detector
        self.attitude_reached_detector.set_target(
            quaternion.from_euler(
                self.tlm_fcam["ref_ned_start_angles_yaw"],
                self.camera_pitch_position,
                self.tlm_fcam["ref_ned_start_angles_roll"],
            )
        )

        self.target_reached = False

        # Fcam pitch axis configuration
        self.output.has_front_cam_config = True
        fcam_config = self.output.front_cam_config
        fcam_config.pitch.locked = True
        fcam_config.pitch.filtered = True
        fcam_config.pitch.smoothness = 0.1

    def generate_attitude_references(self):
        # Fcam axes references
        self.output.has_front_cam_reference = True
        fcam_ref = self.output.front_cam_reference
        fcam_ref.pitch.ctrl_mode = cam_cm_pb2.POSITION
        fcam_ref.pitch.frame_of_ref = cam_for_pb2.NED
        fcam_ref.pitch.position = self.camera_pitch_position

    def get_triggers(self):
        return (gdnc_core.Trigger.TIMER, 30, 30)  # Aprox 33.33 Hz

    def enter(self):
        self.msghub.attach_message_sender(self.evt_sender, self.channel)

    def exit(self):
        self.msghub.detach_message_sender(self.evt_sender)

    def begin_step(self):
        # Telemetry: get sample
        self.tlm_fcam.fetch_sample()

    def end_step(self):
        self.attitude_reached_detector.process(
            quaternion.from_euler(
                self.tlm_fcam["ref_ned_start_angles_yaw"],
                self.tlm_fcam["ref_ned_start_angles_pitch"],
                self.tlm_fcam["ref_ned_start_angles_roll"],
            )
        )

        self.target_reached = self.attitude_reached_detector.is_reached()

        if self.target_reached:
            tosend = look_down_mode_pb2.Event()
            tosend.done.SetInParent()
            gdnc_core.msghub_send(self.evt_sender, tosend)

    def generate_drone_reference(self):
        # Unused in this mode. Not mandatory.
        pass

    def correct_drone_reference(self):
        # Unused in this mode. Not mandatory.
        pass


# Export classes

GUIDANCE_MODES = {
    "com.parrot.missions.samples.road_runner.look_down": LookDownMode
}
