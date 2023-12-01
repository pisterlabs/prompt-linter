import numpy as np

import libpomp
import telemetry

import cam_controller.frame_of_reference_pb2 as cam_for_pb2
import cam_controller.control_mode_pb2 as cam_cm_pb2

import guidance.core as gdnc_core

import samples.hello.guidance.messages_pb2 as HelloGroundModeMessages

CONFIG_SUFFIX = "/" + HelloGroundModeMessages.Config.DESCRIPTOR.full_name


class HelloGroundMode(gdnc_core.Mode):
    FCAM_PITCH_ANIMATION_PERIOD_MS = 5000
    FCAM_PITCH_ANIMATION = [
        0.0, -0.2, -0.8, -2.0, -3.8, -6.6, -10.4, -15.5, -22.0, -30.1, -40.0,
        -25.0, -10.0, 4.9, 19.9, 34.9, 49.9, 55.5, 42.0, 28.5, 15.0, 1.5,
        -11.9, -25.4, -26.9, -22.4, -18.0, -13.5, -9.0, -4.4, 0.0
    ]

    def __init__(self, guidance, name):
        super().__init__(guidance, name)
        self.loop = self.guidance.get_loop()
        self.msghub = self.guidance.get_message_hub()
        self.front_cam_pitch_index = 0

        subset = [
            "attitude_euler_angles.yaw",
            "attitude_euler_angles.pitch",
            "attitude_euler_angles.roll",
        ]
        self.tlm_dctl = telemetry.TlmSection(
            "/dev/shm", "drone_controller", subset=subset
        )
        self.timer_cb = libpomp.pomp_timer_cb_t(lambda t, d: self._timer_cb())
        self.timer = libpomp.pomp_timer_new(self.loop, self.timer_cb, None)

        self.channel = self.guidance.get_channel(
            gdnc_core.ChannelKind.GUIDANCE
        )
        self.evt_sender = gdnc_core.MessageSender(
            HelloGroundModeMessages.Event.DESCRIPTOR.full_name
        )

        self.say = False
        self.say_count = 0

    def shutdown(self):
        self.loop = None
        self.msghub = None
        self.tlm_dctl = None
        libpomp.pomp_timer_destroy(self.timer)
        self.timer_cb = None
        self.timer = None
        self.evt_sender = None

    def get_triggers(self):
        return (gdnc_core.Trigger.TIMER, 30, 30)

    def configure(self, msg, disable_oa, override_fcam, override_stereo):
        if not msg.type_url.endswith(CONFIG_SUFFIX):
            raise ValueError("Ground: unexpected config: %s" % msg.type_url)

        ground_mode_msg = HelloGroundModeMessages.Config()
        msg.Unpack(ground_mode_msg)
        self.say = ground_mode_msg.say

        self.output.has_front_cam_config = True
        self.output.front_cam_config.yaw.locked = True
        self.output.front_cam_config.yaw.filtered = False
        self.output.front_cam_config.roll.locked = True
        self.output.front_cam_config.roll.filtered = False
        self.output.front_cam_config.pitch.locked = True
        self.output.front_cam_config.pitch.filtered = False

        if self.say:
            libpomp.pomp_timer_set_periodic(
                self.timer,
                # the initial delay (phase) is close to zero, in order
                # to start the animation right away, but not zero
                # because that would deactivate the timer.
                1,
                HelloGroundMode.FCAM_PITCH_ANIMATION_PERIOD_MS,
            )
            self.say_count = 0
        else:
            # clear the timer here, because the mode might be
            # reconfigured (set_mode with the same mode), in which
            # case exit() is not called
            libpomp.pomp_timer_clear(self.timer)

    def enter(self):
        self.msghub.attach_message_sender(self.evt_sender, self.channel)

    def exit(self):
        self.msghub.detach_message_sender(self.evt_sender)
        libpomp.pomp_timer_clear(self.timer)

    def begin_step(self):
        self.tlm_dctl.fetch_sample()

    def end_step(self):
        if (
            self.front_cam_pitch_index
            < len(HelloGroundMode.FCAM_PITCH_ANIMATION) - 1
        ):
            self.front_cam_pitch_index += 1

    def generate_drone_reference(self):
        pass

    def correct_drone_reference(self):
        pass

    def generate_attitude_references(self):
        # Front
        self.output.has_front_cam_reference = True
        fcam_ref = self.output.front_cam_reference

        fcam_ref.yaw.ctrl_mode = cam_cm_pb2.POSITION
        fcam_ref.yaw.frame_of_ref = cam_for_pb2.NED
        fcam_ref.yaw.position = self.tlm_dctl["attitude_euler_angles.yaw"]
        fcam_ref.pitch.ctrl_mode = cam_cm_pb2.POSITION
        fcam_ref.pitch.frame_of_ref = cam_for_pb2.NED
        fcam_ref.pitch.position = (
            HelloGroundMode.FCAM_PITCH_ANIMATION[self.front_cam_pitch_index]
            * np.pi
            / 180.0
        )
        fcam_ref.roll.ctrl_mode = cam_cm_pb2.POSITION
        fcam_ref.roll.frame_of_ref = cam_for_pb2.NED
        fcam_ref.roll.position = 0.0
        # Stereo
        self.output.has_stereo_cam_reference = True
        stcam_ref = self.output.stereo_cam_reference

        stcam_ref.yaw.ctrl_mode = cam_cm_pb2.POSITION
        stcam_ref.yaw.frame_of_ref = cam_for_pb2.NED
        stcam_ref.yaw.position = self.tlm_dctl["attitude_euler_angles.yaw"]

        stcam_ref.pitch.ctrl_mode = cam_cm_pb2.POSITION
        stcam_ref.pitch.frame_of_ref = cam_for_pb2.NED
        stcam_ref.pitch.position = self.tlm_dctl["attitude_euler_angles.pitch"]

        stcam_ref.roll.ctrl_mode = cam_cm_pb2.POSITION
        stcam_ref.roll.frame_of_ref = cam_for_pb2.NED
        stcam_ref.roll.position = self.tlm_dctl["attitude_euler_angles.roll"]

    def _timer_cb(self):
        self.log.info("Hello world")
        self.front_cam_pitch_index = 0
        self.say_count += 1

        msg = HelloGroundModeMessages.Event()
        msg.count = self.say_count
        gdnc_core.msghub_send(self.evt_sender, msg)


GUIDANCE_MODES = {"com.parrot.missions.samples.hello.ground": HelloGroundMode}
