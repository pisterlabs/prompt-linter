from cursor_control import CursorControl
from key_control import KeyControl
from tracker import Tracker
from guidance_control import GuidanceControl


class BackEnd(object):

    def __init__(self, state):
        self.state = state
        self.key_control = KeyControl(self.state)
        self.cursor_control = CursorControl(self.state)
        self.tracker = Tracker(self.state)
        self.guidance_control = GuidanceControl(self.state)

    def update(self):

        if self.state.TR_active and self.state.TR_reset:
            self.tracker.init_tracker()

        if not self.state.KC_manual and not self.state.GS_active and self.state.TR_active:
            self.guidance_control.init_guidance_control()
