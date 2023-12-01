try:
    import lib.adcs.state_actions
    import lib.adcs.state_transitions
    import lib.adcs.data_types as tp

    from lib.adcs.sensors import sensors
    from lib.adcs.actuators import actuators
    from lib.adcs.guidance import guidance
    from lib.adcs.control import control
    from lib.adcs.estimator import estimator
    from lib.adcs.ekf_commands import ekf_commands
except:

    import state_actions
    import state_transitions
    import data_types as tp

    from sensors import sensors
    from actuators import actuators
    from guidance import guidance
    from control import control
    from estimator import estimator
    from ekf_commands import ekf_commands

class state_machine():

    MAX_ATTEMPTS = 4

    def __init__(self, data: tp.adcs_data_t,
                    sen : sensors, act : actuators, guid : guidance, ctrl : control):
        self.sensors = sen
        self.actuators = act
        self.guidance = guid
        self.control = ctrl
        self.data = data
        self.ekf = ekf_commands(self.data.ekf_data)

        self.adcs_delay = 0
        self.adcs_delay_count = 0

        self.stayBdot = False

    def state_error(self, flag):
        # cc_ADCS_info->state_attempt = flag ? (cc_ADCS_info->state_attempt+1) : 0;
	    # cc_ADCS_info->status = flag ? ADCS_ERROR : ADCS_OK;
        if (flag):
            self.data.state_attempt = self.data.state_attempt + 1
        else:
            self.data.state_attempt = 0

    # method for determineing where or not to retry adcs state or enter safe mode
    def retry_state(self):
        if ((self.data.state_attempt < self.MAX_ATTEMPTS) and (self.data.status == tp.status_t.ERROR)):
            return True
        self.data.state_attempt = 0
        return False

    # //generic mode transition function (should be used in all transXxxTransition states)
    # //checks for commanded mode, then checks for errors, otherwise returns nominal transition
    def mode_transition(self, ideal_state : tp.state_t):
        # in state_transitions.py
        # don't need here
        pass

    def executeADCSaction(self, state : tp.state_t):
        if (state >= 0 and state < len(tp.ctrl_states)):
            state_actions.ctrl_actions[tp.ctrl_states[state].action](self)
        else:
            pass
       
    def executeADCStransition(self, transition: tp.transition_t):
        # TODO: use the cmd library in the top to shorten the code
        if (transition >=0 and transition < len(state_transitions.ctrl_transitions)):
            self.data.state = state_transitions.ctrl_transitions[transition](self, self.adcs_delay)
        else:
            print("undefined adcs state: entering safe mode")
            # TODO: find g_Safe_Mode in C code
            self.data.state = state_transitions.transSafe1(self)
            #     LogCommand("Undefined ADCS State: Entering Safe Mode 1");
            # g_SafeMode.mode = SAFE_MODE_1;
            # g_SafeMode.set = true;
            # g_ccADCSinfo.state = transSafe1(&g_ccADCSinfo, g_ADCSdelay);
            # break;
