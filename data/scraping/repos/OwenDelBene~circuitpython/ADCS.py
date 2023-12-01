

try:
    from lib.adcs.sensors import sensors
    from lib.adcs.actuators import actuators
    from lib.adcs.guidance import guidance
    from lib.adcs.control import control
    from lib.adcs.estimator import estimator

    from lib.adcs.state_machine import state_machine
    import lib.adcs.data_types as tp
except:

    import sensors
    import actuators
    import guidance
    import control
    import estimator

    from state_machine import state_machine
    import data_types as tp

class ADCS():

    ## constructor

    def __init__(self, sta : state_machine):
        
        self.state_machine = sta
        self.transition : tp.transition_t
        self.adcs_delay = 0

        


    ## functions

    # processADCSevents()
    def processADCSevents(self):
        # if time >= adcs delay time
            # execute adcs action
            # adcs delay = self.data.state.delay
            # execute adcs transition
        # if (time >= self.adcs_delay):
        # if (self.cubesat.pib >= self.adcs_delay):
        while True:
            self.state_machine.executeADCSaction(self.state_machine.data.state)
            self.adcs_delay_count = 0

            self.adcs_delay = (tp.ctrl_states[self.state_machine.data.state]).delay
            self.state_machine.executeADCStransition(tp.ctrl_states[self.state_machine.data.state].transition)

            print(self.state_machine.data.state)
            if (self.adcs_delay > 0):
                break




    
