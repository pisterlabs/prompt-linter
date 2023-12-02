import logging
from typing import Any, Dict

from gym_connectors import BonsaiConnector, GymSimulator

log = logging.getLogger("mountain-car")

class MountainCar(GymSimulator):
    """ Implements the methods specific to Open AI Gym Mountain Car environment 
    """    
    
    environment_name = 'MountainCar-v0'      # Environment name, from openai-gym

    def __init__(self, iteration_limit=200, skip_frame=1):
        """ Initializes the Mountain Car environment
        """
        self.bonsai_state = {"position": 0.0,
                             "speed": 0.0}

        super().__init__(iteration_limit, skip_frame)

    # convert openai gym observation to our state type

    def gym_to_state(self, observation) -> Dict[str, Any]:
        """ Converts openai environment observation to Bonsai state, as defined in inkling
        """
        self.bonsai_state = {"position": float(observation[0]),
                             "speed": float(observation[1])}

        return self.bonsai_state

    def action_to_gym(self, action: Dict[str, Any]):
        """ Converts Bonsai action type into openai environment action.       
        """
        actionValue = action['command']    #Open AI env doesn't expect array here
        return actionValue

    def get_state(self) -> Dict[str, Any]:
        """ Returns the current state of the environment 
        """
        log.debug('get_state: {}'.format(self.bonsai_state))
        return self.bonsai_state


if __name__ == "__main__":
    """ Creates a Mountain Car environment, passes it to the BonsaiConnector 
        that connects to the Bonsai service that can use it as a simulator  
    """
    logging.basicConfig()
    log = logging.getLogger("mountain-car")
    log.setLevel(level='DEBUG')

    #if more information is needed, uncomment this
    #gymlog = logging.getLogger("GymSimulator")
    #gymlog.setLevel(level='DEBUG')

    mountain_car = MountainCar()
    connector = BonsaiConnector(mountain_car)

    while connector.run():
        continue
