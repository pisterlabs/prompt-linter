import logging
from typing import Any, Dict

from gym_connectors import BonsaiConnector, GymSimulator

log = logging.getLogger("cartpole")

class CartPole(GymSimulator):
    """ Implements the methods specific to Open AI Gym CartPole environment 
    """    

    environment_name = 'CartPole-v1'      # Environment name, from openai-gym

    def __init__(self, iteration_limit=200, skip_frame=1):
        """ Initializes the CartPole environment
        """
        self.bonsai_state = {"cart_position": 0.0,
                            "cart_velocity": 0.0,
                            "pole_angle": 0.0,
                            "pole_angular_velocity": 0.0}

        super().__init__(iteration_limit, skip_frame)

    def gym_to_state(self, observation):
        """ Converts openai environment observation to Bonsai state, as defined in inkling
        """    
        self.bonsai_state = {"cart_position": float(observation[0]),
                       "cart_velocity": float(observation[1]),
                       "pole_angle":    float(observation[2]),
                       "pole_angular_velocity": float(observation[3])}
					   
        return self.bonsai_state

    def state_to_gym(self, state):
        return [state["cart_position"],state["cart_velocity"], state["pole_angle"], state["pole_angular_velocity"] ]

    def action_to_gym(self, action):
        """ Converts Bonsai action type into openai environment action.       
        """ 
        return action['command']

    def gym_to_action(self, gym_action):
        return {"command": gym_action}
        
    def get_state(self) -> Dict[str, Any]:
        """ Returns the current state of the environment 
        """
        log.debug('get_state: {}'.format(self.bonsai_state))
        return self.bonsai_state

if __name__ == "__main__":
    """ Creates a CartPole environment, passes it to the BonsaiConnector 
        that connects to the Bonsai service that can use it as a simulator  
    """
    logging.basicConfig()
    log = logging.getLogger("cartpole")
    log.setLevel(level='DEBUG')

    #if more information is needed, uncomment this
    #gymlog = logging.getLogger("GymSimulator")
    #gymlog.setLevel(level='DEBUG')

    cartpole = CartPole()
    connector = BonsaiConnector(cartpole)

    while connector.run():
        continue
