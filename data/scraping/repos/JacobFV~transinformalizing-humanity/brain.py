import numpy as np
import langchain


class Brain:
    """
    Long term memory: vector KG (emb holds state and syntax (syn1 - syn2 = edge emb))
    Short term memory: LLM context window
    Sensorimotor memory: recent trajectories
    """

    def update(self):
        """
        
        Nodes are replaced based on the state S and syntax R
        
        n' := f(S, R)
        
        
        
        
        
        """
