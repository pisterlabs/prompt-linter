import json
from typing import Dict

from langchain.chat_models import ChatOpenAI
from langchain.load.dump import dumps

from prompts import (CONTEXT_REFLECT_EXAMPLES, NO_CONTEXT_REFLECT_EXAMPLES,
                     reflection_template)


class Retro:
    def __init__(self,
                 temperature,
                 with_context: bool
                 ):
        self.replay_buffer = [] # Triplet (reflection prompt X_{k,i}, response: y_{k,i}, Return:G_{k,i}), trial i task k
        self.retro_temperature = temperature
        # here we trick langchain into thinking we are using openai model while we are actually using the fastchat one
        # conditioned on the fact that the server is in fact runing

        self.base_api_url = "http://localhost:8000/v1"
        self.openai_api_key = "EMPTY"
        self.model_name = "gpt-3.5-turbo"
        self.model = ChatOpenAI(openai_api_base=self.base_api_url,
                                openai_api_key=self.openai_api_key,
                                model=self.model_name,
                                temperature=self.retro_temperature)
        
        if with_context:
            self.reflection_examples = CONTEXT_REFLECT_EXAMPLES
        else:
            self.reflection_examples = NO_CONTEXT_REFLECT_EXAMPLES
        

    def generate_reflections(self, trajectories: Dict[int, Dict]) -> Dict[int, str]:
        """ Generate reflections for each task in the list of trajectories. 
            Returns a dictionary of task_id and corresponding reflection. """
        
        reflections = {}
        # loop over trajectories
        for task_id, data in trajectories.items():
            reflection_prompt = data["reflection_prompt"]

            # format the reflection prompt for each trail
            reflection_prompt = reflection_template.format(
                few_shot_demonstation=self.reflection_examples,
                previous_trial=reflection_prompt,
            )

            reflection = self.model.predict(reflection_prompt)
            reflections[task_id] = reflection

        
        return reflections

    def backward_pass(self, trajectory):
        
        pass
            
