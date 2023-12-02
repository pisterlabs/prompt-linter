import autogen
import guidance
from typing import List
from multi_agent_talecraft_llm.agents import config

from multi_agent_talecraft_llm.agents.utils import is_termination_msg

CHARACTER_PROMPT = """A character in a story. Interact with the other characters. Follow the narrative. Only use dialogues to speak to other characters. 

NAME: {{name}} 
PERSONALITY: {{personality}}"""

class CharacterCreationTool:
    '''
    Functions that agents can use
    '''
    def __init__(self) -> None:
        self.characters = []

    @property
    def character_agents(self) -> List[autogen.AssistantAgent]:
        return self.characters

    def create_characters(self, names: List[str], personality_prompts: List[str]) -> None:
        for name, personality in zip(names, personality_prompts):
            # if name has any spaces, strip and replace with underscore
            name = name.strip().replace(" ", "_")
            character = autogen.AssistantAgent(
                    name=name,
                    llm_config=config.base_config,
                    system_message=guidance(CHARACTER_PROMPT, name, personality),
                    code_execution_config=False,
                    is_termination_msg=is_termination_msg,
            )
            self.characters.append(character)
