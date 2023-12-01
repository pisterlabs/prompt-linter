from typing import *
from openai import Agent

def change_tool_name_and_description(agent: Agent) -> None:
    '''
    Change the tool name and description of image_transformer to disassociate it from 'image' and 'prompt'.
    '''
    agent.toolbox['modifier'] = agent.toolbox.pop('image_transformer')
    agent.toolbox['modifier'].description = agent.toolbox['modifier'].description.replace(
        'transforms an image according to a prompt', 'modifies an image'
    )
