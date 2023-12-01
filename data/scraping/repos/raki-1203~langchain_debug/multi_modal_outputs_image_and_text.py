import os
import re

# from steamship import Block, Steamship

from langchain import OpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import SteamshipImageGenerationTool

import config as c

os.environ['OPENAI_API_KEY'] = c.OPENAI_API_KEY
os.environ['SERPAPI_API_KEY'] = c.SERPAPI_API_KEY
os.environ['SERPER_API_KEY'] = c.SERPER_API_KEY
os.environ['GOOGLE_API_KEY'] = c.GOOGLE_API_KEY
os.environ['GOOGLE_CSE_ID'] = c.GOOGLE_CSE_ID
os.environ['STEAMSHIP_API_KEY'] = c.STEAMSHIP_API_KEY

if __name__ == '__main__':
    llm = OpenAI(temperature=0)

    # Dall-E
    SteamshipImageGenerationTool.update_forward_refs()
    tools = [
        SteamshipImageGenerationTool(model_name='dall-e'),
    ]

    mrkl = initialize_agent(tools,
                            llm,
                            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                            verbose=True)

    output = mrkl.run("How would you visualize a parot playing soccer?")
    print(output)








