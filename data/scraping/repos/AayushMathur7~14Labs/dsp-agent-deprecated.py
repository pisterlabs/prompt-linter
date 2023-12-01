'''
MultiOn has various limitations that will not allow us to upload audio on Spotify.
Additionally, Langchain itself has limitations.
Selenium will be used as an alternative instead.
'''


import os
from dotenv import load_dotenv

from langchain.llms import OpenAI

from langchain.agents import initialize_agent
from langchain.agents import AgentType

os.environ["LANGCHAIN_TRACING"] = "true"
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
from langchain.tools import StructuredTool
from human_input import HumanInputRun


from multion import MultionToolSpec


def agent(query: str):
    multion_toolkit = MultionToolSpec(use_api=True, mode="auto")
    # multion.set_remote(True)

    tool = StructuredTool.from_function(multion_toolkit.browse)
    human_input = HumanInputRun()

    llm = OpenAI(temperature=0)

    # Structured tools are compatible with the STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION agent type.
    agent_executor = initialize_agent(
        [tool, human_input],
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    return agent_executor.run(query)

#     1.
    # 1. If it is not logged in yet, then log in to Spotify for Podcasters. Click on continue to spotify if that option appears and log in as whoever's profile saved, else ask for credentials. Else, skip this step.


    # 3. Generate in the details for the title, episode description, publish date (now), explicit content (no)
    # 4. Generate a sample image and use that as the cover art
    # 5. Once filled in details click next until review step
    # 6. Publish it and retrieve the spotify link

    # Concise summary of content for digestible ear candies!

PROMPT = f"""
You are an expert AI Agent whose job is to `upload a mp3 audio file on spotify podcasters and retrieve the spotify link to the audio` (https://podcasters.spotify.com/pod/dashboard/home).

    Here are the high-level steps:
    1. Click on the new episode button
    2. Take the music-example.mp3 file and upload it
    3. Open
"""

# PROMPT = f"""
# You are an expert AI Agent whose job is to `display weather data` (https://www.google.com).

#     Here are the high-level steps:
#     1. Go to google
#     2. Get the average temperature of today
# """
agent(query=PROMPT)
