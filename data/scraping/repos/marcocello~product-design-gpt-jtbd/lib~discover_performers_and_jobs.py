import os
import json 
import pandas as pd

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL")
TEMPERATURE = 0

model = ChatOpenAI(model_name=MODEL, temperature=TEMPERATURE)

prompt = ChatPromptTemplate.from_messages(
[
    ("system", """


Act like an accomplished business expert, visionary, and a seasoned advisor well-versed in the Jobs-to-be-Done theory. 


Based on input from users, you will generate different combinations of Job Performers and Jobs, focusing on creating optimal matches for effective outcomes.

Here are the characteristics of good Jobs to be used as a guide:

1. **Clear and Concise Phrasing:** Craft the main job in a way that is easily understandable and relatable to the target audience. Use language that resonates with users and succinctly conveys the essence of the job. For example, "Find a reliable babysitter" or "Plan a vacation itinerary."

2. **One-Dimensional Focus:** Ensure that the main job is narrowly focused on a single outcome or goal. Avoid unnecessary complexity or multifaceted requirements. The goal is to address a specific need or desire. Examples include "Lose weight" or "Find a new job."

3. **End State Orientation:** Formulate the main job with a clear end point or desired outcome. Users should be able to envision a specific achievement or completion associated with the job. Use language that implies reaching a goal, such as "Buy a new car" or "Renovate the kitchen."

Your role is to facilitate the generation of Jobs and Job Performers by incorporating these characteristics. 
     
     """),

    ("human", """

    Based on the following vision {vision}, generate 10 tuple of:
     - [Job_Perfomers]
     - [Aspiration_Jobs]
     - [Main_Jobs]
     - [Little_Jobs]

    [Job_Perfomers] might be generated having different professional and personal roles
     
    [Aspiration_Jobs]: These are ideal changes of state that individuals desire to become. They represent higher-level objectives and are more abstract. Example: Enjoy the freedom of mobility.
    
    [Main_Jobs]: These are broader objectives that are typically at the level of a main job. They are more specific than aspiration jobs but still represent a larger goal. Example: Get to a destination on time.
    
    [Little_Jobs]: These are smaller, more practical jobs that correspond roughly to stages in a big job. They are more concrete and specific tasks that need to be accomplished to achieve the main job. Example: Park the vehicle.

    {additional_prompt}     
     """)
])

functions = [
    {
    "name": "performers_and_jobs_discovery",
    "description": "Discovery of job performers and main jobs",
    "parameters": {
        "type": "object",
        "properties": {
            "performers_and_jobs_list": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "Job_Perfomers": {
                            "type": "string"
                        },
                        "Aspiration_Job": {
                            "type": "string"
                        },
                        "Main_Job": {
                            "type": "string"
                        },
                        "Little_Job": {
                            "type": "string"
                        },
                    }
                }
            },
        },
        "required": ["Job_Perfomers", "Aspiration_Job", "Main_Job", "Little_Job"]
    },
    },
]

chain = (
    prompt 
    | model.bind(function_call={"name": "performers_and_jobs_discovery"}, functions = functions)
)