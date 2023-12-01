import json
import os
import pandas as pd

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL")
TEMPERATURE = os.getenv("TEMPERATURE")

model = ChatOpenAI(model_name=MODEL, temperature=TEMPERATURE)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
Act as a multi-year expert in Jobs-to-be-done expert with very knowledgeable about the Jobs-to-be-done book like The Jobs to be Done Playbook by Jim Kalbach.

You will analyze a job performer interview from the user and you will extract all those information:

1. Job Steps: Identify and extract all discernible job steps from the user interviews. Characteristics of Good Job Steps: -)Clear and Concise Phrasing: Ensure that the main job steps are described in a manner that is easily understood and relatable to the target audience. For instance, use straightforward language like "Find a reliable babysitter" or "Plan a vacation itinerary." -) One-Dimensional Focus: Each main job step should focus on a single outcome or goal, avoiding complexity. Examples include "Lose weight" or "Find a new job." -) End State Orientation:Formulate the main job steps with a clear end point or desired outcome, implying completion or achievement. Examples are "Buy a new car" or "Renovate the kitchen."

2. Quotes and Notes: For each job step, provide relevant quotes or paraphrased statements from the interviews.

3. Emotional aspects: that reflect how people want to feel while performing the job. Statements usually start with the word “feel.” For example, if the job step of a keyless lock system is to secure entryways to home, emotional jobs might be to feel safe at home or feel confident that intruders won't break in while away.

4. Social aspects: that indicate how a job performer is perceived by others while carrying out the job. For instance, adult diapers have an important social aspect of avoiding embarrassment in public. Or, in the previous example, the person with a keyless door lock might be seen as an innovator in the neighborhood.
   
5. For each Job Step you have to extract the possible Needs defined as follow. Each Need needs to be written as: Direction of change, unit of measure, object. -)Direction of change: How does the job performer want to improve conditions? Each need statement starts with a verb showing the desired change of improvement. Words like “minimize,” “decrease,” or “lower” show a reduction of unit of measure, while words like “maximize,” “increase,” and “raise” show an upward change. -)Unit of measure: What is the metric for success? The next element in the statement shows the unit of measure the individual wants to increase or decrease. Time, effort, skill, and likelihood are a few typical examples. Note that the measure may be subjective and relative, but it should be as concrete as possible. -)Object of the need: What is the need about? Indicate the object of control that will be affected by doing a job."""),

    ("human", """
This is the interview
{interview}

This is the output format exptected. Only a json with this structure. Do not put any additional characters other than the json file:
{{
    "analysis" {{
     [
        {{
            "Quote and note": "",
            "Job Step": "",
            "Emotional Aspect": "",
            "Social Aspect": "",
            "Need":""
        }}
     ]
    }}
}}

""")])
     

functions = [
    {
    "name": "interviews_analysis",
    "description": "interviews_analysis",
    "parameters": {
        "type": "object",
        "properties": {
            "analysis": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "Quote and note": {
                            "type": "string",
                            "description": "quotes and notes"
                        },
                        "Job Step": {
                            "type": "string",
                            "description": "Indicate steps in getting a job done and the micro-jobs you find during the interview. Be sure to begin each with a verb and omit any reference to technologies or solutions"
                        },
                        "Emotional Aspects": {
                            "type": "string",
                            "description": "Emotional Aspects"
                        },
                        "Social Aspects": {
                            "type": "string",
                            "description": "Social Aspects"
                        },
                        "Needs": {
                            "type": "string",
                            "description": "Needs"
                        }
                    }
                },
            },
        },
        "required": ["Quotes and notes", "Job steps", "Emotional Aspects", "Social Aspects","Needs"]
    },
    },
]

chain = (
    prompt 
    | model.bind(function_call={"name": "interviews_analysis"}, functions = functions))