import os
import openai
import guidance
from dotenv import load_dotenv
from dataclasses import dataclass

from helper.functions import json_response

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


@dataclass
class TopicGenerator:
    grade: str
    standardTopic: str
    standardNum: str
    numQuestions: int

    def __post_init__(self):
        self.gpt4 = guidance.llms.OpenAI("gpt-4")


    # Generates a "student" with a grade, standard, topic, and number of questions.
    def generate_topic(self):
        topic = guidance('''
            {{#system~}}        
                Your task is to generate a relevant topic based on the student's Grade level and a specific Core Learning Standard, which you will also generate.
                This is the Core Learning Standard Topic: {{standardTopic}}
                This is the student Grade level: {{grade}}
                This is the Core Learning Standard number: {{standardNum}}
                You will generate {{numQuestions}} questions.
                            
                Based on the Grade, Core Learning Standard Topic, and Core Learning Standard number, you will generate a specific Core Learning Standard and its definition. 
                As an example: Grade: 4, Core Learning Standard: Writing, Core Learning Standard number: 9 should give: CCSS.ELA-LITERACY.W.4.9 and definition: Draw evidence from literary or informational texts to support analysis, reflection, and research.
                
                You will generate a topic of interest for the student, based on the Grade level of the student and the Core Learning Standard. 
                The topic of interest should be relevant to the potential student, for example, if the student is in 4th grade, the topic should be relevant to 4th graders.
                Please think of new examples to provide each time. Avoid using the given examples when creating the output. Only create unique outputs based on the selections made.                       
                Return the response only in JSON format (nothing else). 
                - - - - - - - - - - - - - - - - - - - - 
                Example JSON response:
                {
                    "grade": "4",
                    "standard_topic": "Writing",
                    "standard_num": "9",
                    "learning_standard": "CCSS.ELA-LITERACY.W.4.9",
                    "learning_standard_definition": "Draw evidence from literary or informational texts to support analysis, reflection, and research",
                    "topic": "Baseball",
                    "num_questions": 3
                }
            {{~/system}}
            {{#assistant~}}
                {{gen 'topic' temperature=0.7 max_tokens=750}}
            {{~/assistant}}
        ''', llm=self.gpt4)

        topic = topic(
            grade=self.grade,
            standardTopic=self.standardTopic,
            standardNum=self.standardNum,
            numQuestions=self.numQuestions
        )
        return json_response(topic)