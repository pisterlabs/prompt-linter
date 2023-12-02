import os
import openai
import guidance
from dotenv import load_dotenv
from dataclasses import dataclass

from helper.functions import json_response

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


@dataclass
class ResponseGenerator:
    questions: dict
    student: dict

    def __post_init__(self):
        self.gpt4 = guidance.llms.OpenAI("gpt-4")


    # Answers questions based on a student's grade, standard, topic, and number of questions. Provides potential grade for each answer and a potential final grade.
    def answer_questions(self):
        answer_questions = guidance('''
        {{#system~}}        
        Your task is to engage in academic role-play. You'll be portraying a student of a specific grade level, responding to a series of questions informed by a Common Core Learning Standard.
        Your foremost goal isn't just answering correctly, but emulating the educational level and understanding of the given Grade level. 
        The input you receive will consist of four elements: 
        1. The student's grade level: {{grade}}
        2. The related Common Core Learning Standard: {{standard}}
        3. The number of questions you should respond to: {{num_questions}}
        4. A dictionary of questions, each with a unique ID: {{questions}}

        With this data, you are to respond to each question in a way that corresponds to the understanding level of the student Grade you are simulating and of the Common Core Learning Standard. 
        Ensure that your responses vary in quality, capturing the full spectrum from 'Unacceptable' to 'Excellent', reflecting grades from 0 to 5 on our evaluation scale.

        Your output, returned in a JSON format, will contain the following elements for each question answered:
        1. The corresponding question's unique ID ("id"),
        2. Your answer to the question ("answer"),
        3. The grade you believe the answer would receive, based on our 0-5 scale ("grade").

        It will also include a cumulative grade ("finalGrade"), based on the individual grades of your responses.
        Keep in mind that each of your answers should not exceed a length of 750 characters. 

        Avoid including any specifics from test cases in your responses. 
        Be unique and creative in your approach, ensuring a diverse range of responses that accurately reflect the varying understanding at each Grade level.        
        Return the response only in JSON format (nothing else). 
        - - - - - - - - - - - - - - - - - - - - 
        Example JSON response:
            {
                "answers": [
                    {"id": 1, "answer": <answer>, "grade": <grade>}, 
                    {"id": ...},
                    ...
                ],
                "finalGrade": <finalGrade>
            }
        {{~/system}}
        {{#assistant~}}
                {{gen 'answer_questions' temperature=0.7 max_tokens=2000}}
        {{~/assistant}}
        ''', llm=self.gpt4)

        answer_questions = answer_questions(
            grade=self.student['grade'],
            standard=self.student['learning_standard'],
            num_questions=self.student['num_questions'],
            questions=self.questions['questions']
        )
        return json_response(answer_questions)