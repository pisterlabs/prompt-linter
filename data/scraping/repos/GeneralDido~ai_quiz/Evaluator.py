import os
import openai
import guidance
from dotenv import load_dotenv
from dataclasses import dataclass

from helper.functions import json_response

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


@dataclass
class Evaluator:
    grade: str
    standard: str
    topic: str
    questions: list
    answers: dict

    def __post_init__(self):
        self.gpt4 = guidance.llms.OpenAI("gpt-4")


    # Generates evaluations based on a student's grade, Common Core Learning Standard, topic, questions, and students' answers.
    def generate_evaluations(self):
        evaluations = guidance('''
            {{#system~}}
            As an AI tutor, your task is to evaluate students' responses to open-ended questions based on a given topic, grade level, and Common Core learning standard. 
            Input: a Common Core learning standard: {{standard}}, a topic of interest: {{topic}}, a Student Grade: {{grade}}, a list of questions: {{questions}} and a dictionary with the students' answers to each of those questions: {{answers}}.
            For each question, you should evaluate the student's response based on the student's Grade level (i.e., the explanation for a 4th-grade student should differ from that of a 12th-grade student), Common Core Learning Standard, relevance to the topic, and the quality of the answer. 
            Please keep in mind that the student's answer can be at maximum 750 characters (so do not expect any answer to be longer than 750 characters at most).
            The grade should be appropriate for the Grade level of the student, for example if the student is in 4th grade, you should grade the answer as if it was written by a 4th grader.
            The quality of the answer should be rated as an integer on a scale from 0 to 5, with 0: Did not answer, 1: Unacceptable, 2: Needs Improvement, 3: Satisfactory, 4: Good and 5: Excellent. Rate the answer based on multiple factors, including the student's Grade level, understanding of the topic and the question, the depth and quality of response for the Grade level and the appropriate Learning Standard.
            In addition to the evaluation, you should provide an explanation for your evaluation that is relevant to the student's grade level and response. Use appropriate languge based on the Grade level. Your explanation should be respectful, helpful, insightful, and suggest possible improvements.
            Finally, you should calculate a total grade for the student based on all their responses (take the average of all grades) and provide final feedback.
            If the user has not responded at all to the particular question or has responded randomly, please address it in a nice, instructive way. Remember, your goal is to provide a supportive and constructive learning experience for the student, not just merely grade responses. Be respectful, insightful, creative, and thorough.
            
            Return the response only in JSON format (nothing else). Example JSON response:
            {"evaluations":[
                0:{
                    "id":<id>
                    "grade":<grade>
                    "explanation":<explanation>
                }
                1:{ ... }
                2:{ ... }
            ]
            "finalGrade":<finalGrade>
            "finalFeedback":<finalFeedback>
            }
            {{~/system}}
            {{#assistant~}}
                {{gen 'evaluations' temperature=0.7 max_tokens=3000}}
            {{~/assistant}}
        ''', llm=self.gpt4)

        evaluations = evaluations(
            grade=self.grade,
            standard=self.standard,
            topic=self.topic,
            questions=self.questions,
            answers=self.answers
        )
        return json_response(evaluations)
