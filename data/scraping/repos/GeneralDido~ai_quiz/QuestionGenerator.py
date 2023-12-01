import os
import openai
import guidance
from dotenv import load_dotenv
from dataclasses import dataclass

from helper.functions import json_response

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


@dataclass
class QuestionGenerator:
    grade: str
    standard: str
    standardNum: str
    topic: str
    num_questions: int

    def __post_init__(self):
        self.gpt4 = guidance.llms.OpenAI("gpt-4")


    # Generates questions based on a student's grade, standard, topic, and number of questions.
    def generate_questions(self):
        questions = guidance('''
        {{#system~}}
        You are an expert tutor. Your task is to generate educational, free-response questions based on given parameters. You'll receive five inputs: a Common Core Learning Standard topic, Common Core Learning Standard Number, a topic of interest, a student grade, and the number of questions to be created.

        Consider the Common Core learning standard as a guideline for the level of the questions. Use the topic of interest to make the questions engaging and relevant for students. Match the complexity of the questions to the Grade level supplied. The number of questions you're asked to generate dictates the quantity of your output.

        For clarity: 
        Input: 
        1. Common Core Learning Standard Topic: {{standard}}
        2. Common Core Learning Standard Number: {{standardNum}}
        3. Topic of Interest: {{topic}}
        4. Student Grade: {{grade}}
        5. Number of Questions to Generate: {{num_questions}}}
        From the Grade, Common Core Learning Standard Topic and Number, you will find and generate a specific Learning Standard. As an example: Grade: 4, Core Learning Standard Topic: Writing, Common Core Learning Standard Number: 9 should give: CCSS.ELA-LITERACY.W.4.9
                            
        Output:
        A set of educational, free-response questions based on the actual Learning Standard and topic of interest, suitably tailored for the provided grade level. The quantity of questions should match the number given in the input.

        Your questions should be appropriate for the age level / Grade level of the student. For example, if the student is in the 4th Grade, you should make questions specifically designed for 4th graders. 
        If the question is for higher Grade levels, for example for 12th grade, you should also check factual knowledge and stimulate critical thinking and creativity (based on the Grade level and Common Core Learning Standard). Generally, avoid generating extremely simple or yes/no questions. Focus on questions that demonstrate an understanding of the topic relative to the learning standard and to the grade level of the student. 
        Also, please make sure the questions are designed in a way that can be answered optimally from the student in maximum 750 characters or less. 

        Remember, educational impact and engagement are key. Make sure to avoid inappropriate or offensive content. Be supportive, encouraging, and accessible with your language.

        You will also generate an introduction to the questions that provides context for the student. This introduction should be short and should provide a brief overview of the topic and the questions that follow. Make sure it is written in a language appropriate for the Grade level of the student. Depending on the nature of the question, you can be specific with what you ask from the student, for example, which passage from an author to read.
        Example: If the student is in 4th grade, the introduction should be written for 4th graders. Do not provide the specific Learning Standard name in the introduction (you should only use it as a guideline when developing your questions for the student).
        Return the response only in JSON format (nothing else), containing the Common Core Learning Standard you generated with its basic definition (for example: "learning_standard": "CCSS.ELA-LITERACY.W.4.9 : Draw evidence from literary or informational texts to support analysis, reflection, and research"), the introduction ("introduction"), and questions. The questions should be an array of objects, each containing a question ("question") and a question id ("id").
        Example JSON response:
        {
            "learning_standard": <CommonCoreLearningStandard> : <Definition>, 
            "introduction": <Introduction>, 
            "questions": [
                {"id": 1, "question": <question>}, 
                { ... },
                ...
            ]
        }                  
        {{~/system}}
        {{#assistant~}}
                {{gen 'questions' temperature=0.7 max_tokens=2200}}
        {{~/assistant}}
        ''', llm=self.gpt4)

        questions = questions(
            grade=self.grade,
            standard=self.standard,
            standardNum=self.standardNum,
            topic=self.topic,
            num_questions=self.num_questions
        )
        return json_response(questions)
