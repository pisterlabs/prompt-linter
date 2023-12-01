''' Application to create exam questions from a topic using a GPT-3 engine'''

import openai
from dotenv import load_dotenv
import os

# load environment variables from .env file
load_dotenv()

# get api key from environment variable
openai.api_key = os.environ["OPENAI_API_KEY"]

# Create the test prompt (pre LLM)
def create_test_prompt_pre_LLM(topic, num_questions, num_possible_answers):
    prompt = f"Topic: {topic}\n\n"
    for i in range(num_questions):
        prompt += f"Question {i+1}: "
        prompt += "__________\n\n"
        for j in range(num_possible_answers):
            prompt += f"- Option {j+1}\n"
        prompt += "\n"
    return prompt

# prompt = create_test_prompt_pre_LLM("Python", 4, 4)
# print(prompt)

# Create the test prompt for input to a LLM
def create_test_prompt(topic, num_questions, num_possible_answers):
    prompt = f"""Create a multiple choice quiz on the topic of {topic} consisting of {num_questions} questions.
                 Each question should have {num_possible_answers} as options.
                 Also include the correct answer for each question usinf the starting string 'Correct Answer: '"""
    return prompt

# Create the students view of the test from the response
def create_student_view(test, num_questions):
    student_view = {1: ''}
    question_number  = 1
    for line in test.split("\n"):
        if not line.startswith('Correct Answer'):
            student_view[question_number] += line + '\n'
        elif question_number < num_questions :
            question_number += 1
            student_view[question_number] = '' 

    return student_view

# Create a list of answers from the response
def extract_answers(test, num_questions):
    answers = {1: ''}
    question_number  = 1
    for line in test.split("\n"):
        if line.startswith('Correct Answer'):
            answers[question_number] += line + '\n'

            if question_number < num_questions :
                question_number += 1
                answers[question_number] = '' 

    return answers

# Set the exam
def take_exam(questions):
    student_answers = {}
    for question, question_view in questions.items():
        print(question_view)
        student_answers[question] = input("Answer: ")
        print("\n")

    return student_answers

#Grade the exam
def grade_exam(answers, student_answers):
    score = 0
    for question, answer in student_answers.items():
        if answer.upper() == answers[question][16]:
            score += 1
    grade = score / len(student_answers)*100

    return grade

prompt = create_test_prompt("USA History", 4, 4)

response = openai.Completion.create(engine="text-davinci-003",
                                    prompt=prompt,
                                    max_tokens=256,
                                    temperature=0.7)
#aprint(response['choices'][0]['text'])

questions = create_student_view(response['choices'][0]['text'], 4)
# for key in questions:
#     print(f"Question {key}:")
#     print(questions[key])
#     print("\n")

answers = extract_answers(response['choices'][0]['text'], 4)

# for key in answers:
#     print("Question "+str(key) +" "+ answers[key])

# Take the exam
student_answers = take_exam(questions)
# print(student_answers)

# Grade the exam
grade = grade_exam(answers, student_answers)
if grade < 60:
    print(f"Your grade is {grade}. You failed the exam")
else:
    print(f"Your grade is {grade}. You passed the exam")

