import os
from flask import Flask, request, jsonify
import openai
import constants
import requests
import json
import time

# export OPENAI_API_KEY="YOUR_API_KEY"


def format_answer_without_justification(question, model_answer, student_answer):
    template = """
        Question: {}
        Model answer: {}
        Student answer: <> {} <>

        Format:
        Score: [score]
    """.format(question, model_answer, student_answer)
    return template


openai.api_key = os.getenv("OPENAI_API_KEY")

instruction = "You are a teacher. You will be given a question, a model answer, and a student's answer. You are tasked to grade your student's answer out of 100 based on its accuracy when compared to the model answer. There can be some leeway and the answer need not follow the model answer word for word. Do not provide explanation, just give the score."

ALL_QUESTIONS = [constants.QUESTIONS_WRONG, constants.QUESTIONS_WRONG2]

for questions in ALL_QUESTIONS:
    with open("results4.txt", "a") as file:
        for question in questions:
            prompt = format_answer_without_justification(
                question["question"], question["model_answer"], question["student_answer"])

            scores = [None] * 5
            for i in range(5):
                print("Sending request to OpenAI API...")
                completion = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": prompt}
                    ]
                )
                text = completion.choices[0].message.content

                print(text)

                # Find the index of "Score: "
                start_index_s = text.find("Score: ")

                # Extract the score value
                scores[i] = text[start_index_s + len("Score: "):].strip()

                print(f"Openai Score: {scores[i]}")

            # Write the results to the file
            file.write("Question: " + question["question"] + "\n")
            file.write("Model Answer: " + question["model_answer"] + "\n")
            file.write("Student Answer: " + question["student_answer"] + "\n")
            for i in range(5):
                file.write("OpenAI Scores: " + scores[i] + "\n")
            file.write("\n\n")
