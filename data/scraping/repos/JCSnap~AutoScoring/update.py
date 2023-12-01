import os
import re
from flask import Flask, request, jsonify
import openai
import constants
import requests
import json
import time

openai.api_key = os.getenv("OPENAI_API_KEY")

with open("result.txt", "a") as file:
    for i in range(10):
        try:
            question_generation_instruction = constants.QUESTION_GENERATION_INSTRUCTION
            topic = "Big O Notation"
            grade = "University"
            prompt = constants.get_user_setting(
                grade, topic) + constants.MCQ_FORMAT

            print(f"Generating questions...")
            start_time_openai = time.time()
            completion = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": question_generation_instruction},
                    {"role": "user", "content": prompt}
                ]
            )
            end_time_openai = time.time()

            question = completion.choices[0].message.content

            # Use regular expressions to extract the lines
            lines = re.findall(
                r"(Question:.*|^(?!Subtopic:).*)", question, re.MULTILINE)

            # Create a new string without the subtopic line
            new_text = "\n".join(lines)

            original_question = new_text.strip()

            print(original_question)
            print(f"Time taken: {end_time_openai - start_time_openai}")

            accuracy_check_instruction = constants.ACCURACY_CHECK_INSTRUCTION + constants.MCQ_FORMAT
            prompt = question
            updated_bool = True

            print(f"Checking questions...")
            start_time_openai = time.time()
            completion = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": accuracy_check_instruction},
                    {"role": "user", "content": prompt}
                ]
            )
            text = completion.choices[0].message.content
            end_time_openai = time.time()

            print(text)
            time_taken_accuracy_check = end_time_openai - start_time_openai
            print(f"Time taken: {end_time_openai - start_time_openai}")

            updated_whole = text

            # Extract the updated information
            updated_match = re.search(r"Updated:([\s\S]*)", text)
            if updated_match:
                updated = updated_match.group(1).strip()
            else:
                updated = "false"

            status_question = text[text.find("Updated: ") + len("Updated: "):]
            if status_question == "false" or status_question == "False" or status_question == "false." or status_question == "False.":
                print("Updated")
                updated_bool = False
            print(status_question)

            file.write("Topic: " + topic + "\n")
            file.write("Grade: " + grade + "\n")
            file.write("Original Question: " + original_question + "\n")
            file.write("Is Updated: " + str(updated_bool) + "\n")
            file.write("Updated: " + updated + "\n")
            file.write("Time taken for accuracy check: " +
                       str(time_taken_accuracy_check) + "\n\n")
        except Exception as e:
            print(e)
            continue
