"DocString"
import os
from dotenv import load_dotenv
import pathlib

import math
import random
import openai

dotenv_path = pathlib.Path('.env')
load_dotenv(dotenv_path=dotenv_path)

key_index = random.randint(0, 4)
key_array = [os.getenv("KEY1"), os.getenv("KEY2"), os.getenv("KEY3"), os.getenv("KEY4"), os.getenv("KEY5")]

openai.api_key = key_array[key_index]


def gen_response(prompt):
    "DocString"
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{'role': 'system', 'content': prompt}]
    )
    response = completion.choices[0]['message']['content']

    return response


def model_soln(ans, model):
    "DocString"
    prompt = "give me a rough percentage to which is the first answer" + \
        "similar to the second considering the secod one is absolutely correct: " + \
        ans + " and " + model + \
        " Make sure to just print out the rough percentage without " + \
        "the percentage symbol and no other information"
    response = gen_response(prompt)
    response = response.replace("%", "")
    return int(response)


def from_question(question, ans):
    try:
        "DocString"
        prompt = "give me a rough percentage whether the answer: " + ans + \
            " is correct for the question" + question + \
            "Make sure to just print out the rough percentage without " + \
            "the percentage symbol and no other information."
        response = gen_response(prompt)
        response = response.replace("%", "")
        return int(response)
    except:
        return 0


def ai_marks(ans, total_marks, question, model=""):
    "DocString"
    total_per = 0
    count = 0
    if len(model) != 0:
        from_model = model_soln(question, ans)
        total_per += from_model
        count += 1
    if len(question) != 0:
        from_ques = from_question(question, ans)
        total_per += from_ques
        count += 1
    calc = (total_marks * total_per)/(100*count)
    return f"{math.floor(calc)}-{math.ceil(calc)}"
