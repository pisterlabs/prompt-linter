import os
import random
import json
from tracemalloc import start
import openai
from typing import Literal
from retry import retry
import time
from app.simple_redis import redis_store, redis_retrieve, redis_check

# openai.api_key = os.getenv("OPENAI_KEY")

openai.api_type = "azure"
openai.api_base = "https://silicongrovegpt.openai.azure.com/"
# openai.api_base = "https://thispagedoesnotexist.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = os.getenv("AZURE_OPENAI_KEY")
# openai.api_key = "fa67770c7d4143aa89117da2ebe19dd3"

@retry(tries=3, delay=0.2)
def _gpt(messages):
    print("trying...")
    print(openai.api_base)
    raise Exception("no calls right now thanks")
    response = openai.ChatCompletion.create(
        # model="gpt-3.5-turbo-0613",
        engine="gpt35-16k-sg",
        # engine="turbo-west",
        messages=messages,
        temperature=0.5,
        max_tokens=2300,
    )
    # print(response)
    print('yay!')
    # print(response)
    content = response["choices"][0]["message"]["content"]

    website = parse_html(content)

    return content, website

# @retry(tries=3, delay=0.2)
# def _gpt(messages):
#     print("trying...")
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo-0613",
#         messages=messages,
#         temperature=0.4,
#         max_tokens=1500,
#     )
#     # print(response)
#     print('yay!')
#     print(response)
#     content = response["choices"][0]["message"]["content"]

#     website = parse_html(content)

#     return content, website

SYSTEM_MESSAGE = """You are an AI programmer specialised in creating single-file demo websites. You are modelled after the world's best programmers, such as Jeff Dean and Grady Booch. Your programming skills are unparalleled, and you use them to perform the requests for your users. You always iterate on your design, to reach the best possible page."""

START_USER_MESSAGE = """Hey! You're the world's best programming AI expert, modelled after Jeff Dean and Grady Booch. Your skills in creating efficient, beautiful, one-page website demos are unparalleled.

Please, create a sample one-page landing page for a {theme} {type_}. Make up everything. Make sure the CSS is aligned with theme.
Don't include content like copyright notices or similar things! Try to not make the page too big too :)
You can use bootstrap, html5, css and javascript. You will make sure your answer is in a markdown codeblock, starting with "```html" and ending with "````". You must include everything within this code block, including the css <style> and javascript <script>. You cannot provide any other file but this one big html file.

Let's go!
"""

REFINE_1 = """Good start... Now make it look better! Improve on the design! Improve on the colour scheme... Ensure your website looks fantastic and very modern!"""

REFINE_2 = """You're doing great... Remember, you don't have access to images, so think of something to replace them. Maybe ASCII? Keep on improving. Self-critique and improve on the website, return the updated page in a code block."""

REFINE_PERSO = """This is good, but how about making it a bit more personalised? Give the website a name, write some content, don't just stick to the name by what it is! Return an improved version of the page in a code block."""

REFINE_4 = """Time to find some more... Jeff Dean himself would review the website, but he's busy at the moment. Please, try to make do without the review and improve the code. If you have clickable buttons, maybe open a small closable overlay on click? Return an improved version of the page based on your findings."""

REFINE_5 = """Okay, it's time to finish up, and add an ad if you can. Add some content and better design if you can. Please insert one of those three ads somewhere and return a code block."""


APPRAISAL = """As the lead AI Programming Expert modelled after Jeff Dean, you're shown this website made by """

REFINES = [REFINE_1, REFINE_2, REFINE_PERSO, REFINE_4, REFINE_5]

def store_website_in_redis(key: str, website: str, messages: dict, response: str, iteration: int=0, start: bool = False):
    key = f"{key}:website"
    if start:
        redis_json = {}
        redis_json['website'] = {}
        redis_json['website']['v0'] = website
        redis_json['most_recent'] = 0
    else:
        redis_json = redis_retrieve(key)
        redis_json['website'][f'v{iteration}'] = website
        redis_json['most_recent'] = iteration
    messages_to_store = messages + [{"role": "assistant", "content": response}]
    redis_json['messages'] = messages_to_store
    redis_store(key, redis_json)

def store_fetch_in_redis(key: str, start: bool = False):
    key = f"{key}:interaction"
    if start:
        redis_json = {}
        redis_json['interaction'] = 0
    else:
        redis_json = redis_retrieve(key)
        redis_json['interaction'] += 1
    redis_store(key, redis_json)

def generate_website(session_id: str):
    theme = ""
    type_ =""
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": START_USER_MESSAGE.format(theme=theme, type_=type_)},
    ]

    start_time = time.time()
    response, website = _gpt(messages)
    print(f"Generated in: {time.time() - start_time}")
    start_time = time.time()
    store_fetch_in_redis(key=session_id, start=True)
    store_website_in_redis(key=session_id, website= website, messages= messages, response= response, iteration= 0, start= True)
    print(f"Stored in: {time.time() - start_time}")

    return website

def fetch_iteration(key: str, target_interaction: int | str = "default"):
    if target_interaction == "default":
        current_interaction = redis_retrieve(f"{key}:interaction")['interaction'] + 1
    else:
        current_interaction = target_interaction
    current_website = redis_retrieve(f"{key}:website")['website'][f"v{current_interaction}"]
    store_fetch_in_redis(key=key)
    return current_website, current_interaction

def parse_html(response):
    try:
        assert "```html" in response
        assert "```" in response.split("```html")[1]
    except:
        print("______")
        print("______")
        print("ASSERTION ERROR")
        print("______")
        print("______")
        print(response)
        raise AssertionError
    return response.split("```html")[1].split("```")[0]

def iterate_on_website(session_id: str):
    for i in range(0, len(REFINES)):
        print(f"iteration {i} for {session_id}")
        print('doing this')
        if i == 4:
            if random.random() > 0.5:
                # Appraisal
                pass
        iteration = i + 1
        prompt = redis_retrieve(f"{session_id}:website")['messages']
        if len(prompt) >= 5:
            prompt = prompt[:2] + prompt[-3:]

        prompt.append({"role": "user", "content": REFINES[i]})

            # keep all the elements except the first assistant message, and the first user reply
            # we need to keep elements 0 and 1 because they are the system message and the first user message
        response, website = _gpt(prompt)
        store_website_in_redis(key=session_id, website= website, messages= prompt, response= response, iteration= iteration, start= False)
        print(f'stored iteration {iteration}')