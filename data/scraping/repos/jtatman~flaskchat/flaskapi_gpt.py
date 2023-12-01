import time
from datetime import datetime
import calendar
import os
import csv
import json
import requests
from openai import OpenAI
from google.api_core import retry
import cohere

'''
storywriter using multiple llms and a flask restapi backend for memory
requires: two llms - this examples uses openai and cohere, but two openais
    or two llamas may be utilized instead
todo: implement better summarizing for different models depending on
    context length and "rope" possibilities
'''

co = cohere.Client(os.environ['COHERE_API_KEY'])
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

@retry.Retry()
def openai_response(prompt_template):
    try:
        background = "You are a sentient AI that is in charge of developing stories. Please respond to input as if you were in charge of a scriptwriting department. Do not end responses with a question, but make sure the story ends in a complete thought."
        chat_completion = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            temperature=0.75,
            max_tokens=4096,
            messages = [
                    {"role": "system", "content": background},
                    {"role": "user", "content": prompt_template}
            ]
        )
        response = chat_completion.choices[0].message.content
        return response
    except Exception as e:
        if e:
            print(e)
            print("timeout error on openai, next iter...")
            time.sleep(15)
        else:
            raise e

@retry.Retry()
def cohere_response(prompt_template):
    try:
        prompt_template = prompt_template[:1512]
        system_template = f"You are a sentient AI that is in charge of developing stories. Please respond to input as if you were in charge of a scriptwriting department. Do not end responses with a question, but make sure the story ends in a complete thought.\nHere is the current story: {prompt_template}"
        completion = co.generate(
            model="command-nightly",
            prompt = system_template,
            max_tokens=2048,
            temperature=0.75,
        )
        res = completion.generations[0].text
        return res
    except Exception as e:
        if e:
            print(e)
            print("timeout error on cohere, next iter...")
            time.sleep(15)
        else:
            raise e

def write_response(response, prompt, model):
    #date = datetime.utcnow()
    #timestamp = calendar.timegm(date.utctimetuple())
    timestamp = datetime.utcnow().isoformat()
    text_file = "flaskapi_prompts.txt"
    csv_file = "flaskapi_prompts.csv"
    json_file = "flaskapi_prompts.jsonl"

    with open(text_file, "a") as f:
        if response is not None:
            f.write('"' + str(response) + '"')

    with open(csv_file, "a") as f:
        if response is not None:
            writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
            csvdata = [ timestamp, model, prompt, '"' + str(response) + '"' ]
            writer.writerow(csvdata)

    with open(json_file, "a") as f:
        if response is not None:
            response_line = {
                "model": model,
                "timestamp": timestamp,
                "prompt": prompt,
                "text": response
            }
            json.dump(response_line, f, default=str)
            f.write('\n')


prompt_template = f"You are a prompt writing machine. You have all the ideas for all the prompts and story writing. Please develop a thorough story using the idea: Explore the idea of parallel universes through the lens of a character who can access different versions of their own life. How does this impact their sense of self and purpose?"

post_url = 'http://localhost:4231/post'
get_url = 'http://localhost:4231/get'

# insert prompt template into flask queue
data = {"message": prompt_template}
json_string = json.dumps(data)
initial_post = requests.post(post_url, json=json.loads(json_string))
if initial_post.status_code == 200:
    print("successful post: " + str(initial_post.status_code))
else:
    print("post did not complete...")

for i in range(25):
    # get flask queue here
    response = requests.get(get_url)
    if response.status_code == 200:
        messages = response.json()
        if isinstance(messages, list):
            # extract message strings from JSON array
            message_strings = [f"{message['message']}" for message in messages]
        else:
            # extract message from single json object
            message_strings = [f"{messages['message']}"]
        # convert list of message strings into a single string with newlines and commas
        result_string = ',\n'.join(message_strings)
        print(result_string)
    else:
        print(f"Error: {response.status_code}")
    if result_string is not None:
        prompt_template = f"You are a story writing machine. You have all the ideas for all the stories. Please develop a thorough story using these continuing ideas: {result_string}"
        openai_res = openai_response(prompt_template)
        if openai_res:
            try:
                # Flask app expects JSON data in the request body
                data = {"message": openai_res}
                json_string = json.dumps(data)
                # send the POST request
                post_response = requests.post(post_url, json=json.loads(json_string))
            except Exception as e:
                print(e)
            print("openai response: " + str(openai_res))
            write_response(openai_res, prompt=prompt_template, model="openai")
        else:
            print("no openai response")
    else:
        print("no result_string returned")

    response = requests.get(get_url)
    if response.status_code == 200:
        messages = response.json()
        if isinstance(messages, list):
            # extract message strings from JSON array
            message_strings = [f"{message['message']}" for message in messages]
        else:
            # extract message from single json object
            message_strings = [f"{messages['message']}"]
        # convert list of message strings into a single string with newlines and commas
        result_string = ',\n'.join(message_strings)
        print(result_string)
    else:
        print(f"Error: {response.status_code}")

    if result_string:
        prompt_template = f'You are a story writing machine. You have all the ideas for all the stories. Please assist in developing a story using these continuing ideas: {result_string}'
        cohere_res = cohere_response(prompt_template)
        if cohere_res:
            try:
                # flask app expects JSON data in the request body
                data = {'message': cohere_res}
                json_string = json.dumps(data)
                # send POST request
                post_response = requests.post(post_url, json=json.loads(json_string))
            except Exception as e:
                print(e)
            print("cohere response: " + str(cohere_res))
            write_response(cohere_res, prompt=prompt_template, model="cohere")
        else:
            print("no cohere response")
    else:
        print("no result_string returned")

