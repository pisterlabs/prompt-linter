from distutils.command.clean import clean
from urllib import response
import openai
import os
import sys
import json
import requests
import time
import random


# the follow is a python script that will let rasa learn from an api
# the demonstration api is at the url: https://beta.openai.com/docs/api-reference/
# the api key is: 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
# 
# the completion  api is at the url: https://api.openai.com/v1/engines/davinci/completions
openai_api_key = "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"
openai_api_url = "https://api.openai.com/v1/engines/davinci/completions"

os.environ['OPENAI_API_URL'] = openai_api_url
os.environ['OPENAI_API_KEY'] = openai_api_key

def get_api_key():
    # get the api key from the environment variable
    api_key = os.environ.get('OPENAI_API_KEY')
    if api_key is None:
        print("Please set the environment variable OPENAI_API_KEY")
        sys.exit(1)
    return api_key


def get_url():
    url = os.environ.get('OPENAI_API_URL')
    if url is None:
        print("Please set the environment variable OPENAI_API_URL")
        sys.exit(1)
    return url

def discover_capabilities(url, api_key, npl_phrase):
    # create the first intent from the api
    # url = os.environ.get('OPENAI_API_URL')
    # api_key = os.environ.get('OPENAI_API_KEY')
    headers = {'Authorization': 'Bearer ' + api_key}
    data = {
        'api_key': api_key,
        'engine': 'davinci',
        'text': npl_phrase,
        'max_tokens': 100
    }
    response = requests.post(url, data=data)
    if response.status_code != 200:
        print("Failed to discover capabilities")
        print(response.text)
        sys.exit(1)
    else:
        print("Successfully discovered capabilities")
        print(response.text)
    print("Created intent")
    return response.json()

def create_intent_from_phrase(phrase):
    # create the first intent from the api
    # from the template update the phrase
    file = open("ask_openai_intent.txt", "r") 
    contents = file.read() 
    runtime_phrase = contents.replace("replace@runtime", phrase)
    file.close() 
    if(runtime_phrase == phrase):
        print("Error: could not replace the phrase")
        sys.exit(1)
    # print("NEW PHRASE:" + runtime_phrase)
    # print("ORIGINAL PHRASE:" + phrase)
    print("==intent_from_phrase==")
    response = do_completion(runtime_phrase)
    return response

def create_phrase_from_intent(intent):
    # create the first intent from the api
    print(intent)
    print("replace the intent with " + str(intent))
    file = open("ask_openai_phrase.txt", "r")
    contents = file.read() 
    runtime_intent = contents.replace("replace@runtime", str(intent)) 
    file.close() 
    if(runtime_intent == intent):
        print("Error: could not replace the phrase")
        sys.exit(1)
    # print("DOING THE POST:" + runtime_intent)
    # print("phrase:" + intent)
    print("==phrase_from_intent==")
    response = do_completion(intent)
    return response.json()

def create_phrase_from_phrase(phrase):
    # create the first intent from the api
    intent = create_intent_from_phrase(phrase)
    new_phrase = create_phrase_from_intent(intent)
    return new_phrase

def create_intent_from_intent(intent):
    phrase = create_phrase_from_intent(intent)
    new_intent = create_intent_from_phrase(phrase)
    return new_intent

def string_to_json(string):
    return json.loads(string)

def to_json(data):
    return json.loads(data)

## REQUEST PARAMTERS
# {
#   "prompt": "Say this is a test",
#   "max_tokens": 5,
#   "temperature": 1,
#   "top_p": 1,
#   "n": 1,
#   "stream": false,
#   "logprobs": null,
#   "stop": "\n"
# }

## RESPONSE
# {
#   "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
#   "object": "text_completion",
#   "created": 1589478378,
#   "model": "text-davinci-002",
#   "choices": [
#     {
#       "text": "\n\nThis is a test",
#       "index": 0,
#       "logprobs": null,
#       "finish_reason": "length"
#     }
#   ]
# }


def do_completion(text):
    url = get_url()
    api_key = get_api_key()
    #cleaned = text.join([chr(char) for char in range(1, 32)])
    cleaned = text
    headers = {'Authorization': 'Bearer ' + api_key}
    headers.update({'Content-Type': 'application/json'})
    data = {
        'prompt': cleaned,
        'max_tokens': 100
    }
    data = json.dumps(data)
    response = requests.post(url, data=data, headers=headers)
    if(response.status_code != 200):
        print("FAILED to talk to SENSEI: " + str(response.status_code))
        print(response.text)
        print("REQUEST WAS: " + str(data))
        sys.exit(1)
    data = response.text
    jsondata = json.loads(data)
    #print("do_completion json data:")
    #print("--returning from do_complete--")
    #print(jsondata['choices'])
    #sys.exit(1)
    return jsondata['choices'][0]['text']

# def davinci_completion(text, max_tokens=40, temperature=0.7, n=3):
#     params = {'text': text,
#               'max_tokens': max_tokens,
#               'temperature': temperature,
#               'top_p': 1.0,
#               'n': n}
#     headers = {'Content-Type': 'application/json'}
#     headers.update(('Authorization', 'Bearer ' + get_api_key()))
#     r = requests.post('https://api.openai.com/v1/engines/davinci/completions', headers=headers, json=params)

#     return r.json()

###
# This function takes in a phrase and returns a "better" phrase
# Better means that it is more likely to be a good phrase since we cycle through
# several iterations of the completion api
def dream_of_intents(intent, dream_value):
    while(dream_value):
        dream_value -= 1
        intent = create_intent_from_intent(intent)
        print("dream_of_intents...")
    return intent

def dream_of_phrases(phrase, dream_value=3):
    while(dream_value):
        print("dream_of_phrases...")       
        dream_value -= 1
        phrase = create_phrase_from_phrase(phrase)
    return phrase


# discover and write out the list of intents to a file
def bootstrap_rasa():
    gensis_phrase = "Hello A.I.. Can you tell me what your main capabilities are? What can you do and how should I communicate with you? A simple list is good"
    better_phrase = create_intent_from_phrase(gensis_phrase)
    print("BACK FROM CREATE")
    #completed_text = complete_text(phrase)
    #print(gensis_phrase)
    #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(better_phrase)
    #intent = create_intent_from_phrase(url, api_key, "Hello A.I. Can you tell me what you main capabilities are?")
    return
    response = requests.post(url, data=data)
    if response.status_code != 200:
        print("Failed to dicover capabilities")
        print(response.text)
        sys.exit(1)
    else:
        print("Successfully discovered capabilities")
        print(response.text)
    print("Created intent")
    return response.json()

if __name__ == "__main__":
    bootstrap_rasa()

