# This is a solution to the functions task from aidevs2 course
# See https://zadania.aidevs.pl/ for details
#
#
# The overall goal is to:
# 1. Get a token from aidevs
# 2. Get the task from aidevs
# 3. Write down the function call structure
# 4. Send the answer to aidevs
# 5. Profit!

import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
import json
import yaml
import os.path
import os
import openai

def load_apikey():
    """Loads the aidevs API key from ~/.aidevs2"""
    with open(os.path.expanduser('~/.aidevs2'), 'r') as file:
        api_key = yaml.safe_load(file)
    return api_key['APIKEY']

def load_openai_key():
    """Loads the openai API key from ~/.aidevs2"""
    with open(os.path.expanduser('~/.aidevs2'), 'r') as file:
        api_key = yaml.safe_load(file)
    return api_key['OPENAI_KEY']

BASE_URL = 'https://zadania.aidevs.pl'
APIKEY = load_apikey()
OPENAI_KEY = load_openai_key()
TASK = 'functions'

# STEP 1: Get the token from aidevs
url = BASE_URL + '/token/' + TASK
print(f'aidevs: Getting {url}, sending {APIKEY}')
response1 = requests.post(url, json={ "apikey": APIKEY })
data = json.loads(response1.text)
token = data['token']
print(f"aidevs: My token is {token}")

# STEP 2: Get the task from aidevs
url = BASE_URL + '/task/' + token

query={  }
print(f"aidevs: Sending {query} to {url}")
response2 = requests.post(url, data=query)
data2 = json.loads(response2.text)

print(f"aidevs: the task is {data2}")

# STEP 3: SOLUTION
# This is as easy as it gets. Just write down the function call structure.
answer = {"answer": {
            "name": "addUser",
            "description": "adds new user to a database",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "first name of the user"
                    },
                    "surname": {
                        "type": "string",
                        "description": "surname of the user"
                    },
                    "year": {
                        "type": "number",
                        "description": "year of birth"
                    },
                }
            }
        }
    }

# STEP 4: Send the answer
url = BASE_URL + '/answer/' + token
print(f'aidevs: Getting {url}, sending {answer}')
response3 = requests.post(url, json = answer)
data3 = json.loads(response3.text)
print(f"aidevs: /answer/token returned {data3}")
