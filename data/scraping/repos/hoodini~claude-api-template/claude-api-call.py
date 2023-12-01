import requests
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import json
import time

API_KEY = '<INSERT_API_KEY_HERE>'

def get_claude_response(user_input):
    url = 'https://api.anthropic.com/v1/complete'
    headers = {
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json',
        'x-api-key': API_KEY  
    }

    data = {
        'model': 'claude-2',
        'prompt': f"{HUMAN_PROMPT}{user_input}{AI_PROMPT}",
        'max_tokens_to_sample': 256,
        'stream': True
    }

    response = requests.post(url, headers=headers, json=data, stream=True)

    for line in response.iter_lines():
        decoded_line = line.decode('utf-8').strip()
        if decoded_line.startswith('data:'):
            json_data = json.loads(decoded_line[5:])
            if 'completion' in json_data:
                for char in json_data['completion']:
                    print(char, end="", flush=True)
                    time.sleep(0.1)

# Get user input
user_input = input("Ask something: ")
# Get response from Claude API
get_claude_response(user_input)
