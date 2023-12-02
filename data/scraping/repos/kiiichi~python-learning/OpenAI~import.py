import openai
import os

'''
# Load API key from .env file
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
'''

openai.api_key = os.environ.get('OPENAI_API_KEY')

def get_completion(prompt, model='gpt-3.5-turbo'):
    messages = [{'role': 'user', 'content': prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the output 
    )
    return response.choices[0].message['content']

text = f"""
Hello, how are you? \
hummm... I'm not sure I understand what you mean. \
I'm a chatbot, I'm here to help you. \
I'm sorry, I don't understand. \
I'm sorry, I don't understand. \
I'm sorry, I don't understand. \
"""
prompt = f"""
Summarize the text delimited by triple backticks \
into a single  sentence. 
```{text}```
"""
response = get_completion(prompt)
print(response)

prompt = f"""
Generate a list of 3 made-game tiles along with their \
descriptions, genres, and titles.
Provide them in JSON format with the following keys: 
game_id, title, description, genre, price.
"""
response = get_completion(prompt)
print(response)