import openai
import os 
import getpass

key = getpass.getpass(prompt='Enter your OpenAI API key: ')
openai.api_key = key

openai.api_key = ''

#prompt = input('Enter your text: ')
prompt = 'give me a motto for a futuristic motorcycle company'

# roles => system, user, assistant
messages = [
    {'role': 'system', 'content':'you are a good and smart assistant'},
    {'role': 'user', 'content':prompt},
]
response = openai.ChatCompletion.create(
    model = 'gpt-3.5-turbo',
    messages = messages,
    temperature = 1,
    top_p = 0.8,
    max_tokens = 1000,
    n = 2
)

print(response['choices'][0]['message']['content'])
print(response['choices'][1]['message']['content'])