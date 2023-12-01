
import openai
import os
import getpass

key = getpass.getpass(prompt='Enter your OpenAI API key: ')
openai.api_key = key

# prompt user to enter their text
#prompt = input('Enter your text: ')
#print('prompt is: ', repr(prompt))

prompt = 'tell me the name of the largest city in the world'

# roles => system, user, assistant
messages = [
    {'role': 'system', 'content':'Answer as detailed as possible, also while rhyming as much as possible.'},
    {'role': 'user', 'content':prompt},
]
response = openai.ChatCompletion.create(
    model = 'gpt-3.5-turbo',
    messages = messages,
    temperature = 0.8,
    max_tokens = 1000
)


"""
#openai.Completion.create()              # will take some paramaters and return a response object
response = openai.Completion.create(
    model='text-davinci-003',
    prompt=prompt,
    temperature=0.8,    # controls the randomness of the output, the higher the temp the more random. 0-2  default is 1.  0 is deterministic
    max_tokens=1000     # max generated for completion, 4096 is the max, input and output count to this limit
)
"""


#print(response)
print(response['choices'][0]['message']['content'])