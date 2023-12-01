# input ./shred
# output screen

import os
import sys
import openai
import time

# Get the OpenAI API key from the environment
openai.api_key = os.environ.get('OPENAI_API_KEY')

# Check if the API key is set
if openai.api_key is None:
    print("OpenAI API key not found. Make sure to set the OPENAI_API_KEY environment variable.")
    sys.exit(1)

# Define the initial prompt to initiate the conversation
init_prompt = """
You are a secretary to a departmental meeting. Your task is to write a concise meeting notes that highlighting important discussions, facts and decision. Tabulate facts and figure for references. Put issues that are similar in context together.
"""
# Generate the conversation with ChatGPT
conversation = [init_prompt]

# Set the directory containing the text files
directory = '/home/abi/minutes/shred'

# Set the directory containing the text files
file_list = os.listdir(directory)

# Iterate over each file in the list
for filename in file_list:
    file_path = os.path.join(directory, filename)

    with open(file_path, 'r') as file:
        # Read the first line of the file
        hd = file.readline().strip()
        print(hd)

        # Get the text from the file and format it
        text = ' '.join(file.read().splitlines())

        # Request summarization from ChatGPT
        prompt = [f"List the main points with title where each point not more than 2 paragraphs: {text}"]
        response = openai.Completion.create(
            engine='text-davinci-003',
            prompt='\n'.join(conversation + prompt),
            max_tokens=555,
            temperature=0.6
        )
        rs = response.choices[0].text.strip()
        print(rs)

    print("")
    time.sleep(1)
