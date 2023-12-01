#!/usr/bin/env python

import openai
import argparse
import os
import pyperclip  # This is for copying to clipboard

# Set up the argument parser
parser = argparse.ArgumentParser(description='Generate insights on a file.')
parser.add_argument('filename', type=str, help='The name of the file to generate insights on.')
parser.add_argument('-i', '--insight', action='store_true', help='Generate insights on the file.')
parser.add_argument('-r', '--review', action='store_true', help='Review the file for improvements.')
parser.add_argument('-c', '--clipboard', action='store_true', help='Copy the output to clipboard.')

# Parse the arguments
args = parser.parse_args()

# Read the file
with open(args.filename, 'r') as file:
    file_contents = file.read()

# Get the path of the script
script_dir = os.path.dirname(os.path.realpath(__file__))
# Get the path of the prompt files
prompt1_path = os.path.join(script_dir, 'prompt1.txt')
prompt2_path = os.path.join(script_dir, 'prompt2.txt')


# Set prompt1 = prompt1.txt, so on and so forth
with open(prompt1_path, 'r') as file:
    prompt1 = file.read() + '\n\n' + file_contents
with open(prompt2_path, 'r') as file:
    prompt2 = file.read() + '\n\n' + file_contents

# Set up the OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"];

# Generate insights using GPT-3.5
if args.insight:
    response = openai.Completion.create(
      engine="text-davinci-003",
      prompt=prompt1,
      temperature=0.5,
      max_tokens=100, # max amount is 2048
    )
    output = response.choices[0].text.strip()
    print(output)
    if args.clipboard:
        pyperclip.copy(output)

# Conduct a review
if args.review:
    response = openai.Completion.create(
      engine="text-davinci-003",
      prompt=prompt2,
      temperature=0.5,
      max_tokens=1000,
    )
    output = response.choices[0].text.strip()
    print(output)
    if args.clipboard:
        pyperclip.copy(output)

