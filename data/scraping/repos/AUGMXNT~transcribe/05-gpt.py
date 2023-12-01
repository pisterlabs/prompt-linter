#!/usr/bin/env python
import os
import openai
import sys

openai.api_key = os.getenv("OPENAI_API_KEY")

def call_openai(content):
    print("Calling gpt-3.5-turbo-16k...")
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo-16k",
      messages=[
        {
          "role": "system",
          "content": "Reformat the following transcript into Markdown, bolding the speakers. Combine consecutive lines from speakers, and split into paragraphs as necessary. Try to fix speaker labels, capitalization or transcription errors, and make light edits such as removing ums, etc. There is some Danish, please italicize the Danish sentences."
        },
        {
          "role": "user",
          "content": content
        }
      ],
      temperature=0.1,
      max_tokens=8000,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    return response["choices"][0]["message"]["content"]

line_count = 0
max_lines = 100
input = ""
output = ""
with open("transcript.txt") as file:
    for line in file:
        input += line
        line_count += 1 

        if line_count >= max_lines:
            output += call_openai(input)
            line_count = 0
            input = ""


output += call_openai(input)

with open("transcript-edited-by-gpt3.5.md", "w") as file:
    file.write(output)

# print(output)
'''
real    4m53.263s
user    0m0.481s
sys     0m2.186s
'''
