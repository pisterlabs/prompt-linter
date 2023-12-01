#!/usr/bin/env python3
""" A simple script to summarize text using OpenAI's API.

0. Set the OPENAI_API_KEY environment variable to your OpenAI API key.
1. Install the required packages first 
$pip install pyperclip openai

# Ref: 
# https://github.com/htlin222/dotfiles/blob/main/pyscripts.symlink/split_translate_zh_only.py?fbclid=IwAR0oCEu8mlN5KzQ1OGv9-Pqyqq7qzRwvv8BV587qxfYvSD5vhT8whOoXnJE
# 
"""
import os
import pyperclip as pc
import openai

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
PROMPT_PREFIX="Summrize this to four items for me: "

def retreive_text():
    """Retreive the text from the clipboard.
    """
    text = pc.paste()
    return text

def combine_prompt(text):
    """Combine the prompt prefix and the text."""
    prompt = PROMPT_PREFIX + text
    return prompt


def openai_api_call(prompt):
    """Call the OpenAI API."""
    openai.api_key = OPENAI_API_KEY
    response = openai.Completion.create(
      engine="text-davinci-003",
      prompt=prompt,
      temperature=0.7,
      max_tokens=1000,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    return response


def copy_to_clipboard(response):
    """Copy the summary to the clipboard."""
    summary = response.choices[0].text
    # Export the summary to the clipboard
    print(summary)

if __name__ == "__main__":

    # 1. Retrieve the text from the clipboard
    text = retreive_text()
    # 2. Combine the prompt prefix and the text
    prompt = combine_prompt(text)
    # 3. Call the OpenAI API
    response = openai_api_call(prompt)
    # 4. Copy the summary to the clipboard
    copy_to_clipboard(response)
    
    




    