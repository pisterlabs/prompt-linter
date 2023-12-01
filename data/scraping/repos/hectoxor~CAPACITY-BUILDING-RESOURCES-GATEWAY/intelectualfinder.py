import os
import openai

openai.api_key = ""

def knowtxt(filename):

    fileex = filename + "\n"

    gpt_prompt = fileex

    response = openai.Completion.create(
    model="text-davinci-002",
    prompt=gpt_prompt,
    temperature=0.3,
    max_tokens=150,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
    )
    
    return(response['choices'][0]['text'])
