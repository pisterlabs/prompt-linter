import os
import openai

openai.api_key = ""

def meentxt(filename):
    #open text file in read mode
    text_file = open(filename, "r")
    
    #read whole file to a string
    fileex = text_file.read(4000) + "\n"
    
    #close file
    text_file.close()
    gpt_prompt = "Correct this to standard English:\n\n" + fileex

    response = openai.Completion.create(
    model="text-davinci-002",
    prompt=gpt_prompt,
    temperature=0.5,
    max_tokens=150,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
    )

    return(response['choices'][0]['text'])
print(meentxt('ACRIMSAT.txt'))
