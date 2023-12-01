import reasoner
import chatgpt

import json
import os
# Make a key: https://platform.openai.com/account/api-keys
# os.environ["OPENAI_API_KEY"] = "<your key here>"
os.environ["OPENAI_API_KEY"] = open('openai_key.txt', 'r').read().strip('\n')

pdf_data = """
18.443 Problem Set 3 Spring 2015
Statistics for Applications
Due Date: 2/27/2015
prior to 3:00pm

Problems from John A. Rice, Third Edition. [Chapter.Section.Problem]

1. Problem 8.10.21.

2. Problem 8.10.45. A Random walk Model for Chromatin
Only parts (a) through (g). See R script file “Rproject3.script4.Chromatin.r” in Rproject3. You can edit this file for your answers; turn in hard-copy of an html file compiled by creating a “notebook” from the script (press the button on the script window that looks like a notebook). Circle/highlight your answers on the hard-copy.

3. Problem 8.10.51 Double Exponential (Laplace) Distribution

4. Problem 8.10.58 Gene Frequencies of Haptoglobin Type See R Script file “Rproject3.script1.multinomial.simulation.r” in Rproject3. You can edit this file for your answers; turn in hard-copy of an html file compiled by creating a “notebook” from the script (press the button on the script window that looks like a notebook). Circle/highlight your answers on the hard-copy.

1

MIT OpenCourseWare
http://ocw.mit.edu

18.443 Statistics for Applications
Spring 2015

For information about citing these materials or our Terms of Use, visit: http://ocw.mit.edu/terms.

"""



import openai
openai.api_key = open('openai_key.txt', 'r').read().strip('\n')

def get_completion(system_prompt, msg, model="gpt-3.5-turbo"):
    messages = [{
            "role": "system",
            "content": system_prompt + " DO NOT OUTPUT ANY MORE TEXT. BE A ROBOT.",
        },
        {
            "role": "user",
            "content": msg,
        }]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    # print(response)
    return response.choices[0].message["content"]

def get_pdf_data(pdf_data):
    # title_and_author = get_completion("ONLY OUTPUT THE TEXTBOOK NAME AND AUTHOR.", pdf_data)
    arr = get_completion("""ONLY RETURN A STRING ARRAY OF EACH PROBLEM IN THIS FORMAT ["Chapter.Section.Problem"].""", pdf_data)
    print(arr)
    print(json.loads(arr))
    problem_data =  json.loads(arr)[0].split("."),
    print(problem_data)
    return {
        "title_and_author": title_and_author,
        "chapter": problem_data[0],
        "section": problem_data[1],
        "problem": problem_data[2],
    }
# print(get_pdf_data(pdf_data))