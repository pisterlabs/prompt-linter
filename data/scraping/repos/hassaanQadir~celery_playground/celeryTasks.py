""" from celery import Celery, group, chord
import time
import re
import os
import openai
from dotenv import load_dotenv


app = Celery('celeryTasks', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')

# celery -A celeryTasks worker --loglevel=info -c 9

# Load environment variables
load_dotenv('.env')

# Use the environment variables for the API keys if available
openai.api_key = os.getenv('OPENAI_API_KEY')

@app.task
def complete(prompt):
    # Specify the model you want to use
    model = 'text-davinci-003'

    # Generate a completion using the OpenAI API
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=50  # Adjust the number of tokens as per your requirements
    )

    # Return the generated completion
    return response.choices[0].text.strip()


@app.task
def agent1(input):
    time.sleep(1)
    answer = "We received 1 query of" + input + " and we output lePhase 1, lePhase 2, and lePhase 3"
    return answer

@app.task
def agent2(input):
    time.sleep(1)
    answer = "We received" + input + " and we output theStep 1, theStep 2, and theStep 3"
    return answer

@app.task
def agent3(input):
    time.sleep(1)
    answer = "We received" + input + " and we output Substep 1, Substep 2, and Substep 3"
    return answer

@app.task
def agent4(input):
    time.sleep(1)
    answer = "We received" + input + " and we output Command 1, Command 2, and Command 3"
    return answer

@app.task
def agent5(input):
    time.sleep(1)
    answer = "We received" + input + " and we output API code in 1 line"
    return answer

@app.task
def extract(s):
    # Match any number and get it with its preceding characters
    pattern = r'(.{0,9}\d+)'
    matches = re.findall(pattern, s)
    # Truncate each match to the last 9 characters plus the number
    answer = [match[-10:] for match in matches]
    # Remove the first entry from the list, which is just repeating the function's input
    cleanAnswer = answer[1:]
    return cleanAnswer

@app.task
def process_results(results):
    # This is a Celery task that processes the results from all the tasks in the group
    # The input 'results' is a list of the outputs from each task in the group
    theList = []
    for result in results:
        theList.extend(result)
    return theList
 """