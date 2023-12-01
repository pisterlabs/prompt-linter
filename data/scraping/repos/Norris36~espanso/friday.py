import os
import openai 
import pyperclip as py
import csv
from datetime import datetime
# Load the .env file
from dotenv import load_dotenv
load_dotenv()


openai.api_type         = os.getenv("OPENAI_TYPE")
openai.api_base         = os.getenv("OPENAI_BASE")
openai.api_version      = os.getenv("OPENAI_VERSION")
openai.api_key          = os.getenv("OPENAI_KEY")


def get_log_path():
   # The `return os.path.join(os.path.dirname(__file__), "data.csv")` line of code is returning the
   # full path of the `data.csv` file by joining the directory path of the current file with the
   # filename `data.csv`.
   return os.path.join(os.path.dirname(__file__), "gpt_log.csv")

#py.copy('Is greece bigger than Italy? ')

prompt = py.paste().strip()


base_message = [{"role":"system","content":"""
                As a Technical Developer with several years of experience in Python and Django, you will be responsible for creating data flow diagrams
                You will be provided with a set of paths, functions, and a short description, and you will be expected to write a short description of how the Python files call the function sending the data
                The description you write will be read by other developers and non-technical users, so it is important that its written in a clear fashion for both audiences.
                All paths should be written as from dotcom
                
                the description should be between 25-50 words, and should include the following:
                - the name of the event
                - the name of the event in the dotcom data
                - the files sending the data, with a reletaive path from dotcom and down
                - the files triggering the data, with a reletaive path from dotcom and down
                
                the paths should be relative to dotcom, and should be written as follows:
                - dotcom\path\to\file.py
                 
                                    """},
            {"role":"user",
            "content": f"""
            Please write a description of how the code triggers the event, and how the data flows to the database from the front end.
            {prompt}    
            """}
            #{"role":"assistant","content":"[your evaluation from 0 to 100, start with numbers] \n [your reasoning for the evaluation]"}
            ] 
message = base_message


response = openai.ChatCompletion.create(
    deployment_id = 'gpt-35-turbo',
    engine = 'gpt-35-turbo',
    messages = message,
    temperature = 0.2,
    max_tokens = 2000,
    top_p = 0.95,
    frequency_penalty = 0,
    presence_penalty = 0,
    stop = None
)

response = response['choices'][0]['message']['content']

timestamp = int(datetime.utcnow().timestamp())

path = r'C:\Users\jbay\AppData\Roaming\espanso\match\scripts\gpt_log.csv'

with open(get_log_path(), mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([prompt, response, timestamp])

# for i in range(len(prompt)):
#     response += f'\n i, {prompt[i]}'

print(response)
# except:
#     response = 'error' 
#     for i in range(len(prompt_list)):
#         response += f'\n {i}, {prompt_list[i]}'
#     print(response)