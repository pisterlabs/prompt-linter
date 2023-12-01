# api works for curl, python and node 
# for shell: curl -X POST https://api.openai.com/v1/engines/davinci/completions \
# for node: const openai = require('openai-api')

# --- Imports --- #
from openai import OpenAI
import os

# --- File Handling --- #
def file_reader(file_path):
    with open(file_path, 'r') as file:
        file_content = file.read()
        return file_content

file_paths = [

    "a_company_profile.txt",
    "b_job_descriptions.txt",
    "c_meeting_notes.txt",
    "d_project_docs.txt",
    "e_SOPs.txt",
    "f_team_structure.txt"
    
]

a_profile, b_job_descriptions, c_meeting_notes, d_project_docs, e_SOPs, f_team_structure = [
    file_reader(file_path) for file_path in file_paths]

# --- API --- #
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY_BM"], # API Key saved in environment variable
    # org_id=os.environ["OPENAI_ORG_ID"],
    # project_id=os.environ["OPENAI_PROJECT_ID"],
    # dev_mode=True
)

# --- Completion --- #
completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": f"You are a the virtual assistant for the company described in {a_profile}."},
    {"role": "user", "content": f"Tell me about your company."},
    {"role": "assistant", "content": f"Tell them about the company, and what it does, giving an example from the project file in {d_project_docs}."},
  ]
)

# --- Output --- #
print(completion.choices[0].message)
print("\n")
print(type(completion.choices[0].message)) # <class 'openai.types.chat.chat_completion_message.ChatCompletionMessage'>
