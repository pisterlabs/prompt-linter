# --------------------------------------------------------------
# PROJECT BILLABLE HOURS DAYS
# --------------------------------------------------------------
import os
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv, find_dotenv
import openai

# Load API key
load_dotenv(find_dotenv())

# Get API key
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

# assign model id
model_id = 'gpt-4'

def chatgpt_conversation(conversation_Log):
    response = openai.ChatCompletion.create(
        model=model_id,
        messages=conversation_Log
    )
    # extract the role and content from the response message
    role = response.choices[0].message['role']
    content = response.choices[0].message['content']
    
    conversation_Log.append({
        'role': role.strip(),
        'content': content.strip(),
    })
    return conversation_Log  # get the convo log

conversations = []
# roles = system, user, assistant
conversations.append({
    'role': 'system', 
    'content': 'Hey sudz, I am a ServiceNow Certified Master Architect Assistant. I am here to help you manage your daily objectives and generate your daily accomplishments. Please input your objectives, each followed by a new line.'
})
conversations = chatgpt_conversation(conversations)
print('{0}: {1}\n'.format(conversations[-1]['role'].strip(), conversations[-1]['content'].strip()))

while True:
    prompt = input('User:')
    if prompt.strip().lower() == 'quit':
        break
    conversations.append({'role': 'user', 'content': prompt})
    conversations = chatgpt_conversation(conversations)
    print('{0}: {1}\n'.format(conversations[-1]['role'].strip(), conversations[-1]['content'].strip()))

    # Check if the user input is daily objectives
    if conversations[-1]['role'] == 'user':
        objectives = conversations[-1]['content'].split('\n')  # split the objectives by new line
        for obj in objectives:
            # Generate a grammar-corrected objective and an accomplishment for each objective
            conversations.append({'role': 'user', 'content': f'Correct the grammar: {obj}'})
            conversations = chatgpt_conversation(conversations)
            corrected_obj = conversations[-1]['content'].strip()
            print(f'Corrected Objective: {corrected_obj}\n')

            conversations.append({'role': 'user', 'content': f'Generate accomplishment: {corrected_obj}'})
            conversations = chatgpt_conversation(conversations)
            accomplishment = conversations[-1]['content'].strip()
            print(f'Accomplishment: {accomplishment}\n')
