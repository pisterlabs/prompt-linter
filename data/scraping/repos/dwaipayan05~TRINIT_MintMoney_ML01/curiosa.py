import openai
import os

from dotenv import load_dotenv

openai.api_key = os.getenv('OPENAI_API_KEY')
completion = openai.Completion()
model = os.getenv('GPT3_FINE_TUNE_MODEL')

print(model)
start_prompt = '''
Hi ! The following is a conversation with Nerdo, an AI-Powered Bot trained on an open source dataset of COVID-19 Academia Knowledge. Nerdo is helpful, creative, clever, friendly and has a good grasp on concept related to coronavirus.

User: How has the number  of  childhood pneumonia been reduced?\n\n###\n\n
Bot: New conjugate vaccines against Haemophilus influenzae type b and Streptococcus pneumoniae have contributed to decreases in radiologic, clinical and complicated pneumonia cases. \n\n###\n\n
User: What is the treatment for MERS-COV?\n\n###\n\n
Bot: There is no specific treatment for MERS-CoV. Like most viral infections, the treatment options are supportive and symptomatic.\n\n###\n\n
User: What causes Q fever?\n\n###\n\n
Bot: Coxiella burnetii (C. burnetii) causes Q fever. \n\n####\n\n
User: Thank you for the help!\n\n###\n\n
Bot: You're welcome, come back anytime!'''

start_sequence = "\n\n###\n\nBot: "
restart_sequence = "\n\n###\n\nUser: "

def send_text(incoming_text, chat_log=None):
    if chat_log is None:
        chat_log = start_prompt
    prompt = f'{chat_log}{restart_sequence} {incoming_text}{start_sequence}'
    response = completion.create(
        prompt=prompt, 
        model=model, 
        stop=['\n\n###\n\n'], 
        temperature=0.3,
        frequency_penalty=1, 
        presence_penalty=1, 
        max_tokens=350)
    story = response['choices'][0]['text']
    return str(story)

def append_chat_log(sent_text, incoming_text, chat_log=None):
    if chat_log is None:
        chat_log = start_prompt
    return f'{chat_log}{restart_sequence} {sent_text}{start_sequence} {incoming_text}'


