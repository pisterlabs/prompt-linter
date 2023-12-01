import openai
from os import getenv
from dotenv import load_dotenv
import openai

load_dotenv()
# OpenAI API key
openai.api_key = getenv("OPENAI_API_KEY")
gpt3apikey = openai.api_key
completion = openai.Completion()

# Fichero Memoria.
f1 = open (f'config/memoria.data','r')
start_chat_log = f1.read()

# Consulta a GPT-3.
async def gpt3_consulta(nombre_bot, nombre_persona, question, chat_log=None):
    # Secuencias de inicio y reinicio para GPT-3.
    start_sequence = f"\n{nombre_bot}: "
    restart_sequence = f"\n\n{nombre_persona}: "
    if chat_log is None:
        chat_log = start_chat_log
    prompt = f'{chat_log}{restart_sequence}: {question}{start_sequence}:'
    response = completion.create(
        prompt=prompt, 
        # Explicación Modelos: https://beta.openai.com/docs/engines
        engine="davinci", # davinci, # curie, # babbage, # ada, # davinci-instruct-beta, # curie-instruct-beta, # davinci-codex, # cushman-codex
        stop=['\n'], 
        temperature=0.7,
        top_p=1, 
        frequency_penalty=0, 
        presence_penalty=0.3,
        max_tokens=90)
    answer = response.choices[0].text.strip()
    return answer